import gc
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import wave
from pathlib import Path

import numpy as np
import torch

import folder_paths
import nodes as comfy_nodes
from comfy import model_management
from comfy.utils import ProgressBar


class H3AutoFinalizeConcatPreviewV2:
    """Low-VRAM recovery for MiniMax H3 AV latent checkpoints.

    Loads one saved AV latent at a time via the installed MiniMaxH3LoadLatent
    node, decodes it with tiled VAE decode, trims continuation overlap for clips
    after the first, writes a temporary H.264/AAC MP4, releases memory, then
    losslessly concatenates all temporary MP4 files with FFmpeg stream copy.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                # Required on purpose: the node cannot run unless the final
                # Save AV Latent node has really executed and returned its path.
                "finalize_trigger": ("STRING", {"forceInput": True}),
                # Visible build marker so a screenshot immediately proves which
                # backend schema is actually loaded.
                "plugin_build": ("STRING", {"default": "H3 AutoFinalize V2.1.2", "multiline": False}),
                "latent_folder": ("STRING", {"default": "h3_context"}),
                "first_clip": ("INT", {"default": 1, "min": 1, "max": 99999, "step": 1}),
                "last_clip": ("INT", {"default": 10, "min": 1, "max": 99999, "step": 1}),
                "overlap_frames": ("INT", {"default": 22, "min": 0, "max": 1000, "step": 1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "h3_final/minimax_h3_1-10_final"}),
                "tile_size": ("INT", {"default": 384, "min": 128, "max": 2048, "step": 32}),
                "spatial_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 16}),
                "temporal_size": ("INT", {"default": 16, "min": 8, "max": 512, "step": 4}),
                "temporal_overlap": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1}),
                "keep_temp_segments": ("BOOLEAN", {"default": False}),
                "clear_vram_each_clip": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "VHS_FILENAMES")
    RETURN_NAMES = ("final_video_path", "status", "final_video")
    FUNCTION = "recover"
    CATEGORY = "MiniMax H3/Recovery"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "AUTO FINALIZE V2.1.2 — waits for the final Save AV Latent trigger, then "
        "Decodes only one clip at a time, trims overlap, writes it to disk, "
        "frees memory, and finally losslessly concatenates all clips with FFmpeg."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # NaN is ComfyUI's conventional "always changed" sentinel.
        # This node performs file I/O and should never be skipped from cache.
        return float("NaN")

    @staticmethod
    def _resolve_ffmpeg():
        """Resolve the same FFmpeg binary that VideoHelperSuite can use.

        Cloud ComfyUI images (including RunningHub-like environments) often do
        not expose ffmpeg on PATH even though VideoHelperSuite has a bundled
        imageio-ffmpeg executable.  v1.0.0 only checked PATH plus two module
        names, which was too narrow.
        """
        checked = []

        def accept(candidate, source):
            if not candidate:
                return None
            try:
                candidate = os.fspath(candidate)
            except Exception:
                return None
            checked.append(f"{source}={candidate}")
            # A module can expose just 'ffmpeg' rather than an absolute path.
            resolved = shutil.which(candidate) or candidate
            if not os.path.isfile(resolved):
                return None
            try:
                probe = subprocess.run(
                    [resolved, "-version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                if probe.returncode == 0:
                    return resolved
            except Exception:
                return None
            return None

        # 1) Explicit environment overrides.
        for env_name in ("FFMPEG_PATH", "FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
            hit = accept(os.environ.get(env_name), f"env:{env_name}")
            if hit:
                return hit

        # 2) Normal PATH.
        hit = accept(shutil.which("ffmpeg"), "PATH")
        if hit:
            return hit

        # 3) VideoHelperSuite is already imported by ComfyUI.  Its exact package
        # name varies with the custom_nodes folder name, so scan loaded modules
        # instead of assuming a Python import name.
        for mod_name, mod in list(sys.modules.items()):
            lname = mod_name.lower()
            if "videohelpersuite" not in lname or not lname.endswith("utils"):
                continue
            try:
                hit = accept(getattr(mod, "ffmpeg_path", None), f"module:{mod_name}.ffmpeg_path")
                if hit:
                    return hit
            except Exception:
                pass

        # 4) Try common import aliases used by VHS installs.
        try:
            import importlib
            for name in (
                "videohelpersuite.utils",
                "comfyui_videohelpersuite.videohelpersuite.utils",
                "ComfyUI-VideoHelperSuite.videohelpersuite.utils",
                "comfyui-videohelpersuite.videohelpersuite.utils",
            ):
                try:
                    mod = importlib.import_module(name)
                    hit = accept(getattr(mod, "ffmpeg_path", None), f"import:{name}.ffmpeg_path")
                    if hit:
                        return hit
                except Exception:
                    pass
        except Exception:
            pass

        # 5) Most important cloud fallback: VideoHelperSuite declares
        # imageio-ffmpeg as a dependency, and that package ships/locates a usable
        # ffmpeg binary even when /usr/bin/ffmpeg is absent from PATH.
        try:
            import imageio_ffmpeg
            hit = accept(imageio_ffmpeg.get_ffmpeg_exe(), "imageio_ffmpeg")
            if hit:
                return hit
        except Exception as e:
            checked.append(f"imageio_ffmpeg_error={type(e).__name__}:{e}")

        # 6) Typical Linux/conda locations as a final fallback.
        for candidate in (
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/conda/bin/ffmpeg",
            "/root/miniconda3/bin/ffmpeg",
            "/root/miniconda3/envs/app/bin/ffmpeg",
        ):
            hit = accept(candidate, "common_path")
            if hit:
                return hit

        details = "; ".join(checked[-12:]) if checked else "no candidates found"
        raise RuntimeError(
            "FFmpeg could not be resolved. This recovery node now checks PATH, "
            "VideoHelperSuite's loaded ffmpeg_path, imageio-ffmpeg, and common "
            f"conda/Linux locations. Checked: {details}"
        )

    @staticmethod
    def _safe_output_path(filename_prefix: str) -> Path:
        output_root = Path(folder_paths.get_output_directory()).resolve()
        raw = filename_prefix.strip().replace("\\", "/") or "h3_recovery/recovered_merged"
        raw = raw[:-4] if raw.lower().endswith(".mp4") else raw
        target = (output_root / raw).resolve()
        try:
            target.relative_to(output_root)
        except ValueError:
            raise ValueError("filename_prefix must stay inside ComfyUI/output")
        target.parent.mkdir(parents=True, exist_ok=True)
        candidate = target.with_suffix(".mp4")
        if not candidate.exists():
            return candidate
        counter = 1
        while True:
            candidate = target.parent / f"{target.name}_{counter:03d}.mp4"
            if not candidate.exists():
                return candidate
            counter += 1

    @staticmethod
    def _call_load_latent(latent_folder: str, clip_index: int):
        load_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3LoadLatent")
        if load_cls is None:
            raise RuntimeError(
                "MiniMaxH3LoadLatent is not installed. Install/enable the MiniMax H3 Extender node pack first."
            )
        loader = load_cls()
        fn_name = getattr(load_cls, "FUNCTION", None)
        if not fn_name:
            raise RuntimeError("MiniMaxH3LoadLatent has no FUNCTION declaration")
        fn = getattr(loader, fn_name)
        # The extenders' loader accepts optional VAEs. Passing none prevents it
        # from decoding IMAGE/AUDIO, which is the critical low-VRAM behavior.
        try:
            result = fn(latent_path=latent_folder, clip_index=int(clip_index), vae=None, audio_vae=None)
        except TypeError:
            # Compatibility with versions that name the folder argument positionally.
            result = fn(latent_folder, int(clip_index), None, None)
        if isinstance(result, dict) and "result" in result:
            result = result["result"]
        if not isinstance(result, (tuple, list)) or not result:
            raise RuntimeError(f"MiniMaxH3LoadLatent returned an unexpected value for clip {clip_index}")
        latent = result[0]
        if not isinstance(latent, dict) or "samples" not in latent:
            raise RuntimeError(f"clip {clip_index}: loaded object is not a LATENT dictionary")
        return latent

    @staticmethod
    def _split_av_samples(latent):
        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            parts = samples.unbind()
            if len(parts) < 2:
                return parts[0], None
            return parts[0], parts[-1]
        # Some older checkpoints can be video-only.
        return samples, None

    @staticmethod
    def _decode_video_tiled(vae, samples, tile_size, spatial_overlap, temporal_size, temporal_overlap):
        """Decode video without bypassing ComfyUI's VAE wrapper.

        MiniMax H3's native VAE declares ``handles_tiling=True`` and performs its
        own 256px spatial tiling plus temporal chunk streaming.  Calling
        ``comfy.utils.tiled_scale_multidim`` directly (v1.0.0/v1.0.1) bypassed
        that model-owned path and could feed H3's non-linear temporal upscale
        formula into the generic tiler.  On ComfyUI 0.33.x this can produce
        ``narrow(): length must be non-negative``.

        For H3/model-owned tilers we therefore call the public ``vae.decode``
        API and let ComfyUI/native H3 manage VRAM.  Other VAEs use the exact
        public ``vae.decode_tiled`` path used by ComfyUI's VAEDecodeTiled node.
        """
        vae.throw_exception_if_invalid()

        model = getattr(vae, "first_stage_model", None)
        model_name = type(model).__name__ if model is not None else ""
        handles_tiling = bool(getattr(vae, "handles_tiling", False))
        is_h3 = model_name == "MiniMaxH3VideoVAE" or "MiniMaxH3" in model_name

        # MiniMax H3 has its own temporal chunking and spatial tiler. Its decode
        # streams finalized chunks to ComfyUI's intermediate device, so this is
        # already the intended low-VRAM path.
        if handles_tiling or is_h3:
            decoded = vae.decode(samples)
        else:
            tile_size = int(tile_size)
            spatial_overlap = int(spatial_overlap)
            temporal_size = int(temporal_size)
            temporal_overlap = int(temporal_overlap)

            if tile_size < spatial_overlap * 4:
                spatial_overlap = tile_size // 4
            if temporal_size < temporal_overlap * 2:
                temporal_overlap = temporal_overlap // 2

            temporal_compression = vae.temporal_compression_decode()
            if temporal_compression is not None:
                temporal_size = max(2, temporal_size // int(temporal_compression))
                temporal_overlap = max(
                    1,
                    min(temporal_size // 2, temporal_overlap // int(temporal_compression)),
                )
            else:
                temporal_size = None
                temporal_overlap = None

            compression = vae.spacial_compression_decode()
            decoded = vae.decode_tiled(
                samples,
                tile_x=tile_size // compression,
                tile_y=tile_size // compression,
                overlap=spatial_overlap // compression,
                tile_t=temporal_size,
                overlap_t=temporal_overlap,
            )

        if len(decoded.shape) == 5:
            decoded = decoded.reshape(
                -1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1]
            )
        return decoded

    @staticmethod
    def _decode_audio(audio_vae, audio_samples):
        if audio_samples is None:
            return None
        audio = audio_vae.decode(audio_samples)
        # New ComfyUI VAE wrappers return channels-last (BTC); older raw AudioVAE
        # implementations returned channels-first (BCT).
        if hasattr(audio_vae, "first_stage_model"):
            audio = audio.movedim(-1, 1)
        sample_rate = getattr(
            audio_vae,
            "audio_sample_rate_output",
            getattr(audio_vae, "output_sample_rate", None),
        )
        if sample_rate is None:
            sample_rate = getattr(
                getattr(audio_vae, "first_stage_model", None),
                "output_sample_rate",
                44100,
            )
        return {"waveform": audio.detach().float().cpu(), "sample_rate": int(sample_rate)}

    @staticmethod
    def _trim_audio(audio, trim_frames, remaining_frames, fps):
        if audio is None:
            return None
        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        start = int(round(float(trim_frames) / float(fps) * sr))
        wanted = int(round(float(remaining_frames) / float(fps) * sr))
        waveform = waveform[..., start:start + wanted]
        if waveform.shape[-1] < wanted:
            pad = wanted - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return {"waveform": waveform.contiguous(), "sample_rate": sr}

    @staticmethod
    def _write_wav(audio, wav_path: Path):
        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])
        if waveform.ndim == 3:
            waveform = waveform[0]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        # [channels, samples] -> [samples, channels]
        pcm = torch.clamp(waveform, -1.0, 1.0).mul(32767.0).round().to(torch.int16)
        pcm_np = pcm.transpose(0, 1).contiguous().numpy()
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(int(pcm_np.shape[1]))
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_np.tobytes())

    @staticmethod
    def _encode_segment(ffmpeg, images, audio, fps, crf, out_path: Path, work_dir: Path):
        images = torch.clamp(images.detach().float().cpu(), 0.0, 1.0)
        if images.ndim != 4 or images.shape[0] < 1:
            raise RuntimeError("Decoded clip contains no video frames")
        height = int(images.shape[1])
        width = int(images.shape[2])

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s:v", f"{width}x{height}",
            "-r", f"{float(fps):.8f}",
            "-i", "pipe:0",
        ]

        wav_path = None
        if audio is not None:
            wav_path = work_dir / (out_path.stem + ".wav")
            H3AutoFinalizeConcatPreviewV2._write_wav(audio, wav_path)
            cmd += ["-i", str(wav_path)]

        cmd += [
            "-map", "0:v:0",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p",
            "-g", str(max(1, int(round(float(fps))))),
        ]
        if audio is not None:
            cmd += [
                "-map", "1:a:0",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", str(int(audio["sample_rate"])),
                "-shortest",
            ]
        else:
            cmd += ["-an"]
        cmd += ["-movflags", "+faststart", str(out_path)]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for frame in images:
                arr = frame.mul(255.0).round().to(torch.uint8).numpy()
                proc.stdin.write(arr.tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            code = proc.wait()
        finally:
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            except Exception:
                pass
        if code != 0:
            raise RuntimeError(f"FFmpeg segment encode failed ({out_path.name}): {stderr[-3000:]}")
        if wav_path is not None:
            try:
                wav_path.unlink()
            except OSError:
                pass

    @staticmethod
    def _concat_segments(ffmpeg, segments, final_path: Path, work_dir: Path):
        concat_file = work_dir / "concat.txt"
        with concat_file.open("w", encoding="utf-8") as f:
            for path in segments:
                # ffconcat uses single quotes; escape embedded single quotes defensively.
                escaped = str(path.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-movflags", "+faststart", str(final_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {res.stderr[-4000:]}")

    @staticmethod
    def _clear_vram():
        gc.collect()
        try:
            model_management.soft_empty_cache()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _prepare_for_finalize():
        gc.collect()
        try:
            model_management.unload_all_models()
        except Exception:
            pass
        try:
            model_management.soft_empty_cache()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def recover(
        self,
        video_vae,
        audio_vae,
        latent_folder,
        first_clip,
        last_clip,
        overlap_frames,
        fps,
        filename_prefix,
        tile_size,
        spatial_overlap,
        temporal_size,
        temporal_overlap,
        crf,
        keep_temp_segments,
        clear_vram_each_clip,
        finalize_trigger,
        plugin_build="H3 AutoFinalize V2.1.2",
    ):
        first_clip = int(first_clip)
        last_clip = int(last_clip)
        if last_clip < first_clip:
            raise ValueError("last_clip must be >= first_clip")
        if not finalize_trigger or not str(finalize_trigger).strip():
            raise RuntimeError(
                "AUTO FINALIZE V2.1.2: finalize_trigger is empty. "
                "Connect the LAST MiniMaxH3SaveLatent.latent_path output."
            )

        print(
            f"[H3 AutoFinalize V2.1.2] START: trigger={finalize_trigger!s}, "
            f"clips={first_clip}-{last_clip}"
        )

        # In formal one-click mode finalize_trigger comes from the final
        # MiniMaxH3SaveLatent.latent_path output, so reaching this line means the
        # last AV latent has already been committed to disk.
        self._prepare_for_finalize()

        ffmpeg = self._resolve_ffmpeg()
        final_path = self._safe_output_path(filename_prefix)
        output_root = Path(folder_paths.get_output_directory()).resolve()
        temp_root = output_root / "h3_recovery" / f".oneclick_tmp_{uuid.uuid4().hex[:8]}"
        temp_root.mkdir(parents=True, exist_ok=True)

        # Cloud filesystems can lag slightly after the final safetensors rename.
        # Wait briefly and report exactly what is missing instead of silently
        # completing without ever entering the recovery phase.
        import time
        expected_missing = []
        for _attempt in range(20):
            expected_missing = []
            base = output_root / latent_folder
            for idx in range(first_clip, last_clip + 1):
                exact = base / f"clip_{idx:05d}.safetensors"
                if not exact.exists():
                    # support custom prefixes used by the H3 saver
                    matches = list(base.glob(f"*_{idx:05d}.safetensors"))
                    if not matches:
                        expected_missing.append(idx)
            if not expected_missing:
                break
            time.sleep(0.25)

        if expected_missing:
            present = sorted(p.name for p in (output_root / latent_folder).glob("*.safetensors"))
            raise FileNotFoundError(
                "AUTO FINALIZE V2.1.2 was triggered, but expected AV latent files are missing: "
                f"{expected_missing}. Present files: {present}"
            )

        segments = []
        pbar = ProgressBar(last_clip - first_clip + 1)
        status_lines = []
        success = False
        try:
            for clip_index in range(first_clip, last_clip + 1):
                latent = self._call_load_latent(latent_folder, clip_index)
                video_samples, audio_samples = self._split_av_samples(latent)

                images = self._decode_video_tiled(
                    video_vae,
                    video_samples,
                    tile_size,
                    spatial_overlap,
                    temporal_size,
                    temporal_overlap,
                )
                audio = self._decode_audio(audio_vae, audio_samples)

                trim = 0 if clip_index == first_clip else int(overlap_frames)
                if trim >= int(images.shape[0]):
                    raise ValueError(
                        f"clip {clip_index}: overlap_frames={trim} is not smaller than decoded frame count={images.shape[0]}"
                    )
                if trim:
                    images = images[trim:]
                audio = self._trim_audio(audio, trim, int(images.shape[0]), fps)

                segment_path = temp_root / f"clip_{clip_index:05d}.mp4"
                self._encode_segment(ffmpeg, images, audio, fps, crf, segment_path, temp_root)
                segments.append(segment_path)
                status_lines.append(
                    f"clip {clip_index}: {int(images.shape[0])} delivered frames"
                    + (f" (trimmed {trim})" if trim else "")
                )

                del latent, video_samples, audio_samples, images, audio
                if clear_vram_each_clip:
                    self._clear_vram()
                pbar.update(1)

            self._concat_segments(ffmpeg, segments, final_path, temp_root)
            success = True

            if keep_temp_segments:
                keep_dir = final_path.parent / (final_path.stem + "_segments")
                keep_dir.mkdir(parents=True, exist_ok=True)
                for segment in segments:
                    shutil.copy2(segment, keep_dir / segment.name)

            status = (
                f"Recovered clips {first_clip}-{last_clip}; final: {final_path}; "
                f"overlap trim={int(overlap_frames)} frames; tiled decode temporal_size={int(temporal_size)}, tile_size={int(tile_size)}; "
                f"ffmpeg={ffmpeg}; auto_finalize_trigger={bool(finalize_trigger)}.\n"
                + "\n".join(status_lines)
            )
            # Expose the final MP4 directly in the node UI using the same
            # ``ui.gifs`` payload that VideoHelperSuite uses for video previews.
            # This keeps the low-VRAM disk concat path while still giving the
            # user an in-workflow playable final movie.
            rel = final_path.resolve().relative_to(output_root)
            subfolder = "" if str(rel.parent) == "." else str(rel.parent).replace(os.sep, "/")
            preview = {
                "filename": final_path.name,
                "subfolder": subfolder,
                "type": "output",
                "format": "video/h264-mp4",
                "frame_rate": float(fps),
                "fullpath": str(final_path),
            }
            print(f"[H3 AutoFinalize V2.1.2] COMPLETE: {final_path}")
            vhs_files = (True, [str(final_path)])
            return {
                # Exact key/shape used by VHS_VideoCombine.
                "ui": {
                    "gifs": [preview],
                    "text": [f"H3 AutoFinalize V2.1.2 COMPLETE: {final_path}"],
                },
                "result": (str(final_path), status, vhs_files),
            }
        finally:
            # Temp working files are never needed after the final/copy stage.
            # When keep_temp_segments=True, clean copies have already been placed
            # next to the final movie under *_segments/.
            shutil.rmtree(temp_root, ignore_errors=True)
            self._clear_vram()


class H3SegmentCountTriggerV21:
    """Select how many clips to generate (1-10) with true lazy evaluation.

    Only the selected clip_N_path input is requested. Because every later H3
    segment depends on the earlier ones, selecting N naturally executes clips
    1..N and leaves clips N+1..10 dormant.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 11):
            optional[f"clip_{i}_path"] = (
                "STRING",
                {
                    "forceInput": True,
                    "lazy": True,
                    "tooltip": f"片段{i} Save AV Latent 的 latent_path",
                },
            )
        return {
            "required": {
                "run_segments": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "display": "number",
                        "tooltip": "只改这个数字：1=只跑片段1，3=跑1~3，10=跑1~10",
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("finalize_trigger", "last_clip")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Recovery"
    DESCRIPTION = "选择本次生成段数；只执行 1..N，并把第 N 段保存完成作为自动最终拼接触发器。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always reconsider the lazy selected input on every queue.
        return float("NaN")

    def check_lazy_status(self, run_segments, **kwargs):
        n = max(1, min(10, int(run_segments)))
        key = f"clip_{n}_path"
        if kwargs.get(key) is None:
            return [key]
        return []

    def select(self, run_segments, **kwargs):
        n = max(1, min(10, int(run_segments)))
        key = f"clip_{n}_path"
        path = kwargs.get(key)
        if not path:
            raise RuntimeError(
                f"H3 AutoFinalize V2.1.2: 选择了 {n} 段，但 {key} 没有拿到保存路径。"
            )
        print(f"[H3 AutoFinalize V2.1.2] run_segments={n}; trigger={path}")
        return (str(path), n)


class H3AutoFinalizeConcatPreviewV21(H3AutoFinalizeConcatPreviewV2):
    """V2.1 finalizer driven by H3SegmentCountTriggerV21."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "finalize_trigger": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "由“生成段数选择器”自动连接，不需要手动改。",
                    },
                ),
                "last_clip": (
                    "INT",
                    {
                        "forceInput": True,
                        "min": 1,
                        "max": 10,
                        "tooltip": "由“生成段数选择器”自动传入。",
                    },
                ),
                "plugin_build": (
                    "STRING",
                    {
                        "default": "H3 AutoFinalize V2.1.2",
                        "multiline": False,
                    },
                ),
                "latent_folder": ("STRING", {"default": "h3_context"}),
                "first_clip": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "overlap_frames": ("INT", {"default": 22, "min": 0, "max": 1000, "step": 1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "filename_prefix_base": (
                    "STRING",
                    {
                        "default": "h3_final/minimax_h3",
                        "tooltip": "最终文件会自动命名为 *_1-N_final.mp4",
                    },
                ),
                "tile_size": ("INT", {"default": 384, "min": 128, "max": 2048, "step": 32}),
                "spatial_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 16}),
                "temporal_size": ("INT", {"default": 16, "min": 8, "max": 512, "step": 4}),
                "temporal_overlap": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1}),
                "keep_temp_segments": ("BOOLEAN", {"default": False}),
                "clear_vram_each_clip": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "VHS_FILENAMES")
    RETURN_NAMES = ("final_video_path", "status", "final_video")
    FUNCTION = "recover_v21"
    CATEGORY = "MiniMax H3/Recovery"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "AutoFinalize V2.1.0：生成段数由左侧选择器决定。"
        "第N段保存完成后自动恢复 1..N、裁重叠、磁盘拼接并显示最终视频。"
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def recover_v21(
        self,
        video_vae,
        audio_vae,
        finalize_trigger,
        last_clip,
        plugin_build,
        latent_folder,
        first_clip,
        overlap_frames,
        fps,
        filename_prefix_base,
        tile_size,
        spatial_overlap,
        temporal_size,
        temporal_overlap,
        crf,
        keep_temp_segments,
        clear_vram_each_clip,
    ):
        first_clip = int(first_clip)
        last_clip = int(last_clip)
        if not (1 <= last_clip <= 10):
            raise ValueError(f"last_clip must be 1..10, got {last_clip}")
        if first_clip > last_clip:
            raise ValueError("first_clip must be <= selected segment count")

        base = str(filename_prefix_base).rstrip("_-/ ")
        filename_prefix = f"{base}_{first_clip}-{last_clip}_final"

        return super().recover(
            video_vae=video_vae,
            audio_vae=audio_vae,
            latent_folder=latent_folder,
            first_clip=first_clip,
            last_clip=last_clip,
            overlap_frames=overlap_frames,
            fps=fps,
            filename_prefix=filename_prefix,
            tile_size=tile_size,
            spatial_overlap=spatial_overlap,
            temporal_size=temporal_size,
            temporal_overlap=temporal_overlap,
            crf=crf,
            keep_temp_segments=keep_temp_segments,
            clear_vram_each_clip=clear_vram_each_clip,
            finalize_trigger=finalize_trigger,
            plugin_build=plugin_build,
        )


NODE_CLASS_MAPPINGS = {
    "H3SegmentCountTriggerV21": H3SegmentCountTriggerV21,
    "H3AutoFinalizeConcatPreviewV21": H3AutoFinalizeConcatPreviewV21,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SegmentCountTriggerV21": "MiniMax H3 生成段数选择器 V2.1",
    "H3AutoFinalizeConcatPreviewV21": "MiniMax H3 自动最终拼接 + 视频预览 V2.1",
}
