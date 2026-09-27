import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "OpenAI.H3AutoFinalizePreviewV2.1.2",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "H3AutoFinalizeConcatPreviewV21") return;

        const oldCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = oldCreated?.apply(this, arguments);

            const root = document.createElement("div");
            root.style.cssText =
                "width:100%;box-sizing:border-box;padding:8px;background:#10151c;" +
                "border:1px solid #50657d;border-radius:7px;min-height:82px;";

            const banner = document.createElement("div");
            banner.textContent = "H3 AutoFinalize Preview V2.1.2 已加载｜等待本次 N 段生成完成…";
            banner.style.cssText = "font-size:12px;color:#d9e6f2;margin-bottom:7px;";

            const video = document.createElement("video");
            video.controls = true;
            video.playsInline = true;
            video.preload = "metadata";
            video.style.cssText =
                "display:none;width:100%;max-height:520px;background:#000;border-radius:5px;";

            const info = document.createElement("div");
            info.textContent = "左侧“生成段数”可选 1~10；第 N 段保存后自动拼接并在这里显示。";
            info.style.cssText =
                "font-size:11px;color:#9fb3c8;margin-top:6px;word-break:break-all;";

            const downloadButton = document.createElement("button");
            downloadButton.textContent = "保存 / 下载最终视频到电脑";
            downloadButton.disabled = true;
            downloadButton.style.cssText =
                "display:block;width:100%;margin-top:8px;padding:8px 10px;" +
                "border:1px solid #5d7896;border-radius:6px;" +
                "background:#26384a;color:#e8f1f8;font-size:12px;" +
                "cursor:not-allowed;opacity:.55;";

            let currentDownload = null;

            downloadButton.addEventListener("click", async () => {
                if (!currentDownload?.url) return;

                downloadButton.disabled = true;
                downloadButton.textContent = "正在准备下载…";

                try {
                    const response = await fetch(currentDownload.url, {
                        method: "GET",
                        credentials: "same-origin",
                    });
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }

                    const blob = await response.blob();
                    const objectUrl = URL.createObjectURL(blob);

                    const a = document.createElement("a");
                    a.href = objectUrl;
                    a.download = currentDownload.filename || "minimax_h3_final.mp4";
                    a.style.display = "none";
                    document.body.appendChild(a);
                    a.click();
                    a.remove();

                    setTimeout(() => URL.revokeObjectURL(objectUrl), 30000);

                    downloadButton.textContent = "下载已开始";
                    setTimeout(() => {
                        downloadButton.textContent = "保存 / 下载最终视频到电脑";
                        downloadButton.disabled = false;
                    }, 1500);
                } catch (error) {
                    console.error("[H3 AutoFinalize V2.1.2] download failed", error);
                    downloadButton.textContent = "下载失败，点击重试";
                    downloadButton.disabled = false;
                }
            });

            root.appendChild(banner);
            root.appendChild(video);
            root.appendChild(info);
            root.appendChild(downloadButton);

            const widget = this.addDOMWidget("h3_autofinal_preview_v21", "preview", root, {
                serialize: false,
                hideOnZoom: false,
            });
            widget.computeSize = (width) => [
                Math.max(300, width),
                video.style.display === "none" ? 92 : 430,
            ];

            this.__h3AutoFinalizeV21 = {
                root,
                banner,
                video,
                info,
                widget,
                downloadButton,
                setDownloadTarget: (url, filename) => {
                    currentDownload = { url, filename };
                    downloadButton.disabled = false;
                    downloadButton.style.cursor = "pointer";
                    downloadButton.style.opacity = "1";
                    downloadButton.textContent = "保存 / 下载最终视频到电脑";
                },
            };
            return r;
        };

        const oldExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            oldExecuted?.apply(this, arguments);

            const ui = this.__h3AutoFinalizeV21;
            if (!ui) return;

            const item = message?.gifs?.[0] ?? message?.ui?.gifs?.[0];
            if (!item) {
                ui.banner.textContent = "V2.1.2 已执行，但没有收到最终视频预览数据";
                ui.banner.style.color = "#ffbd66";
                this.graph?.setDirtyCanvas(true, true);
                return;
            }

            const params = new URLSearchParams({
                filename: item.filename || "",
                subfolder: item.subfolder || "",
                type: item.type || "output",
                t: String(Date.now()),
            });

            const src = api.apiURL(`/view?${params.toString()}`);
            ui.banner.textContent = "最终视频已完成｜可直接播放";
            ui.banner.style.color = "#9ee493";
            ui.info.textContent = item.fullpath || `${item.subfolder || ""}/${item.filename || ""}`;

            ui.video.onerror = () => {
                ui.banner.textContent = "最终视频已生成，但浏览器播放器加载失败";
                ui.banner.style.color = "#ff7b7b";
                ui.info.textContent = `${ui.info.textContent} ｜ URL=${src}`;
                this.graph?.setDirtyCanvas(true, true);
            };

            ui.video.src = src;
            ui.video.style.display = "block";
            ui.video.load();

            ui.setDownloadTarget?.(
                src,
                item.filename || "minimax_h3_final.mp4"
            );

            if (typeof this.setSize === "function") {
                this.setSize([this.size[0], Math.max(this.size[1], 760)]);
            }
            this.graph?.setDirtyCanvas(true, true);
        };
    },
});
