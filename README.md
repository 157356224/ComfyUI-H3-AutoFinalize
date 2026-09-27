ComfyUI-H3-AutoFinalize v2.1.2

核心变化
- 不再固定 10 段。
- 新增 “MiniMax H3 生成段数选择器 V2.1”。
- 只需要改 run_segments：1~10。
- 使用 ComfyUI lazy evaluation，只请求第 N 段 Save AV Latent 的 latent_path。
  因此正常链路会只执行 1..N，N+1..10 不会作为最终输出依赖被请求。
- 第 N 段保存完成后，自动最终节点恢复并拼接 1..N。
- 最终文件名自动带实际范围，例如：
    run_segments=1 -> minimax_h3_1-1_final.mp4
    run_segments=3 -> minimax_h3_1-3_final.mp4
    run_segments=10 -> minimax_h3_1-10_final.mp4
- 保留节点内最终视频播放器。

安装
1. 停止 ComfyUI。
2. 删除旧目录：
   /root/comfyui/custom_nodes/ComfyUI-H3-AutoFinalize
3. 解压本 ZIP，最终目录必须是：
   /root/comfyui/custom_nodes/ComfyUI-H3-AutoFinalize/
4. 重启 ComfyUI。
5. 浏览器强制刷新。
6. 导入 MiniMax H3 连续视频工作流 v1.4.2。

使用
只改一个数字：
“生成段数（1~10）｜只改这里”
例如 3，运行一次：
片段1 -> 片段2 -> 片段3 -> 自动最终拼接1~3 -> 节点内显示最终视频。


v2.1.2 hotfix:
- Fix stale H3OneClickRecoverConcat class reference in _encode_segment audio WAV path.
- Variable 1~10 segment selection logic is unchanged.


v2.1.2:
- 在最终视频预览节点内新增“保存 / 下载最终视频到电脑”按钮。
- 最终 MP4 生成后按钮自动启用。
- 点击后浏览器会把最终 MP4 下载到本机默认下载目录。
- 不改变 1~10 段可变生成、自动拼接、低显存恢复逻辑。
