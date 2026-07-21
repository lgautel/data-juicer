#!/usr/bin/env python3
"""从三路相机同名 mp4 发给 Qwen3.5 解释视频内容.

三路相机目录:
  observation.images.cam_high
  observation.images.cam_left_wrist
  observation.images.cam_right_wrist

Qwen3.5 / SGLang 接受的视频输入形式(见 README Video Input / sglang load_video):
  - https://... 远程 URL
  - file:///abs/path.mp4 本地文件
  - data:video/mp4;base64,... 内联 base64

本机原始视频多为 AV1; OpenCV 等解码器不稳定, 默认先转码为 H.264 mp4
再发送, 兼容性更好. 可用 --no-transcode 直接发送原文件.

用法(需先启动 SGLang 服务 ./serve_qwen35_sglang.sh):
  /mnt/r/VENV/dj/bin/python explain_triple_cam_videos.py
  /mnt/r/VENV/dj/bin/python explain_triple_cam_videos.py --episode 0 --no-thinking
  /mnt/r/VENV/dj/bin/python explain_triple_cam_videos.py --episode episode_000001.mp4 --url-mode base64
  /mnt/r/VENV/dj/bin/python explain_triple_cam_videos.py --dry-run --keep-transcoded /tmp/qwen35_vids
"""

from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

DEFAULT_VIDEO_ROOT = Path(
    "/mnt/r/DATA/pre_train_v1/post_train/stack_bowls_three/videos/chunk-000"
)
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
CAMERA_LABELS = {
    "observation.images.cam_high": "头顶相机(cam_high)",
    "observation.images.cam_left_wrist": "左手腕相机(cam_left_wrist)",
    "observation.images.cam_right_wrist": "右手腕相机(cam_right_wrist)",
}
DEFAULT_PROMPT = (
    "下面是同一机器人操作任务的三路同步相机视频, 顺序为:\n"
    "1) 头顶相机(cam_high)\n"
    "2) 左手腕相机(cam_left_wrist)\n"
    "3) 右手腕相机(cam_right_wrist)\n\n"
    "请结合这三段视频解释整个过程: 场景与物体如何摆放、机器人双臂如何协作、"
    "任务目标是什么、动作大致分几个阶段。用中文简洁回答。"
)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="三路相机同名视频发给 Qwen3.5 解释"
    )
    p.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help="chunk-000 目录路径",
    )
    p.add_argument(
        "--episode",
        default="episode_000000.mp4",
        help="同名视频文件名, 或纯数字索引如 0 / 3",
    )
    p.add_argument(
        "--url-mode",
        choices=("file", "base64"),
        default="file",
        help="发给模型的 URL 形式: file:// 本地路径, 或 data URI base64 (默认 file)",
    )
    p.add_argument(
        "--no-transcode",
        action="store_true",
        help="不转码, 直接使用原始 AV1 mp4 (默认会转 H.264)",
    )
    p.add_argument(
        "--keep-transcoded",
        type=Path,
        default=None,
        help="把转码后的 H.264 mp4 保留到该目录; 未指定则用临时目录并自动清理",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="传给模型的视频采样 fps (mm_processor_kwargs, 默认 2.0, 与 README 一致)",
    )
    p.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
    )
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="关闭思考模式(Instruct)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析/转码视频, 不请求模型",
    )
    return p.parse_args()


def resolve_episode_name(video_root: Path, episode: str) -> str:
    if episode.isdigit():
        name = f"episode_{int(episode):06d}.mp4"
    elif episode.endswith(".mp4"):
        name = episode
    else:
        name = f"{episode}.mp4"

    for cam in CAMERA_KEYS:
        path = video_root / cam / name
        if not path.exists():
            raise FileNotFoundError(f"三路相机缺少同名视频: {path}")
    return name


def probe_codec(path: Path) -> str:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return out or "unknown"


def transcode_to_h264(src: Path, dst: Path) -> None:
    """转成 Qwen/常见解码器更友好的 H.264 + yuv420p mp4."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"转码失败或输出为空: {src} -> {dst}")


def to_video_url(path: Path, mode: str) -> str:
    if mode == "file":
        return path.resolve().as_uri()  # file:///...
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def build_multimodal_content(prompt: str, videos: list[tuple[str, str]]) -> list[dict]:
    """videos: [(camera_key, video_url), ...]"""
    content: list[dict] = []
    for cam_key, url in videos:
        label = CAMERA_LABELS.get(cam_key, cam_key)
        content.append({"type": "text", "text": f"[{label}]"})
        content.append(
            {
                "type": "video_url",
                "video_url": {"url": url},
            }
        )
    content.append({"type": "text", "text": prompt})
    return content


def main() -> None:
    args = build_args()
    episode = resolve_episode_name(args.video_root, args.episode)
    do_transcode = not args.no_transcode

    tmp_dir: tempfile.TemporaryDirectory | None = None
    work_dir: Path
    if do_transcode:
        if args.keep_transcoded is not None:
            work_dir = args.keep_transcoded
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            tmp_dir = tempfile.TemporaryDirectory(prefix="qwen35_vid_")
            work_dir = Path(tmp_dir.name)
    else:
        work_dir = Path(".")

    try:
        prepared: list[tuple[str, Path]] = []
        for cam in CAMERA_KEYS:
            src = args.video_root / cam / episode
            codec = probe_codec(src)
            size_mb = src.stat().st_size / (1024 * 1024)
            print(f"[源] {cam}: {src.name} codec={codec} size={size_mb:.2f}MB")

            if do_transcode:
                dst = work_dir / f"{Path(episode).stem}__{cam.split('.')[-1]}__h264.mp4"
                if not dst.exists() or dst.stat().st_size == 0:
                    print(f"  -> 转码 H.264: {dst.name}")
                    transcode_to_h264(src, dst)
                else:
                    print(f"  -> 复用已有转码: {dst}")
                prepared.append((cam, dst))
            else:
                prepared.append((cam, src))

        video_urls: list[tuple[str, str]] = []
        for cam, path in prepared:
            url = to_video_url(path, args.url_mode)
            preview = url if len(url) < 120 else f"{url[:80]}...({len(url)} chars)"
            print(f"[URL] {cam}: mode={args.url_mode} -> {preview}")
            video_urls.append((cam, url))

        if args.dry_run:
            print("[dry-run] 跳过模型请求")
            return

        content = build_multimodal_content(args.prompt, video_urls)
        messages = [{"role": "user", "content": content}]

        if args.no_thinking:
            sampling = dict(temperature=0.7, top_p=0.8, presence_penalty=1.5)
            extra_body: dict = {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
                "mm_processor_kwargs": {"fps": args.fps, "do_sample_frames": True},
            }
        else:
            sampling = dict(temperature=1.0, top_p=0.95, presence_penalty=1.5)
            extra_body = {
                "top_k": 20,
                "mm_processor_kwargs": {"fps": args.fps, "do_sample_frames": True},
            }

        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        print(
            f"[请求] {args.base_url} model={args.model} "
            f"thinking={'off' if args.no_thinking else 'on'} fps={args.fps}"
        )

        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
            **sampling,
        )
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            print("\n[思考]\n" + reasoning)
        print("\n[回复]\n" + (msg.content or ""))
        print("\n[用量]", resp.usage)
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


if __name__ == "__main__":
    main()
