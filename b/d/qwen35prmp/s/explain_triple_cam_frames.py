#!/usr/bin/env python3
"""从三路相机同名 mp4 抽取同一时间点帧, 发给 Qwen3.5 解释.

三路相机目录:
  observation.images.cam_high
  observation.images.cam_left_wrist
  observation.images.cam_right_wrist

用法(需先启动 SGLang 服务 ./serve_qwen35_sglang.sh):
  /mnt/r/VENV/dj/bin/python explain_triple_cam_frames.py
  /mnt/r/VENV/dj/bin/python explain_triple_cam_frames.py --episode episode_000000.mp4 --time 1.5
  /mnt/r/VENV/dj/bin/python explain_triple_cam_frames.py --episode 3 --frame-index 100 --no-thinking
  /mnt/r/VENV/dj/bin/python explain_triple_cam_frames.py --save-dir /tmp/qwen35_frames --dry-run
"""

from __future__ import annotations

import argparse
import base64
import os
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
    "observation.images.cam_high": "high camera",
    "observation.images.cam_left_wrist": "left wrist camera",
    "observation.images.cam_right_wrist": "right wrist camera",
}
DEFAULT_PROMPT = (
    "Below are three synchronized camera views of the same AGILE X robot manipulation task "
    "at the same moment, in the following order:\n"
    "1) high camera \n"
    "2) left wrist camera \n"
    "3) right wrist camera \n\n"
    "The manipulation task is 'Stack the blue bowl with slightly rounded base from the base up to the top'. "
    "Based on these three images, explain the current scene shortly."
    # "Based on these three images, explain the current scene: how the objects on "
    # "the table are arranged, the approximate pose of the robot's two arms, and "
    # "what action is likely being performed at this moment. Answer concisely."
)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="三路相机同名视频同时刻抽帧并交给 Qwen3.5 解释"
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
        help="同名视频文件名, 或纯数字索引如 0 / 3 (默认 episode_000000.mp4)",
    )
    p.add_argument(
        "--time",
        type=float,
        default=None,
        help="抽取时间点(秒)。与 --frame-index / --progress 三选一",
    )
    p.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="抽取帧号(从 0 开始)。与 --time / --progress 三选一",
    )
    p.add_argument(
        "--progress",
        type=float,
        default=0.5,
        help="相对进度 [0,1], 默认 0.5 (视频中点)。当未指定 --time/--frame-index 时生效",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="可选: 把抽出的三帧 JPEG 保存到该目录",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG 质量 1-100 (默认 90)",
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
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="关闭思考模式(Instruct)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只抽帧/保存, 不请求模型",
    )
    return p.parse_args()


def resolve_episode_name(video_root: Path, episode: str) -> str:
    if episode.isdigit():
        name = f"episode_{int(episode):06d}.mp4"
    elif episode.endswith(".mp4"):
        name = episode
    else:
        name = f"{episode}.mp4"

    high_dir = video_root / CAMERA_KEYS[0]
    path = high_dir / name
    if not path.exists():
        raise FileNotFoundError(f"找不到视频: {path}")
    for cam in CAMERA_KEYS[1:]:
        other = video_root / cam / name
        if not other.exists():
            raise FileNotFoundError(f"三路相机缺少同名视频: {other}")
    return name


def _ffprobe_value(path: Path, entries: str, stream: bool = True) -> str:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        entries,
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    if not stream:
        # duration 在 format 层更稳妥
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            entries,
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    if not out:
        raise RuntimeError(f"ffprobe 无输出: {path} entries={entries}")
    return out[0]


def probe_video(path: Path) -> tuple[float, float, int]:
    """返回 (fps, duration_sec, frame_count)。

    这些数据集视频多为 AV1, OpenCV 在本机解码不稳定, 统一用 ffprobe。
    """
    rate = _ffprobe_value(path, "stream=r_frame_rate")
    if "/" in rate:
        num, den = rate.split("/", 1)
        fps = float(num) / float(den)
    else:
        fps = float(rate)

    nframes_s = _ffprobe_value(path, "stream=nb_frames")
    try:
        nframes = int(nframes_s)
    except ValueError:
        nframes = 0

    duration_s = _ffprobe_value(path, "format=duration", stream=False)
    try:
        duration = float(duration_s)
    except ValueError:
        duration = (nframes / fps) if fps > 0 and nframes > 0 else 0.0

    if nframes <= 0 and fps > 0 and duration > 0:
        nframes = max(1, int(round(duration * fps)))
    if fps <= 0 or nframes <= 0:
        raise RuntimeError(f"视频元数据异常: {path} fps={fps} nframes={nframes}")
    return fps, duration, nframes


def resolve_frame_index(
    fps: float,
    duration: float,
    nframes: int,
    time_sec: float | None,
    frame_index: int | None,
    progress: float,
) -> tuple[int, float]:
    if frame_index is not None:
        idx = frame_index
    elif time_sec is not None:
        idx = int(round(time_sec * fps))
    else:
        if not (0.0 <= progress <= 1.0):
            raise ValueError("--progress 必须在 [0, 1]")
        idx = int(round(progress * (nframes - 1)))

    idx = max(0, min(idx, nframes - 1))
    t = idx / fps
    # 避免 seek 到末尾越界
    t = min(t, max(0.0, duration - 1e-3))
    return idx, t


def extract_frame_jpeg_bytes(path: Path, time_sec: float, quality: int = 90) -> bytes:
    """用 ffmpeg 按时间戳抽一帧 JPEG(AV1 兼容)。

    quality: 1-100, 映射到 ffmpeg -q:v (2最好, 31最差)。
    """
    q = max(2, min(31, int(round(31 - (quality / 100.0) * 29))))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        # -ss 放在 -i 前做输入 seek, 对三路同时间点抽帧更高效
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{time_sec:.6f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-q:v",
            str(q),
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        data = out_path.read_bytes()
        if not data:
            raise RuntimeError(f"ffmpeg 抽帧结果为空: {path} t={time_sec}")
        return data
    finally:
        out_path.unlink(missing_ok=True)


def encode_jpeg_base64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("ascii")


def build_multimodal_content(prompt: str, images_b64: list[tuple[str, str]]) -> list[dict]:
    """构造 Qwen3.5 / OpenAI 兼容的多模态 content。

    images_b64: [(camera_key, base64_jpeg), ...]
    """
    content: list[dict] = []
    for idx, (cam_key, b64) in enumerate(images_b64, start=1):
        label = CAMERA_LABELS.get(cam_key, cam_key)
        content.append({"type": "text", "text": f"Picture {idx}: {label}."})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    # Qwen3.5 README: image_url.url 支持 http(s) 或 data URI
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    return content


def main() -> None:
    args = build_args()
    episode = resolve_episode_name(args.video_root, args.episode)

    ref_path = args.video_root / CAMERA_KEYS[0] / episode
    fps, duration, nframes = probe_video(ref_path)
    frame_idx, time_sec = resolve_frame_index(
        fps, duration, nframes, args.time, args.frame_index, args.progress
    )

    print(f"[视频] {episode}")
    print(f"[元信息] fps={fps:.3f} duration={duration:.3f}s frames={nframes}")
    print(f"[抽帧] frame_index={frame_idx} time={time_sec:.3f}s")

    images_b64: list[tuple[str, str]] = []
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    for cam in CAMERA_KEYS:
        path = args.video_root / cam / episode
        jpeg_bytes = extract_frame_jpeg_bytes(path, time_sec, args.jpeg_quality)
        b64 = encode_jpeg_base64(jpeg_bytes)
        images_b64.append((cam, b64))
        print(f"[OK] {cam}: jpeg_bytes={len(jpeg_bytes)} b64_len={len(b64)}")

        if args.save_dir is not None:
            out = (
                args.save_dir
                / f"{Path(episode).stem}__{cam.split('.')[-1]}__t{time_sec:.3f}s.jpg"
            )
            out.write_bytes(jpeg_bytes)
            print(f"  -> saved {out}")

    if args.dry_run:
        print("[dry-run] 跳过模型请求")
        return

    content = build_multimodal_content(args.prompt, images_b64)
    messages = [{"role": "user", "content": content}]

    if args.no_thinking:
        sampling = dict(temperature=0.7, top_p=0.8, presence_penalty=1.5)
        extra_body = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
    else:
        sampling = dict(temperature=1.0, top_p=0.95, presence_penalty=1.5)
        extra_body = {"top_k": 20}

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(f"[请求] {args.base_url} model={args.model} "
          f"thinking={'off' if args.no_thinking else 'on'}")

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


if __name__ == "__main__":
    main()
