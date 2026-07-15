"""Frame extraction and SDKImage wrapping for Cursor SDK."""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np


def extract_frames_base64(
    video_path: str,
    start_frame: int,
    end_frame: int,
    num_frames: int = 8,
) -> list[dict]:
    """Extract uniformly-sampled frames from a video segment.

    Returns list of {"data": base64_str, "mime_type": "image/jpeg"}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame, total_frames)

    if end_frame <= start_frame:
        cap.release()
        return []

    indices = np.linspace(start_frame, end_frame - 1,
                          num=num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode(".jpg", frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode("utf-8")
            frames.append({"data": b64, "mime_type": "image/jpeg"})

    cap.release()
    return frames


def frames_to_sdk_images(frames: list[dict]):
    """Convert base64 frame dicts to Cursor SDKImage objects.

    Falls back to raw dicts if cursor_sdk is not installed (for testing).
    """
    try:
        from cursor_sdk import SDKImage
        return [SDKImage(data=f["data"], mime_type=f["mime_type"])
                for f in frames]
    except ImportError:
        return frames
