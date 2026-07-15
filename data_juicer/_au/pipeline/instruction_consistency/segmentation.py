"""Stage 1: Temporal normalization via speed minima detection.

Pure algorithmic — no LLM/VLM/Cursor Agent needed.
Adapted from video_atomic_action_segment_mapper.
"""
from __future__ import annotations

import numpy as np


def segment_episode(
    positions: np.ndarray,
    smooth_window: int = 5,
    min_window: int = 15,
    min_frames: int = 8,
    max_frames: int = 300,
) -> list[dict]:
    """Segment an episode trajectory into subtask clips by speed minima.

    Args:
        positions: End-effector positions, shape (T, 3).
        smooth_window: Savitzky-Golay smoothing window.
        min_window: Half-window for local minima detection.
        min_frames: Minimum frames per segment.
        max_frames: Maximum frames per segment.

    Returns:
        List of segment dicts with segment_id, start_frame, end_frame.
    """
    if len(positions) < 2:
        return [{"segment_id": 0, "start_frame": 0,
                 "end_frame": len(positions)}]

    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = np.concatenate([[0.0], speed])

    # Savitzky-Golay smoothing
    try:
        from scipy.signal import savgol_filter
        win = min(smooth_window, len(speed))
        if win % 2 == 0:
            win -= 1
        if win >= 3:
            speed = savgol_filter(speed, win, min(2, win - 1))
    except ImportError:
        pass  # fallback: use unsmoothed speed

    # local minima detection
    cut_points = []
    for t in range(min_window, len(speed) - min_window):
        lo = max(0, t - min_window)
        hi = min(len(speed), t + min_window + 1)
        if speed[t] == speed[lo:hi].min():
            cut_points.append(t)

    # build segments, filter by length
    boundaries = [0] + cut_points + [len(speed)]
    segments = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s >= min_frames:
            segments.append({
                "segment_id": len(segments),
                "start_frame": s,
                "end_frame": e,
            })

    if not segments:
        segments = [{"segment_id": 0, "start_frame": 0,
                     "end_frame": len(speed)}]

    return segments
