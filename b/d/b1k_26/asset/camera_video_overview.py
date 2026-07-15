#!/mnt/r/VENV/dj/bin/python
"""
Camera Video Overview — extract sample frames from b1k_26 dataset videos to
show the 3 camera views (ZED + left/right RealSense) across several tasks.

Outputs: camera_overview.png  (4 rows x 6 columns, ~18x16 inches, 200 DPI)

Columns: ZED RGB | Left RS RGB | Right RS RGB | ZED Depth | Left RS Depth | Right RS Depth
Rows:    one per selected episode/task
"""

import glob
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIDEO_BASE = '/mnt/g/DATA/b1k_26/datasets/demos/videos'
EPISODE_META_DIR = '/mnt/g/DATA/b1k_26/datasets/demos/meta/episodes'
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'camera_overview.png')

# Episodes to visualise (index -> display label)
EPISODE_PICKS = [
    (0, 'turning_on_radio'),
    (400, 'putting_away_Halloween_decorations'),
    (4400, 'putting_shoes_on_rack'),
    (10000, 'freeze_fruit'),
]

# Video keys in column order (RGB first, then depth)
VIDEO_KEYS_RGB = [
    'observation.rgb.zed_link_camera_0',
    'observation.rgb.left_realsense_link_camera_0',
    'observation.rgb.right_realsense_link_camera_0',
]
VIDEO_KEYS_DEPTH = [
    'observation.depth_linear.zed_link_camera_0',
    'observation.depth_linear.left_realsense_link_camera_0',
    'observation.depth_linear.right_realsense_link_camera_0',
]
VIDEO_KEYS = VIDEO_KEYS_RGB + VIDEO_KEYS_DEPTH

COL_HEADERS = [
    'ZED RGB', 'Left RS RGB', 'Right RS RGB',
    'ZED Depth', 'Left RS Depth', 'Right RS Depth',
]

# Resolution look-up (width, height) — used for ffmpeg rawvideo parsing
RESOLUTION = {
    'observation.rgb.zed_link_camera_0': (720, 720),
    'observation.rgb.left_realsense_link_camera_0': (480, 480),
    'observation.rgb.right_realsense_link_camera_0': (480, 480),
    'observation.depth_linear.zed_link_camera_0': (720, 720),
    'observation.depth_linear.left_realsense_link_camera_0': (480, 480),
    'observation.depth_linear.right_realsense_link_camera_0': (480, 480),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_episodes() -> pd.DataFrame:
    """Load and concatenate all episode metadata parquet files."""
    all_eps = []
    for chunk in sorted(os.listdir(EPISODE_META_DIR)):
        chunk_dir = os.path.join(EPISODE_META_DIR, chunk)
        if not os.path.isdir(chunk_dir):
            continue
        for f in sorted(glob.glob(os.path.join(chunk_dir, '*.parquet'))):
            all_eps.append(pd.read_parquet(f))
    return pd.concat(all_eps, ignore_index=True)


def video_path(video_key: str, chunk_idx: int, file_idx: int) -> str:
    """Build the full path to a video file."""
    return os.path.join(
        VIDEO_BASE, video_key,
        f'chunk-{chunk_idx:03d}', f'file-{file_idx:03d}.mp4',
    )


def extract_frame_ffmpeg(vpath: str, timestamp: float,
                         w: int, h: int, is_depth: bool) -> np.ndarray | None:
    """Extract a single frame at *timestamp* using ffmpeg subprocess.

    For RGB videos the frame is returned as (H, W, 3) uint8.
    For depth videos (gray12le source) the frame is returned as (H, W) uint16
    so that the full 12-bit dynamic range is preserved for colormapping.
    """
    if not os.path.isfile(vpath):
        return None

    if is_depth:
        pix_fmt = 'gray16le'      # 16-bit LE to hold 12-bit values
        bytes_per_pixel = 2
    else:
        pix_fmt = 'rgb24'
        bytes_per_pixel = 3

    expected_bytes = w * h * bytes_per_pixel

    cmd = [
        'ffmpeg',
        '-ss', f'{timestamp:.4f}',
        '-i', vpath,
        '-frames:v', '1',
        '-f', 'rawvideo',
        '-pix_fmt', pix_fmt,
        '-v', 'error',
        '-',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        raw = result.stdout
        if len(raw) < expected_bytes:
            print(f'  [WARN] Short read ({len(raw)}/{expected_bytes} bytes) '
                  f'from {vpath} @ {timestamp:.2f}s', file=sys.stderr)
            return None
        if is_depth:
            frame = np.frombuffer(raw[:expected_bytes],
                                  dtype=np.uint16).reshape(h, w)
        else:
            frame = np.frombuffer(raw[:expected_bytes],
                                  dtype=np.uint8).reshape(h, w, 3)
        return frame
    except Exception as exc:
        print(f'  [ERROR] ffmpeg failed for {vpath}: {exc}', file=sys.stderr)
        return None


def gray_placeholder(h: int = 480, w: int = 480) -> np.ndarray:
    """Return a neutral gray placeholder image (RGB)."""
    return np.full((h, w, 3), 180, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Loading episode metadata ...')
    eps = load_episodes()

    n_rows = len(EPISODE_PICKS)
    n_cols = len(VIDEO_KEYS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 16))

    for row_idx, (ep_idx, task_label) in enumerate(EPISODE_PICKS):
        ep_row = eps[eps['episode_index'] == ep_idx]
        if ep_row.empty:
            print(f'Episode {ep_idx} not found — filling row with placeholders',
                  file=sys.stderr)
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                ax.imshow(gray_placeholder())
                ax.set_axis_off()
            continue

        ep = ep_row.iloc[0]
        # Use actual task name from metadata if available
        actual_tasks = ep.get('tasks', [task_label])
        if isinstance(actual_tasks, (list, np.ndarray)) and len(actual_tasks):
            display_task = actual_tasks[0]
        else:
            display_task = task_label

        print(f'Episode {ep_idx}: {display_task}')

        for col_idx, vk in enumerate(VIDEO_KEYS):
            ax = axes[row_idx, col_idx]
            is_depth = 'depth_linear' in vk

            # Read chunk/file/timestamps from episode metadata
            prefix = f'videos/{vk}'
            chunk_idx = int(ep[f'{prefix}/chunk_index'])
            file_idx = int(ep[f'{prefix}/file_index'])
            from_ts = float(ep[f'{prefix}/from_timestamp'])
            to_ts = float(ep[f'{prefix}/to_timestamp'])
            mid_ts = (from_ts + to_ts) / 2.0

            vpath = video_path(vk, chunk_idx, file_idx)
            w, h = RESOLUTION[vk]

            print(f'  {vk}  chunk={chunk_idx} file={file_idx} '
                  f'ts={mid_ts:.2f}s  -> {vpath}')

            frame = extract_frame_ffmpeg(vpath, mid_ts, w, h, is_depth)

            if frame is None:
                ax.imshow(gray_placeholder(h, w))
            elif is_depth:
                ax.imshow(frame, cmap='plasma')
            else:
                ax.imshow(frame)

            ax.set_axis_off()

            # Column headers (top row only)
            if row_idx == 0:
                ax.set_title(COL_HEADERS[col_idx], fontsize=11,
                             fontweight='bold', pad=8)

        # Row label on the leftmost cell
        axes[row_idx, 0].text(
            -0.08, 0.5,
            f'Ep {ep_idx}\n{display_task}',
            transform=axes[row_idx, 0].transAxes,
            fontsize=10, fontweight='bold',
            va='center', ha='right',
            rotation=0,
        )

    plt.tight_layout(rect=[0.08, 0.0, 1.0, 1.0])
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f'\nSaved: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
