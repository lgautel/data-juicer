import json

import numpy as np

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader

cv2 = LazyLoader("cv2", "opencv-contrib-python")

OP_NAME = "robot_static_segment_detector_mapper"


def _compute_ssim_pair(img1, img2, win_size=7):
    """Compute SSIM between two grayscale images using cv2 only."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (win_size, win_size), 1.5)
    mu2 = cv2.GaussianBlur(img2, (win_size, win_size), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (win_size, win_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (win_size, win_size), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (win_size, win_size), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / den
    return float(np.mean(ssim_map))


@OPERATORS.register_module(OP_NAME)
class RobotStaticSegmentDetectorMapper(Mapper):
    """Detect static segments using joint video SSIM and state signal evidence.

    Computes per-frame SSIM between adjacent frames and per-frame state
    standard deviation in a sliding window.  Finds contiguous runs where
    both signals indicate stationarity.  Merges with upstream bad-frame mask
    and subtracts protected keyframe indices to produce a final keep/remove
    mask written to Fields.meta.
    """

    def __init__(
        self,
        signal_source: str = "top_level",
        states_key: str = "states",
        top_level_state_key: str = "states",
        video_field_index: int = 0,
        ssim_threshold: float = 0.98,
        state_motion_threshold: float = 0.001,
        window_size: int = 5,
        min_static_run: int = 10,
        require_joint_evidence: bool = True,
        edge_only: bool = True,
        edge_margin_ratio: float = 0.15,
        quality_report_field: str = "frame_quality_report",
        keyframe_report_field: str = "key_frame_report",
        removal_mask_field: str = "video_quality_removal_mask",
        exempt_dims: list = None,
        resize_for_ssim: list = None,
        sampling_fps: float = None,
        original_fps: float = None,
        report_field: str = "static_segment_report",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.signal_source = signal_source
        self.states_key = states_key
        self.top_level_state_key = top_level_state_key
        self.video_field_index = video_field_index
        self.ssim_threshold = ssim_threshold
        self.state_motion_threshold = state_motion_threshold
        self.window_size = window_size
        self.min_static_run = min_static_run
        self.require_joint_evidence = require_joint_evidence
        self.edge_only = edge_only
        self.edge_margin_ratio = edge_margin_ratio
        self.quality_report_field = quality_report_field
        self.keyframe_report_field = keyframe_report_field
        self.removal_mask_field = removal_mask_field
        self.exempt_dims = exempt_dims or []
        self.resize_for_ssim = (
            tuple(resize_for_ssim) if resize_for_ssim else (160, 120)
        )
        self.sampling_fps = sampling_fps
        self.original_fps = original_fps
        self.report_field = report_field

    def _load_frames_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS) or self.original_fps or 15.0
        step = 1
        if self.sampling_fps and self.sampling_fps < fps:
            step = max(1, int(round(fps / self.sampling_fps)))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                if self.resize_for_ssim:
                    frame = cv2.resize(
                        frame, self.resize_for_ssim, interpolation=cv2.INTER_AREA
                    )
                gray = (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if len(frame.shape) == 3
                    else frame
                )
                frames.append(gray)
            idx += 1
        cap.release()
        return frames

    def _compute_ssim_series(self, gray_frames):
        T = len(gray_frames)
        if T < 2:
            return np.zeros(T)
        ssim_vals = np.zeros(T)
        ssim_vals[0] = 0.0
        for i in range(1, T):
            ssim_vals[i] = _compute_ssim_pair(gray_frames[i - 1], gray_frames[i])
        return ssim_vals

    def _compute_state_stationarity(self, states):
        T = states.shape[0]
        check_dims = [
            d for d in range(states.shape[1]) if d not in self.exempt_dims
        ]
        if not check_dims:
            return np.zeros(T)

        state_sub = states[:, check_dims]
        stdev_per_frame = np.zeros(T)
        w = self.window_size
        for t in range(T):
            lo = max(0, t - w)
            hi = min(T, t + w + 1)
            window = state_sub[lo:hi]
            stdev_per_frame[t] = float(np.mean(np.std(window, axis=0)))
        return stdev_per_frame

    def _find_static_runs(self, static_mask, min_run):
        runs = []
        T = len(static_mask)
        i = 0
        while i < T:
            if static_mask[i]:
                j = i
                while j < T and static_mask[j]:
                    j += 1
                length = j - i
                if length >= min_run:
                    runs.append({"start": i, "end": j, "length": length})
                i = j
            else:
                i += 1
        return runs

    def _filter_edge_runs(self, runs, T):
        if not self.edge_only:
            for run in runs:
                run["location"] = "interior"
            return runs
        edge_start = int(T * self.edge_margin_ratio)
        edge_end = T - edge_start
        filtered = []
        for run in runs:
            if run["end"] <= edge_start:
                run["location"] = "start"
                filtered.append(run)
            elif run["start"] >= edge_end:
                run["location"] = "end"
                filtered.append(run)
        return filtered

    def _consolidate_masks(self, T, bad_frame_mask, static_removal_mask,
                           protected_indices):
        remove_mask = np.zeros(T, dtype=bool)
        if bad_frame_mask is not None:
            bfm = np.array(bad_frame_mask, dtype=bool)
            if len(bfm) == T:
                remove_mask |= bfm
        remove_mask |= np.array(static_removal_mask, dtype=bool)

        if protected_indices:
            for idx in protected_indices:
                if 0 <= idx < T:
                    remove_mask[idx] = False

        keep_mask = ~remove_mask
        return keep_mask

    def process_single(self, sample):
        meta = sample.setdefault(Fields.meta, {})

        states_raw = (
            sample.get(self.top_level_state_key)
            if self.signal_source == "top_level"
            else None
        )

        video_keys = sample.get(self.video_key)

        bad_frame_mask = None
        quality_report_str = meta.get(self.quality_report_field)
        if quality_report_str:
            if isinstance(quality_report_str, str):
                quality_report = json.loads(quality_report_str)
            else:
                quality_report = quality_report_str
            bad_frame_mask = quality_report.get("bad_frame_mask")

        protected_indices = []
        keyframe_report_str = meta.get(self.keyframe_report_field)
        if keyframe_report_str:
            if isinstance(keyframe_report_str, str):
                keyframe_report = json.loads(keyframe_report_str)
            else:
                keyframe_report = keyframe_report_str
            protected_indices = keyframe_report.get(
                "protected_frame_indices", []
            )

        states = None
        if states_raw is not None:
            states = np.array(states_raw, dtype=np.float64)

        T = None
        if states is not None:
            T = len(states)

        gray_frames = []
        has_video = False
        if video_keys:
            if isinstance(video_keys, list):
                idx = min(self.video_field_index, len(video_keys) - 1)
                video_path = video_keys[idx]
            else:
                video_path = video_keys
            gray_frames = self._load_frames_from_video(video_path)
            has_video = len(gray_frames) > 0
            if T is None and has_video:
                T = len(gray_frames)

        if T is None or T < 2:
            meta[self.report_field] = json.dumps(
                {"error": "insufficient_data", "num_frames": T or 0},
                ensure_ascii=False,
            )
            if T is not None:
                meta[self.removal_mask_field] = json.dumps(
                    [True] * T, ensure_ascii=False
                )
            return sample

        video_static = np.zeros(T, dtype=bool)
        ssim_series = None
        if has_video:
            ssim_series = self._compute_ssim_series(gray_frames)
            video_static = ssim_series >= self.ssim_threshold
            video_static[0] = video_static[1] if T > 1 else False

        state_static = np.zeros(T, dtype=bool)
        stdev_series = None
        if states is not None and states.shape[0] == T:
            stdev_series = self._compute_state_stationarity(states)
            state_static = stdev_series < self.state_motion_threshold

        if self.require_joint_evidence and has_video and states is not None:
            combined_static = video_static & state_static
        elif has_video:
            combined_static = video_static
        elif states is not None:
            combined_static = state_static
        else:
            combined_static = np.zeros(T, dtype=bool)

        static_runs = self._find_static_runs(combined_static, self.min_static_run)
        edge_runs = self._filter_edge_runs(static_runs, T)

        static_removal_mask = np.zeros(T, dtype=bool)
        for run in edge_runs:
            static_removal_mask[run["start"]:run["end"]] = True

        keep_mask = self._consolidate_masks(
            T, bad_frame_mask, static_removal_mask, protected_indices
        )

        total_static = int(static_removal_mask.sum())
        total_bad = int(np.sum(bad_frame_mask)) if bad_frame_mask else 0
        total_protected = len(protected_indices)
        final_removal = int((~keep_mask).sum())

        run_details = []
        for run in edge_runs:
            detail = {
                "start": run["start"],
                "end": run["end"],
                "length": run["length"],
                "location": run.get("location", "unknown"),
            }
            if ssim_series is not None:
                detail["mean_ssim"] = float(
                    np.mean(ssim_series[run["start"]:run["end"]])
                )
            if stdev_series is not None:
                detail["mean_state_std"] = float(
                    np.mean(stdev_series[run["start"]:run["end"]])
                )
            run_details.append(detail)

        report = {
            "num_static_runs": len(edge_runs),
            "static_runs": run_details,
            "total_static_frames": total_static,
            "total_bad_quality_frames": total_bad,
            "total_protected_frames": total_protected,
            "removal_candidates_before_protection": total_static + total_bad,
            "final_removal_count": final_removal,
        }
        meta[self.report_field] = json.dumps(report, ensure_ascii=False)
        meta[self.removal_mask_field] = json.dumps(
            keep_mask.tolist(), ensure_ascii=False
        )
        return sample
