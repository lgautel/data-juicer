import json

import numpy as np

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields

OP_NAME = "robot_frame_removal_safety_filter"


@OPERATORS.register_module(OP_NAME)
class RobotFrameRemovalSafetyFilter(Filter):
    """Check whether frame removal would cause unsafe action discontinuities.

    Reads the keep/remove mask produced by upstream operators and simulates
    the effect of removal on action trajectories.  At each removal boundary
    (kept → removed → kept), computes the L2 jump in the action vector.
    Compares boundary jumps against a MAD-based adaptive threshold.  Also
    enforces global constraints on maximum removal ratio and minimum
    remaining frame count.

    Writes scalar safety metrics to Fields.stats and a detailed report to
    Fields.meta.
    """

    def __init__(
        self,
        signal_source: str = "top_level",
        states_key: str = "states",
        actions_key: str = "actions",
        top_level_state_key: str = "states",
        top_level_action_key: str = "actions",
        removal_mask_field: str = "video_quality_removal_mask",
        max_action_jump: float = None,
        max_action_jump_mad_scale: float = 5.0,
        max_removed_ratio: float = 0.5,
        min_remaining_frames: int = 20,
        check_dims: list = None,
        exempt_dims: list = None,
        report_field: str = "frame_removal_safety_report",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.signal_source = signal_source
        self.states_key = states_key
        self.actions_key = actions_key
        self.top_level_state_key = top_level_state_key
        self.top_level_action_key = top_level_action_key
        self.removal_mask_field = removal_mask_field
        self.max_action_jump = max_action_jump
        self.max_action_jump_mad_scale = max_action_jump_mad_scale
        self.max_removed_ratio = max_removed_ratio
        self.min_remaining_frames = min_remaining_frames
        self.check_dims = check_dims
        self.exempt_dims = exempt_dims or []
        self.report_field = report_field

    def _write_default_stats(self, sample, safe=True):
        sample[Fields.stats]["frame_removal_safe"] = safe
        sample[Fields.stats]["frame_removal_max_discontinuity"] = 0.0
        sample[Fields.stats]["frame_removal_removed_ratio"] = 0.0
        sample[Fields.stats]["frame_removal_remaining_frames"] = 0
        return sample

    def compute_stats_single(self, sample, context=False):
        if Fields.stats not in sample or sample[Fields.stats] is None:
            sample[Fields.stats] = {}
        meta = sample.setdefault(Fields.meta, {})

        mask_str = meta.get(self.removal_mask_field)
        if mask_str is None:
            self._write_default_stats(sample, safe=True)
            meta[self.report_field] = json.dumps(
                {"skipped": True, "reason": "no_removal_mask"},
                ensure_ascii=False,
            )
            return sample

        if isinstance(mask_str, str):
            keep_mask = np.array(json.loads(mask_str), dtype=bool)
        else:
            keep_mask = np.array(mask_str, dtype=bool)

        T = len(keep_mask)
        remaining = int(keep_mask.sum())
        removed_ratio = 1.0 - remaining / T if T > 0 else 0.0

        if self.signal_source == "top_level":
            actions_raw = sample.get(self.top_level_action_key)
        else:
            actions_raw = sample.get(self.actions_key)

        if actions_raw is None:
            self._write_default_stats(sample, safe=True)
            sample[Fields.stats]["frame_removal_removed_ratio"] = removed_ratio
            sample[Fields.stats]["frame_removal_remaining_frames"] = remaining
            meta[self.report_field] = json.dumps(
                {"skipped": True, "reason": "no_actions"},
                ensure_ascii=False,
            )
            return sample

        actions = np.array(actions_raw, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)

        dim_indices = list(range(actions.shape[1]))
        if self.check_dims is not None:
            dim_indices = self.check_dims
        dim_indices = [d for d in dim_indices if d not in self.exempt_dims]
        actions_check = actions[:, dim_indices] if dim_indices else actions

        kept_indices = np.where(keep_mask)[0]

        if len(kept_indices) < 2:
            sample[Fields.stats]["frame_removal_safe"] = False
            sample[Fields.stats]["frame_removal_max_discontinuity"] = float("inf")
            sample[Fields.stats]["frame_removal_removed_ratio"] = removed_ratio
            sample[Fields.stats]["frame_removal_remaining_frames"] = remaining
            meta[self.report_field] = json.dumps(
                {"safe": False, "reason": "too_few_remaining"},
                ensure_ascii=False,
            )
            return sample

        all_jumps = np.linalg.norm(
            np.diff(actions_check[kept_indices], axis=0), axis=1
        )

        gaps = np.diff(kept_indices)
        boundary_mask = gaps > 1
        boundary_jumps = all_jumps[boundary_mask]

        max_jump = (
            float(boundary_jumps.max()) if len(boundary_jumps) > 0 else 0.0
        )

        if self.max_action_jump is not None:
            threshold = self.max_action_jump
        elif len(all_jumps) > 0:
            median_jump = np.median(all_jumps)
            mad = np.median(np.abs(all_jumps - median_jump))
            threshold = float(
                median_jump + self.max_action_jump_mad_scale * 1.4826 * mad
            )
        else:
            threshold = float("inf")

        safe = (
            max_jump <= threshold
            and removed_ratio <= self.max_removed_ratio
            and remaining >= self.min_remaining_frames
        )

        sample[Fields.stats]["frame_removal_safe"] = bool(safe)
        sample[Fields.stats]["frame_removal_max_discontinuity"] = max_jump
        sample[Fields.stats]["frame_removal_removed_ratio"] = removed_ratio
        sample[Fields.stats]["frame_removal_remaining_frames"] = remaining

        boundary_details = []
        if len(boundary_jumps) > 0:
            boundary_positions = np.where(boundary_mask)[0]
            for bi, pos in enumerate(boundary_positions):
                boundary_details.append({
                    "before_idx": int(kept_indices[pos]),
                    "after_idx": int(kept_indices[pos + 1]),
                    "gap": int(gaps[pos]),
                    "jump_l2": float(boundary_jumps[bi]),
                })

        report = {
            "safe": bool(safe),
            "max_jump": max_jump,
            "threshold": threshold,
            "removed_ratio": removed_ratio,
            "remaining_frames": remaining,
            "total_frames": T,
            "num_boundaries": len(boundary_details),
            "boundaries": boundary_details,
        }
        meta[self.report_field] = json.dumps(report, ensure_ascii=False)
        return sample

    def process_single(self, sample):
        return sample.get(Fields.stats, {}).get("frame_removal_safe", True)
