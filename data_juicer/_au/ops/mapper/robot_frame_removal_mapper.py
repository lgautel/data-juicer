import json

import numpy as np

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.constant import Fields

OP_NAME = "robot_frame_removal_mapper"


@OPERATORS.register_module(OP_NAME)
class RobotFrameRemovalMapper(Mapper):
    """Apply the frame removal mask to trim all array fields in an episode.

    Reads the keep/remove mask and safety check result from upstream
    operators.  If the safety check passed (or is absent), applies the mask
    to states, actions, timestamps, and any additional array columns.
    Optionally recomputes delta actions from the remaining states.
    """

    def __init__(
        self,
        signal_source: str = "top_level",
        states_key: str = "states",
        actions_key: str = "actions",
        top_level_state_key: str = "states",
        top_level_action_key: str = "actions",
        removal_mask_field: str = "video_quality_removal_mask",
        timestamp_key: str = "timestamp",
        frame_index_key: str = "frame_index",
        additional_array_keys: list = None,
        safety_check_field: str = "frame_removal_safe",
        skip_if_unsafe: bool = True,
        recompute_actions: bool = False,
        report_field: str = "frame_removal_report",
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
        self.timestamp_key = timestamp_key
        self.frame_index_key = frame_index_key
        self.additional_array_keys = additional_array_keys or []
        self.safety_check_field = safety_check_field
        self.skip_if_unsafe = skip_if_unsafe
        self.recompute_actions = recompute_actions
        self.report_field = report_field

    def _get_array_keys(self):
        keys = []
        if self.signal_source == "top_level":
            keys.extend([self.top_level_state_key, self.top_level_action_key])
        else:
            keys.extend([self.states_key, self.actions_key])
        keys.append(self.timestamp_key)
        keys.append(self.frame_index_key)
        keys.extend(self.additional_array_keys)
        return keys

    def process_single(self, sample):
        meta = sample.setdefault(Fields.meta, {})

        if self.skip_if_unsafe:
            stats = sample.get(Fields.stats, {})
            if not stats.get(self.safety_check_field, True):
                meta[self.report_field] = json.dumps(
                    {"skipped": True, "reason": "safety_check_failed"},
                    ensure_ascii=False,
                )
                return sample

        mask_str = meta.get(self.removal_mask_field)
        if mask_str is None:
            meta[self.report_field] = json.dumps(
                {"skipped": True, "reason": "no_removal_mask"},
                ensure_ascii=False,
            )
            return sample

        if isinstance(mask_str, str):
            keep_mask = np.array(json.loads(mask_str), dtype=bool)
        else:
            keep_mask = np.array(mask_str, dtype=bool)

        T_original = len(keep_mask)

        if keep_mask.all():
            meta[self.report_field] = json.dumps(
                {
                    "original_frames": T_original,
                    "removed_frames": 0,
                    "remaining_frames": T_original,
                },
                ensure_ascii=False,
            )
            return sample

        for key in self._get_array_keys():
            if key in sample and sample[key] is not None:
                arr = sample[key]
                if isinstance(arr, (list, np.ndarray)):
                    arr = np.asarray(arr)
                    if len(arr) == T_original:
                        sample[key] = arr[keep_mask].tolist()

        if self.recompute_actions:
            sk = (
                self.top_level_state_key
                if self.signal_source == "top_level"
                else self.states_key
            )
            ak = (
                self.top_level_action_key
                if self.signal_source == "top_level"
                else self.actions_key
            )
            if sk in sample and sample[sk] is not None:
                states = np.array(sample[sk], dtype=np.float64)
                if len(states) >= 2:
                    delta = np.diff(states, axis=0)
                    delta = np.vstack([delta, delta[-1:]])
                    sample[ak] = delta.tolist()

        remaining = int(keep_mask.sum())
        meta[self.report_field] = json.dumps(
            {
                "original_frames": T_original,
                "removed_frames": T_original - remaining,
                "remaining_frames": remaining,
                "recomputed_actions": self.recompute_actions,
            },
            ensure_ascii=False,
        )
        return sample
