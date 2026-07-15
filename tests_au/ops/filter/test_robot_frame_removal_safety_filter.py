"""Unit tests for RobotFrameRemovalSafetyFilter."""

import json
import unittest

import numpy as np

from data_juicer._au.ops.filter.robot_frame_removal_safety_filter import (
    RobotFrameRemovalSafetyFilter,
)
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

MASK_FIELD = "video_quality_removal_mask"
REPORT_FIELD = "frame_removal_safety_report"
STATS_KEYS = [
    "frame_removal_safe",
    "frame_removal_max_discontinuity",
    "frame_removal_removed_ratio",
    "frame_removal_remaining_frames",
]


def _make_sample(actions, keep_mask):
    return {
        "id": "ep_test",
        "actions": actions.tolist() if isinstance(actions, np.ndarray) else actions,
        Fields.stats: {},
        Fields.meta: {
            MASK_FIELD: json.dumps(
                keep_mask.tolist()
                if isinstance(keep_mask, np.ndarray)
                else keep_mask
            ),
        },
    }


class RobotFrameRemovalSafetyFilterTest(DataJuicerTestCaseBase):

    def _run(self, op, sample):
        sample = op.compute_stats_single(sample)
        keep = op.process_single(sample)
        return sample, keep

    def test_safe_removal(self):
        """Removing a few frames from a smooth trajectory should be safe."""
        T = 100
        actions = np.cumsum(np.random.randn(T, 6) * 0.1, axis=0)
        keep_mask = np.ones(T, dtype=bool)
        keep_mask[:5] = False
        keep_mask[-5:] = False

        sample = _make_sample(actions, keep_mask)
        op = RobotFrameRemovalSafetyFilter(
            removal_mask_field=MASK_FIELD,
            max_action_jump_mad_scale=5.0,
            max_removed_ratio=0.5,
            min_remaining_frames=20,
            report_field=REPORT_FIELD,
        )
        result, keep = self._run(op, sample)
        self.assertTrue(keep)
        self.assertTrue(result[Fields.stats]["frame_removal_safe"])
        for k in STATS_KEYS:
            self.assertIn(k, result[Fields.stats])

    def test_large_jump_rejected(self):
        """Removal causing a large action discontinuity should be rejected."""
        T = 50
        actions = np.zeros((T, 4))
        actions[:20] = 0.0
        actions[20:] = 100.0
        keep_mask = np.ones(T, dtype=bool)
        keep_mask[18:22] = False

        sample = _make_sample(actions, keep_mask)
        op = RobotFrameRemovalSafetyFilter(
            removal_mask_field=MASK_FIELD,
            max_action_jump=5.0,
            max_removed_ratio=0.5,
            min_remaining_frames=10,
            report_field=REPORT_FIELD,
        )
        result, keep = self._run(op, sample)
        self.assertFalse(keep)
        self.assertFalse(result[Fields.stats]["frame_removal_safe"])

    def test_ratio_too_high(self):
        """Removing more than max_removed_ratio should be rejected."""
        T = 100
        actions = np.random.randn(T, 4) * 0.01
        keep_mask = np.zeros(T, dtype=bool)
        keep_mask[:30] = True

        sample = _make_sample(actions, keep_mask)
        op = RobotFrameRemovalSafetyFilter(
            removal_mask_field=MASK_FIELD,
            max_removed_ratio=0.5,
            min_remaining_frames=5,
            report_field=REPORT_FIELD,
        )
        result, keep = self._run(op, sample)
        self.assertFalse(keep)
        self.assertGreater(
            result[Fields.stats]["frame_removal_removed_ratio"], 0.5
        )

    def test_too_few_remaining(self):
        """Fewer remaining frames than minimum should be rejected."""
        T = 30
        actions = np.random.randn(T, 4) * 0.01
        keep_mask = np.ones(T, dtype=bool)
        keep_mask[5:] = False

        sample = _make_sample(actions, keep_mask)
        op = RobotFrameRemovalSafetyFilter(
            removal_mask_field=MASK_FIELD,
            max_removed_ratio=0.99,
            min_remaining_frames=20,
            report_field=REPORT_FIELD,
        )
        result, keep = self._run(op, sample)
        self.assertFalse(keep)
        self.assertLess(
            result[Fields.stats]["frame_removal_remaining_frames"], 20
        )

    def test_no_removal_mask(self):
        """Missing removal mask should default to safe=True."""
        sample = {
            "id": "ep",
            "actions": np.random.randn(20, 4).tolist(),
            Fields.stats: {},
            Fields.meta: {},
        }
        op = RobotFrameRemovalSafetyFilter(
            removal_mask_field=MASK_FIELD,
            report_field=REPORT_FIELD,
        )
        result, keep = self._run(op, sample)
        self.assertTrue(keep)

    def test_all_kept(self):
        """All-True mask should be safe with zero discontinuity."""
        T = 50
        actions = np.random.randn(T, 4) * 0.01
        keep_mask = np.ones(T, dtype=bool)

        sample = _make_sample(actions, keep_mask)
        op = RobotFrameRemovalSafetyFilter(
            removal_mask_field=MASK_FIELD,
            report_field=REPORT_FIELD,
        )
        result, keep = self._run(op, sample)
        self.assertTrue(keep)
        self.assertEqual(
            result[Fields.stats]["frame_removal_max_discontinuity"], 0.0
        )

    def test_report_present(self):
        """Safety report should be written to meta."""
        T = 30
        actions = np.random.randn(T, 4) * 0.01
        keep_mask = np.ones(T, dtype=bool)
        keep_mask[:3] = False
        sample = _make_sample(actions, keep_mask)

        op = RobotFrameRemovalSafetyFilter(
            removal_mask_field=MASK_FIELD,
            report_field=REPORT_FIELD,
        )
        result, _ = self._run(op, sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertIn("safe", report)
        self.assertIn("threshold", report)
        self.assertIn("boundaries", report)

    def test_pipeline_integration(self):
        """Filter should work through the DJ pipeline."""
        T = 40
        ds_list = []
        for i in range(3):
            actions = np.random.randn(T, 4) * 0.01
            km = np.ones(T, dtype=bool)
            if i == 1:
                km[:30] = False
            ds_list.append(_make_sample(actions, km))
            ds_list[-1]["id"] = f"ep_{i}"

        dataset = Dataset.from_list(ds_list)
        op = RobotFrameRemovalSafetyFilter(
            removal_mask_field=MASK_FIELD,
            max_removed_ratio=0.5,
            min_remaining_frames=15,
            report_field=REPORT_FIELD,
        )
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        result = dataset.to_list()
        kept_ids = [r["id"] for r in result]
        self.assertIn("ep_0", kept_ids)
        self.assertNotIn("ep_1", kept_ids)
        self.assertIn("ep_2", kept_ids)


if __name__ == "__main__":
    unittest.main()
