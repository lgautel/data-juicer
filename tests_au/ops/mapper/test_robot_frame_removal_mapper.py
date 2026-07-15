"""Unit tests for RobotFrameRemovalMapper."""

import json
import unittest

import numpy as np

from data_juicer._au.ops.mapper.robot_frame_removal_mapper import (
    RobotFrameRemovalMapper,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

MASK_FIELD = "video_quality_removal_mask"
REPORT_FIELD = "frame_removal_report"


def _make_sample(T, keep_mask, safe=True):
    states = np.arange(T * 4, dtype=np.float64).reshape(T, 4)
    actions = np.arange(T * 3, dtype=np.float64).reshape(T, 3)
    timestamps = list(range(T))
    sample = {
        "id": "ep_test",
        "states": states.tolist(),
        "actions": actions.tolist(),
        "timestamp": timestamps,
        Fields.stats: {"frame_removal_safe": safe},
        Fields.meta: {
            MASK_FIELD: json.dumps(
                keep_mask.tolist()
                if isinstance(keep_mask, np.ndarray)
                else keep_mask
            ),
        },
    }
    return sample, states, actions


class RobotFrameRemovalMapperTest(DataJuicerTestCaseBase):

    def test_correct_mask_application(self):
        """Removing specific frames should yield correct remaining arrays."""
        T = 10
        keep_mask = np.array(
            [True, True, False, False, True, True, True, False, True, True]
        )
        sample, states, actions = _make_sample(T, keep_mask)
        expected_remaining = int(keep_mask.sum())

        op = RobotFrameRemovalMapper(
            removal_mask_field=MASK_FIELD,
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])

        self.assertEqual(report["remaining_frames"], expected_remaining)
        self.assertEqual(len(result["states"]), expected_remaining)
        self.assertEqual(len(result["actions"]), expected_remaining)
        self.assertEqual(len(result["timestamp"]), expected_remaining)

        kept_states = states[keep_mask]
        np.testing.assert_allclose(
            np.array(result["states"]), kept_states
        )

    def test_all_kept(self):
        """All-True mask should leave arrays unchanged."""
        T = 10
        keep_mask = np.ones(T, dtype=bool)
        sample, _, _ = _make_sample(T, keep_mask)

        op = RobotFrameRemovalMapper(
            removal_mask_field=MASK_FIELD,
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertEqual(report["removed_frames"], 0)
        self.assertEqual(len(result["states"]), T)

    def test_skip_if_unsafe(self):
        """When safety check fails, removal should be skipped."""
        T = 10
        keep_mask = np.array(
            [True, True, False, False, True, True, True, False, True, True]
        )
        sample, states, _ = _make_sample(T, keep_mask, safe=False)

        op = RobotFrameRemovalMapper(
            removal_mask_field=MASK_FIELD,
            skip_if_unsafe=True,
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertTrue(report["skipped"])
        self.assertEqual(len(result["states"]), T)

    def test_proceed_if_unsafe_disabled(self):
        """When skip_if_unsafe=False, removal proceeds even if unsafe."""
        T = 10
        keep_mask = np.array(
            [True, True, False, False, True, True, True, False, True, True]
        )
        sample, _, _ = _make_sample(T, keep_mask, safe=False)

        op = RobotFrameRemovalMapper(
            removal_mask_field=MASK_FIELD,
            skip_if_unsafe=False,
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertFalse(report.get("skipped", False))
        expected = int(keep_mask.sum())
        self.assertEqual(len(result["states"]), expected)

    def test_recompute_actions(self):
        """Delta actions should be recomputed from remaining states."""
        T = 10
        keep_mask = np.array(
            [True, True, False, True, True, True, True, True, True, True]
        )
        sample, _, _ = _make_sample(T, keep_mask)

        op = RobotFrameRemovalMapper(
            removal_mask_field=MASK_FIELD,
            recompute_actions=True,
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        states_out = np.array(result["states"])
        actions_out = np.array(result["actions"])
        expected_delta = np.diff(states_out, axis=0)
        np.testing.assert_allclose(actions_out[:-1], expected_delta)
        np.testing.assert_allclose(actions_out[-1], expected_delta[-1])

    def test_no_mask_skipped(self):
        """Missing mask should skip removal."""
        sample = {
            "id": "ep",
            "states": np.random.randn(10, 4).tolist(),
            "actions": np.random.randn(10, 3).tolist(),
            Fields.stats: {},
            Fields.meta: {},
        }
        op = RobotFrameRemovalMapper(
            removal_mask_field=MASK_FIELD,
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertTrue(report["skipped"])

    def test_additional_array_keys(self):
        """Extra array keys should also be trimmed."""
        T = 10
        keep_mask = np.array(
            [True, True, False, False, True, True, True, True, True, True]
        )
        sample, _, _ = _make_sample(T, keep_mask)
        sample["extra_col"] = list(range(T))

        op = RobotFrameRemovalMapper(
            removal_mask_field=MASK_FIELD,
            additional_array_keys=["extra_col"],
            report_field=REPORT_FIELD,
        )
        result = op.process_single(sample)
        expected = int(keep_mask.sum())
        self.assertEqual(len(result["extra_col"]), expected)


if __name__ == "__main__":
    unittest.main()
