"""Unit tests for RobotStaticSegmentDetectorMapper."""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from data_juicer._au.ops.mapper.robot_static_segment_detector_mapper import (
    RobotStaticSegmentDetectorMapper,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

REPORT_FIELD = "static_segment_report"
MASK_FIELD = "video_quality_removal_mask"


def _make_video(frames, path, fps=15):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def _static_frame(seed=0, h=60, w=80):
    rng = np.random.RandomState(seed)
    return rng.randint(50, 200, (h, w, 3), dtype=np.uint8)


def _moving_frame(idx, h=60, w=80):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    x = int((idx * 5) % (w - 20))
    y = int((idx * 3) % (h - 20))
    cv2.rectangle(frame, (x, y), (x + 20, y + 20), (0, 255, 0), -1)
    frame += np.random.randint(30, 90, (h, w, 3), dtype=np.uint8)
    return frame


class RobotStaticSegmentDetectorMapperTest(DataJuicerTestCaseBase):

    def _make_sample(self, frames, states, tmpdir, meta=None):
        path = os.path.join(tmpdir, "test.mp4")
        _make_video(frames, path)
        sample = {
            "videos": [path],
            "states": states.tolist(),
            Fields.meta: meta or {},
        }
        return sample

    def test_all_moving_no_removal(self):
        """Episode where everything moves should have no static segments."""
        T = 30
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [_moving_frame(i) for i in range(T)]
            states = np.cumsum(np.random.randn(T, 4) * 0.1, axis=0)
            sample = self._make_sample(frames, states, tmpdir)

            op = RobotStaticSegmentDetectorMapper(
                ssim_threshold=0.98,
                state_motion_threshold=0.001,
                min_static_run=5,
                edge_only=False,
                report_field=REPORT_FIELD,
                removal_mask_field=MASK_FIELD,
            )
            result = op.process_single(sample)
            report = json.loads(result[Fields.meta][REPORT_FIELD])
            mask = json.loads(result[Fields.meta][MASK_FIELD])
            self.assertEqual(report["num_static_runs"], 0)
            self.assertTrue(all(mask))

    def test_start_end_static_detected(self):
        """Static frames at start and end should be detected and marked."""
        T = 60
        with tempfile.TemporaryDirectory() as tmpdir:
            static_f = _static_frame(seed=42)
            frames = (
                [static_f.copy() for _ in range(15)]
                + [_moving_frame(i) for i in range(30)]
                + [static_f.copy() for _ in range(15)]
            )
            states = np.zeros((T, 4))
            states[:15] = 0.0
            states[15:45] = np.cumsum(np.random.randn(30, 4) * 0.1, axis=0)
            states[45:] = states[44]

            sample = self._make_sample(frames, states, tmpdir)
            op = RobotStaticSegmentDetectorMapper(
                ssim_threshold=0.95,
                state_motion_threshold=0.01,
                min_static_run=5,
                window_size=3,
                edge_only=True,
                edge_margin_ratio=0.3,
                require_joint_evidence=True,
                report_field=REPORT_FIELD,
                removal_mask_field=MASK_FIELD,
            )
            result = op.process_single(sample)
            report = json.loads(result[Fields.meta][REPORT_FIELD])
            mask = json.loads(result[Fields.meta][MASK_FIELD])
            self.assertGreater(report["final_removal_count"], 0)
            self.assertFalse(all(mask))

    def test_state_only_detection(self):
        """When no video available, use state signal alone."""
        T = 40
        states = np.zeros((T, 4))
        states[10:30] = np.cumsum(np.random.randn(20, 4) * 0.5, axis=0)

        sample = {
            "states": states.tolist(),
            Fields.meta: {},
        }
        op = RobotStaticSegmentDetectorMapper(
            ssim_threshold=0.98,
            state_motion_threshold=0.001,
            min_static_run=5,
            window_size=3,
            edge_only=False,
            require_joint_evidence=False,
            report_field=REPORT_FIELD,
            removal_mask_field=MASK_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        self.assertGreaterEqual(report["num_static_runs"], 0)
        self.assertIn(MASK_FIELD, result[Fields.meta])

    def test_keyframe_protection(self):
        """Protected keyframes should remain in keep_mask even in static segments."""
        T = 30
        states = np.zeros((T, 4))

        keyframe_report = json.dumps({
            "protected_frame_indices": [10, 11, 12, 13, 14],
        })

        sample = {
            "states": states.tolist(),
            Fields.meta: {
                "key_frame_report": keyframe_report,
            },
        }
        op = RobotStaticSegmentDetectorMapper(
            ssim_threshold=0.98,
            state_motion_threshold=0.1,
            min_static_run=5,
            window_size=3,
            edge_only=False,
            require_joint_evidence=False,
            report_field=REPORT_FIELD,
            removal_mask_field=MASK_FIELD,
        )
        result = op.process_single(sample)
        mask = json.loads(result[Fields.meta][MASK_FIELD])
        for idx in [10, 11, 12, 13, 14]:
            self.assertTrue(mask[idx], f"Protected frame {idx} was removed")

    def test_bad_frame_mask_merged(self):
        """Bad frames from quality scorer should be merged into removal mask."""
        T = 20
        states = np.cumsum(np.random.randn(T, 4) * 0.5, axis=0)

        quality_report = json.dumps({
            "bad_frame_mask": [True, True] + [False] * 18,
        })

        sample = {
            "states": states.tolist(),
            Fields.meta: {
                "frame_quality_report": quality_report,
            },
        }
        op = RobotStaticSegmentDetectorMapper(
            ssim_threshold=0.98,
            state_motion_threshold=0.001,
            min_static_run=5,
            edge_only=False,
            require_joint_evidence=False,
            report_field=REPORT_FIELD,
            removal_mask_field=MASK_FIELD,
        )
        result = op.process_single(sample)
        mask = json.loads(result[Fields.meta][MASK_FIELD])
        self.assertFalse(mask[0])
        self.assertFalse(mask[1])

    def test_edge_only_ignores_interior(self):
        """Interior static segments should be ignored when edge_only=True."""
        T = 100
        states = np.zeros((T, 4))
        states[:20] = np.cumsum(np.random.randn(20, 4) * 0.5, axis=0)
        states[40:60] = 0.0
        states[80:] = np.cumsum(np.random.randn(20, 4) * 0.5, axis=0)

        sample = {
            "states": states.tolist(),
            Fields.meta: {},
        }
        op = RobotStaticSegmentDetectorMapper(
            state_motion_threshold=0.001,
            min_static_run=5,
            window_size=3,
            edge_only=True,
            edge_margin_ratio=0.15,
            require_joint_evidence=False,
            report_field=REPORT_FIELD,
            removal_mask_field=MASK_FIELD,
        )
        result = op.process_single(sample)
        report = json.loads(result[Fields.meta][REPORT_FIELD])
        for run in report.get("static_runs", []):
            self.assertIn(run["location"], ("start", "end"))


if __name__ == "__main__":
    unittest.main()
