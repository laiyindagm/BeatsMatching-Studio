"""
pose_extraction/tests/test_beat_detector.py

节拍检测算法单元测试（无需 GPU）。

运行：
    python -m pytest pose_extraction/tests/test_beat_detector.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pose_extraction.core.data_types import PoseTimeSeries
from pose_extraction.core.beat_detector import (
    detect_beats_from_displacement,
    detect_beats_from_rotation,
    detect_beats_joint,
    detect_beats,
)


def _make_vitpose_pts(T=200, J=17, fps=30.0, num_beats=8) -> PoseTimeSeries:
    """构造带有人工节拍的 ViTPose PoseTimeSeries。"""
    rng = np.random.default_rng(42)
    ts = np.arange(T) / fps
    # 生成周期性运动（cos 波形，在峰值处为节拍）
    beat_period = T // num_beats
    coords = np.zeros((T, J, 2), dtype=np.float32)
    for j in range(J):
        amp = rng.uniform(10, 50)
        phase = rng.uniform(0, 2 * np.pi)
        coords[:, j, 0] = amp * np.cos(2 * np.pi * np.arange(T) / beat_period + phase) + 320
        coords[:, j, 1] = amp * np.sin(2 * np.pi * np.arange(T) / beat_period + phase) + 240
    conf = rng.uniform(0.5, 1.0, (T, J)).astype(np.float32)
    return PoseTimeSeries(
        source="vitpose",
        video_path="test.mp4",
        fps=fps,
        timestamps=ts,
        joint_coords=coords,
        joint_coords_conf=conf,
        joint_names=[f"j{j}" for j in range(J)],
    )


def _make_hmr2_pts(T=200, fps=30.0, num_beats=8) -> PoseTimeSeries:
    """构造带有人工节拍的 HMR2 PoseTimeSeries。"""
    ts = np.arange(T) / fps
    beat_period = T // num_beats
    # 旋转量在节拍处为零（转向点）
    rotations = np.zeros((T, 24, 3), dtype=np.float32)
    for j in range(24):
        rotations[:, j, 0] = 0.3 * np.sin(2 * np.pi * np.arange(T) / beat_period)
        rotations[:, j, 1] = 0.2 * np.cos(2 * np.pi * np.arange(T) / beat_period)
    return PoseTimeSeries(
        source="hmr2",
        video_path="test.mp4",
        fps=fps,
        timestamps=ts,
        joint_rotations=rotations,
        joint_names=[f"joint_{j}" for j in range(24)],
    )


class TestDetectBeatsFromDisplacement:
    def test_returns_list(self):
        pts = _make_vitpose_pts()
        beats = detect_beats_from_displacement(pts)
        assert isinstance(beats, list)

    def test_beats_in_range(self):
        pts = _make_vitpose_pts(T=200)
        beats = detect_beats_from_displacement(pts)
        for b in beats:
            assert 0 <= b < 200

    def test_beats_sorted(self):
        pts = _make_vitpose_pts()
        beats = detect_beats_from_displacement(pts)
        assert beats == sorted(beats)


class TestDetectBeatsFromRotation:
    def test_returns_list(self):
        pts = _make_hmr2_pts()
        beats = detect_beats_from_rotation(pts)
        assert isinstance(beats, list)

    def test_smoothed_rotations_filled(self):
        pts = _make_hmr2_pts()
        detect_beats_from_rotation(pts)
        assert pts.smoothed_rotations is not None
        assert pts.smoothed_rotations.shape == (200, 24, 3)

    def test_rotation_gradient_filled(self):
        pts = _make_hmr2_pts()
        detect_beats_from_rotation(pts)
        assert pts.rotation_gradient is not None
        assert pts.rotation_gradient.shape == (200, 24)

    def test_beats_sorted(self):
        pts = _make_hmr2_pts()
        beats = detect_beats_from_rotation(pts)
        assert beats == sorted(beats)


class TestDetectBeatsJoint:
    def test_with_both_signals(self):
        pts_v = _make_vitpose_pts()
        pts_h = _make_hmr2_pts()
        # 合并
        pts = PoseTimeSeries(
            source="joint",
            video_path="test.mp4",
            fps=30.0,
            timestamps=pts_v.timestamps,
            joint_coords=pts_v.joint_coords,
            joint_coords_conf=pts_v.joint_coords_conf,
            joint_rotations=pts_h.joint_rotations,
        )
        beats = detect_beats_joint(pts)
        assert isinstance(beats, list)
        assert beats == sorted(beats)

    def test_with_only_coords(self):
        pts = _make_vitpose_pts()
        beats = detect_beats_joint(pts)
        assert isinstance(beats, list)


class TestDetectBeatsAutoDispatch:
    def test_auto_vitpose(self):
        pts = _make_vitpose_pts()
        result = detect_beats(pts, strategy="auto")
        assert result.beat_frames is not None
        assert result.beat_timestamps is not None

    def test_auto_hmr2(self):
        pts = _make_hmr2_pts()
        result = detect_beats(pts, strategy="auto")
        assert result.beat_frames is not None

    def test_explicit_vitpose(self):
        pts = _make_vitpose_pts()
        result = detect_beats(pts, strategy="vitpose")
        assert result.beat_frames is not None

    def test_explicit_hmr2(self):
        pts = _make_hmr2_pts()
        result = detect_beats(pts, strategy="hmr2")
        assert result.beat_frames is not None

    def test_timestamps_match_frames(self):
        pts = _make_vitpose_pts()
        result = detect_beats(pts, strategy="vitpose")
        for i, (frame, ts) in enumerate(zip(result.beat_frames, result.beat_timestamps)):
            expected_ts = pts.timestamps[frame]
            assert abs(ts - expected_ts) < 1e-9

    def test_no_signals_raises(self):
        pts = PoseTimeSeries(
            source="empty",
            video_path="test.mp4",
            fps=30.0,
            timestamps=np.arange(10) / 30.0,
        )
        with pytest.raises((ValueError, AssertionError)):
            detect_beats(pts, strategy="auto")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
