"""
pose_extraction/tests/test_data_types.py

数据类型单元测试（无需 GPU，无需安装深度学习依赖）。

运行：
    cd d:/毕业设计
    python -m pytest pose_extraction/tests/test_data_types.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pose_extraction.core.data_types import (
    ViTPoseFrameResult,
    ViTPoseSequence,
    HMR2FrameResult,
    HMR2Sequence,
    PoseTimeSeries,
    SMPL_JOINT_NAMES,
    SMPL_PARENT_IDX,
)


# ──────────────────────────────────────────────────────────────────────────────
# ViTPose 数据类测试
# ──────────────────────────────────────────────────────────────────────────────

class TestViTPoseSequence:

    def _make_sequence(self, T=30, J=17, fps=30.0) -> ViTPoseSequence:
        seq = ViTPoseSequence(
            video_path="test.mp4",
            fps=fps,
            total_frames=T,
            joint_names=[f"joint_{j}" for j in range(J)],
        )
        rng = np.random.default_rng(42)
        for t in range(T):
            kpts = rng.uniform(0, 640, size=(J, 3)).astype(np.float32)
            kpts[:, 2] = rng.uniform(0.4, 1.0, size=J)   # 置信度 > 0.3
            frame = ViTPoseFrameResult(
                frame_idx=t,
                timestamp=t / fps,
                persons={0: kpts},
                num_joints=J,
            )
            seq.frames.append(frame)
        return seq

    def test_get_joint_trajectory(self):
        seq = self._make_sequence()
        ts, coords = seq.get_joint_trajectory(joint_idx=9, person_id=0)
        assert ts.shape == (30,)
        assert coords.shape == (30, 2)

    def test_get_all_joints_trajectory(self):
        seq = self._make_sequence()
        ts, coords = seq.get_all_joints_trajectory(person_id=0)
        assert ts.shape == (30,)
        assert coords.shape == (30, 17, 2)

    def test_missing_person(self):
        seq = self._make_sequence()
        ts, coords = seq.get_joint_trajectory(joint_idx=0, person_id=99)
        # 全为 NaN
        assert np.all(np.isnan(coords))

    def test_conf_threshold(self):
        seq = ViTPoseSequence(
            video_path="test.mp4", fps=30.0, total_frames=5,
            joint_names=["nose"]
        )
        for t in range(5):
            kpts = np.array([[100.0, 200.0, 0.1]])  # 低置信度
            seq.frames.append(ViTPoseFrameResult(
                frame_idx=t, timestamp=t/30.0,
                persons={0: kpts}, num_joints=1,
            ))
        _, coords = seq.get_joint_trajectory(0, conf_threshold=0.3)
        assert np.all(np.isnan(coords))


# ──────────────────────────────────────────────────────────────────────────────
# HMR2 数据类测试
# ──────────────────────────────────────────────────────────────────────────────

class TestHMR2FrameResult:

    def _make_frame(self, frame_idx=0, timestamp=0.0) -> HMR2FrameResult:
        rng = np.random.default_rng(0)
        return HMR2FrameResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            person_id=0,
            global_orient=rng.uniform(-0.5, 0.5, (1, 3)).astype(np.float32),
            body_pose=rng.uniform(-0.5, 0.5, (23, 3)).astype(np.float32),
            betas=rng.standard_normal(10).astype(np.float32),
        )

    def test_full_pose_shape(self):
        fr = self._make_frame()
        pose = fr.full_pose
        assert pose is not None
        assert pose.shape == (24, 3)

    def test_get_joint_rotation(self):
        fr = self._make_frame()
        for j in range(24):
            rot = fr.get_joint_rotation(j)
            assert rot is not None and rot.shape == (3,)

    def test_full_pose_none(self):
        fr = HMR2FrameResult(frame_idx=0, timestamp=0.0)
        assert fr.full_pose is None


class TestHMR2Sequence:

    def _make_sequence(self, T=20, fps=25.0) -> HMR2Sequence:
        seq = HMR2Sequence(
            video_path="test.mp4", fps=fps, total_frames=T
        )
        rng = np.random.default_rng(1)
        for t in range(T):
            fr = HMR2FrameResult(
                frame_idx=t,
                timestamp=t / fps,
                person_id=0,
                global_orient=rng.uniform(-0.5, 0.5, (1, 3)).astype(np.float32),
                body_pose=rng.uniform(-0.5, 0.5, (23, 3)).astype(np.float32),
                betas=rng.standard_normal(10).astype(np.float32),
            )
            seq.frames.append(fr)
        seq.person_ids = [0]
        return seq

    def test_joint_rotation_sequence(self):
        seq = self._make_sequence()
        ts, rot = seq.get_joint_rotation_sequence(joint_idx=18, person_id=0)
        assert ts.shape == (20,)
        assert rot.shape == (20, 3)

    def test_all_joints_rotation_sequence(self):
        seq = self._make_sequence()
        ts, rot = seq.get_all_joints_rotation_sequence(person_id=0)
        assert ts.shape == (20,)
        assert rot.shape == (20, 24, 3)


# ──────────────────────────────────────────────────────────────────────────────
# SMPL 常量测试
# ──────────────────────────────────────────────────────────────────────────────

def test_smpl_joint_names_length():
    assert len(SMPL_JOINT_NAMES) == 24

def test_smpl_parent_idx_length():
    assert len(SMPL_PARENT_IDX) == 24

def test_smpl_root():
    assert SMPL_PARENT_IDX[0] == -1  # pelvis 是根节点


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
