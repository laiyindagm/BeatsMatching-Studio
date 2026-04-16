"""
pose_extraction/tests/test_smoothing.py

平滑模块单元测试。

运行：
    python -m pytest pose_extraction/tests/test_smoothing.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pose_extraction.core.smoothing import (
    gaussian_weights,
    uniform_weights,
    smooth_sequence,
    smooth_sequence_fast,
    compute_rotation_gradient,
    compute_rotation_gradient_vectorized,
    compute_displacement,
    compute_cumulative_displacement,
    compute_velocity,
    smooth_coords,
)


class TestWeights:
    def test_gaussian_weights_sum(self):
        w = gaussian_weights(3)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_uniform_weights_sum(self):
        w = uniform_weights(3)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_gaussian_symmetry(self):
        w = gaussian_weights(4)
        assert np.allclose(w, w[::-1])


class TestSmoothSequence:
    def test_shape_preserved(self):
        seq = np.random.randn(50, 24, 3)
        out = smooth_sequence(seq, K=3)
        assert out.shape == seq.shape

    def test_constant_sequence_unchanged(self):
        seq = np.ones((20, 5, 3)) * 3.14
        out = smooth_sequence(seq, K=2)
        assert np.allclose(out, seq)

    def test_fast_shape_preserved(self):
        seq = np.random.randn(60, 24, 3)
        out = smooth_sequence_fast(seq, K=4)
        assert out.shape == seq.shape


class TestRotationGradient:
    def test_zero_gradient_for_constant(self):
        rot = np.ones((30, 24, 3)) * 0.5
        grad = compute_rotation_gradient_vectorized(rot)
        assert np.allclose(grad[1:], 0.0)
        assert np.allclose(grad[0], 0.0)

    def test_gradient_shape(self):
        rot = np.random.randn(25, 24, 3)
        grad = compute_rotation_gradient_vectorized(rot)
        assert grad.shape == (25, 24)

    def test_gradient_nonnegative(self):
        rot = np.random.randn(15, 24, 3)
        grad = compute_rotation_gradient_vectorized(rot)
        assert np.all(grad >= 0)

    def test_consistency_with_loopy_version(self):
        rot = np.random.randn(10, 6, 3)
        g1 = compute_rotation_gradient(rot)
        g2 = compute_rotation_gradient_vectorized(rot)
        assert np.allclose(g1, g2)


class TestDisplacement:
    def test_zero_displacement_ref_frame(self):
        coords = np.random.randn(10, 17, 2)
        disp = compute_displacement(coords, ref_frame=0)
        assert np.allclose(disp[0], 0.0)

    def test_cumulative_shape(self):
        coords = np.random.randn(20, 17, 2)
        cum = compute_cumulative_displacement(coords)
        assert cum.shape == (20, 17)

    def test_cumulative_monotone_increasing(self):
        # 每帧都有移动，累计距离不减
        rng = np.random.default_rng(42)
        coords = np.cumsum(rng.uniform(0.1, 1.0, (20, 5, 2)), axis=0)
        cum = compute_cumulative_displacement(coords)
        assert np.all(np.diff(cum, axis=0) >= -1e-10)

    def test_velocity_shape(self):
        coords = np.random.randn(15, 17, 2)
        vel = compute_velocity(coords, fps=25.0)
        assert vel.shape == (15, 17, 2)


class TestSmoothCoords:
    def test_savgol_shape(self):
        coords = np.random.randn(50, 17, 2)
        out = smooth_coords(coords, method="savgol", window=7)
        assert out.shape == (50, 17, 2)

    def test_gaussian_shape(self):
        coords = np.random.randn(40, 17, 2)
        out = smooth_coords(coords, method="gaussian", K=3)
        assert out.shape == (40, 17, 2)

    def test_nan_handling(self):
        coords = np.random.randn(20, 5, 2)
        coords[5:8, 2, :] = np.nan   # 部分帧关节缺失
        out = smooth_coords(coords, method="savgol")
        # 确保没有崩溃，形状正确
        assert out.shape == (20, 5, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
