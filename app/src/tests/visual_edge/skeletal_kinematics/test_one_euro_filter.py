from typing import Dict

import numpy as np

from app.src.visual_edge.skeletal_kinematics.one_euro_filter import (
    _compute_smoothing_factor,
    apply_one_euro_filter,
    calculate_kinematic_metrics,
)


def test_compute_smoothing_factor_expected_behaviour() -> None:
    """Verifies that the exponential smoothing factor (alpha) calculates correctly based on frequency math."""
    # R = 2 * pi * 1.0 * 1.0 = 6.283
    # Alpha = 6.283 / 7.283 = ~0.862
    alpha: float = _compute_smoothing_factor(time_delta=1.0, cutoff_frequency=1.0)
    assert isinstance(alpha, float)
    assert 0.86 < alpha < 0.87


def test_apply_one_euro_filter_expected_behaviour_for_standard_trajectory() -> None:
    """Verifies the filter successfully traverses a coordinate trajectory and outputs smoothed data."""
    trajectory: np.ndarray = np.random.randn(30, 33, 3).astype(float)
    smoothed: np.ndarray = apply_one_euro_filter(trajectory, fps=30.0)
    
    assert smoothed.shape == trajectory.shape
    assert smoothed.dtype == float
    # The first frame of the filter must strictly match the raw input
    np.testing.assert_array_equal(smoothed[0], trajectory[0])


def test_apply_one_euro_filter_expected_behaviour_for_empty_trajectory() -> None:
    """Ensures the filter gracefully bypasses processing when handed an empty coordinate matrix."""
    empty_trajectory: np.ndarray = np.array([])
    smoothed: np.ndarray = apply_one_euro_filter(empty_trajectory, fps=30.0)
    
    assert smoothed.size == 0


def test_apply_one_euro_filter_expected_behaviour_for_zero_fps() -> None:
    """Ensures the filter falls back to a 30 FPS default (0.033 dt) when provided mathematically invalid frame rates."""
    trajectory: np.ndarray = np.random.randn(5, 33, 3).astype(float)
    smoothed: np.ndarray = apply_one_euro_filter(trajectory, fps=0.0)
    
    assert smoothed.shape == trajectory.shape


def test_calculate_kinematic_metrics_expected_behaviour_for_valid_trajectory() -> None:
    """Verifies the extraction of perceptual jitter and temporal smoothness from valid physical movements."""
    trajectory: np.ndarray = np.random.randn(50, 33, 3).astype(float)
    metrics: Dict[str, float] = calculate_kinematic_metrics(trajectory, fps=50.0)
    
    assert "perceptual_jitter" in metrics
    assert "temporal_smoothness" in metrics
    assert isinstance(metrics["perceptual_jitter"], float)
    assert isinstance(metrics["temporal_smoothness"], float)


def test_calculate_kinematic_metrics_expected_behaviour_for_insufficient_frames() -> None:
    """Ensures metric calculation gracefully returns zeroes when the trajectory lacks geometric depth for acceleration (needs 3 frames)."""
    # 2 frames is enough for velocity, but not enough for acceleration (jerk calculation)
    short_trajectory: np.ndarray = np.random.randn(2, 33, 3).astype(float)
    metrics: Dict[str, float] = calculate_kinematic_metrics(short_trajectory, fps=30.0)
    
    assert metrics["perceptual_jitter"] == 0.0
    assert metrics["temporal_smoothness"] == 0.0


def test_calculate_kinematic_metrics_expected_behaviour_for_processing_failure() -> None:
    """Ensures catastrophic mathematical failures return a safe zeroed dict rather than crashing the pipeline."""
    # Forcing a failure by passing a non-array object
    metrics: Dict[str, float] = calculate_kinematic_metrics(None, fps=30.0)  # type: ignore
    
    assert metrics["perceptual_jitter"] == 0.0
    assert metrics["temporal_smoothness"] == 0.0