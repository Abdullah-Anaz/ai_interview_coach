import logging
from typing import Dict

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


def _compute_smoothing_factor(time_delta: float, cutoff_frequency: float) -> float:
    """
    Calculates the dynamic exponential smoothing factor (alpha) for the One Euro filter.

    Args:
        time_delta (float): The time elapsed between the current and previous frame.
        cutoff_frequency (float): The dynamically adjusted cutoff frequency in Hz.

    Returns:
        float: The calculated alpha coefficient for exponential smoothing.
    """
    r_val: float = 2.0 * np.pi * cutoff_frequency * time_delta
    return r_val / (r_val + 1.0)


def apply_one_euro_filter(
    trajectory: np.ndarray,
    fps: float,
    min_cutoff: float = 1.0,
    beta: float = 0.004,
    d_cutoff: float = 1.0
) -> np.ndarray:
    """
    Applies the Casiez et al. One Euro Filter across a trajectory matrix.

    Args:
        trajectory (np.ndarray): Coordinate sequence matrix shaped (T, ...).
        fps (float): Frame rate of the input sequence.
        min_cutoff (float, optional): Minimum cutoff frequency. Defaults to 1.0.
        beta (float, optional): Cutoff slope (speed coefficient). Defaults to 0.004.
        d_cutoff (float, optional): Derivative cutoff frequency. Defaults to 1.0.

    Returns:
        np.ndarray: The smoothed trajectory array matching the exact input shape.

    Raises:
        TypeError: If the trajectory is not a valid numpy array.
    """
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be a valid numpy ndarray.")

    num_frames: int = trajectory.shape[0]
    if num_frames == 0:
        return trajectory

    dt: float = 1.0 / fps if fps > 0.0 else 0.033
    filtered_output: np.ndarray = np.zeros_like(trajectory, dtype=float)

    x_prev: np.ndarray = np.array(trajectory[0], dtype=float)
    dx_prev: np.ndarray = np.zeros_like(x_prev)
    filtered_output[0] = x_prev

    frame_idx: int
    for frame_idx in range(1, num_frames):
        x_curr: np.ndarray = trajectory[frame_idx]

        a_d: float = _compute_smoothing_factor(dt, d_cutoff)
        dx_curr: np.ndarray = (x_curr - x_prev) / dt
        dx_hat: np.ndarray = a_d * dx_curr + (1.0 - a_d) * dx_prev

        speed: float = float(np.linalg.norm(dx_hat) if dx_hat.ndim > 0 else np.abs(dx_hat))
        cutoff: float = min_cutoff + beta * speed

        a: float = _compute_smoothing_factor(dt, cutoff)
        x_hat: np.ndarray = a * x_curr + (1.0 - a) * x_prev

        filtered_output[frame_idx] = x_hat
        x_prev = x_hat
        dx_prev = dx_hat

    return filtered_output


def calculate_kinematic_metrics(
    raw_trajectory: np.ndarray,
    fps: float,
    min_cutoff: float = 1.0,
    beta: float = 0.004
) -> Dict[str, float]:
    """
    Computes perceptual jitter and temporal smoothness from a coordinate trajectory 
    by utilizing the One Euro filter as a mathematical baseline.

    Args:
        raw_trajectory (np.ndarray): Coordinate sequence matrix shaped (T, ...).
        fps (float): Native frame rate of the temporal sequence.
        min_cutoff (float, optional): Base filter threshold. Defaults to 1.0.
        beta (float, optional): Speed adaptation threshold. Defaults to 0.004.

    Returns:
        Dict[str, float]: The calculated 'perceptual_jitter' and 'temporal_smoothness' scores.
                          Returns 0.0 for both if sequence length is insufficient or calculation fails.
    """
    try:
        if not isinstance(raw_trajectory, np.ndarray):
            raise TypeError("Trajectory must be a valid numpy ndarray.")

        total_frames: int = raw_trajectory.shape[0]
        if total_frames < 3:
            logger.warning("Trajectory sequence under minimum length threshold (<3 frames).")
            return {"perceptual_jitter": 0.0, "temporal_smoothness": 0.0}

        dt: float = 1.0 / fps if fps > 0.0 else 0.033

        filtered_trajectory: np.ndarray = apply_one_euro_filter(
            trajectory=raw_trajectory,
            fps=fps,
            min_cutoff=min_cutoff,
            beta=beta
        )

        deltas: np.ndarray = (raw_trajectory - filtered_trajectory).reshape(total_frames, -1)
        jitter_score: float = float(np.mean(np.linalg.norm(deltas, axis=1)))

        velocities: np.ndarray = np.diff(raw_trajectory, axis=0) / dt
        accelerations: np.ndarray = np.diff(velocities, axis=0) / dt
        accel_flat: np.ndarray = accelerations.reshape(total_frames - 2, -1)
        smoothness_score: float = float(np.mean(np.linalg.norm(accel_flat, axis=1)))

        return {
            "perceptual_jitter": jitter_score,
            "temporal_smoothness": smoothness_score
        }

    except Exception as exc:
        logger.error("Failed calculating kinematic metrics: %s", exc)
        return {"perceptual_jitter": 0.0, "temporal_smoothness": 0.0}