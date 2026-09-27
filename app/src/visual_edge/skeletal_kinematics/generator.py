from typing import Tuple

from app.src.visual_edge.skeletal_kinematics.config import SkeletalConfig


def _evaluate_metric(
    score: float,
    thresholds: Tuple[float, float, float, float],
    descriptors: Tuple[str, str, str, str, str]
) -> str:
    """
    Evaluates a continuous numerical score against strictly ascending boundaries.

    Args:
        score (float): The continuous visual proxy metric calculated from the temporal window.
        thresholds (Tuple[float, float, float, float]): The 20th, 40th, 60th, and 80th percentiles.
        descriptors (Tuple[str, str, str, str, str]): The five text cues mapping to each quintile.

    Returns:
        str: The selected natural language description based on the statistical bucket.
    """
    very_low, low, high, very_high = thresholds

    if score < very_low:
        return descriptors[0]
    if score < low:
        return descriptors[1]
    if score < high:
        return descriptors[2]
    if score < very_high:
        return descriptors[3]
    return descriptors[4]


def generate_behavioral_descriptor(
    perceptual_jitter: float,
    temporal_smoothness: float,
    config: SkeletalConfig
) -> str:
    """
    Translates raw skeletal kinematics into a discrete BeMERC natural language payload.

    Args:
        perceptual_jitter (float): The high-frequency spatial variance score.
        temporal_smoothness (float): The low-frequency spatial trajectory score.
        config (SkeletalConfig): The active structural configuration containing statistical bounds.

    Returns:
        str: A grammatically complete sentence detailing the candidate's physical behavior.

    Raises:
        TypeError: If metric inputs fail strict numeric verification.
        ValueError: If metric inputs fall below absolute mathematical minimums.
        RuntimeError: If evaluation encounters an internal logic failure.
    """
    if not isinstance(perceptual_jitter, (int, float)) or isinstance(perceptual_jitter, bool):
        raise TypeError("perceptual_jitter must be a numeric float.")
    if perceptual_jitter < 0.0:
        raise ValueError("perceptual_jitter cannot be mathematically negative.")

    if not isinstance(temporal_smoothness, (int, float)) or isinstance(temporal_smoothness, bool):
        raise TypeError("temporal_smoothness must be a numeric float.")
    if temporal_smoothness < 0.0:
        raise ValueError("temporal_smoothness cannot be mathematically negative.")

    try:
        jitter_thresholds: Tuple[float, float, float, float] = (
            config.jitter_very_low,
            config.jitter_low,
            config.jitter_high,
            config.jitter_very_high
        )
        
        jitter_cues: Tuple[str, str, str, str, str] = (
            "minimal physical animation",
            "subdued gestural movement",
            "moderate physical animation",
            "active gestural shifts",
            "highly animated body language"
        )

        smoothness_thresholds: Tuple[float, float, float, float] = (
            config.smoothness_very_low,
            config.smoothness_low,
            config.smoothness_high,
            config.smoothness_very_high
        )
        
        smoothness_cues: Tuple[str, str, str, str, str] = (
            "erratic physical shifts and postural swaying",
            "rigid or jerky postural movements",
            "standard postural stability",
            "a composed, fluid posture",
            "a highly composed, fluid posture with controlled movements"
        )

        jitter_desc: str = _evaluate_metric(
            score=float(perceptual_jitter),
            thresholds=jitter_thresholds,
            descriptors=jitter_cues
        )
        
        smoothness_desc: str = _evaluate_metric(
            score=float(temporal_smoothness),
            thresholds=smoothness_thresholds,
            descriptors=smoothness_cues
        )

        return f"The candidate exhibits {smoothness_desc} paired with {jitter_desc}."

    except Exception as exc:
        raise RuntimeError(f"Descriptor generation failed: {exc}") from exc