import json
from pathlib import Path
from typing import Dict, Generator, Tuple

import pytest

from app.src.visual_edge.skeletal_kinematics.config import SkeletalConfig
from app.src.visual_edge.skeletal_kinematics.generator import (
    _evaluate_metric,
    generate_behavioral_descriptor,
)


@pytest.fixture
def valid_calibration_file() -> Generator[Path, None, None]:
    """Generates an actual JSON configuration file on disk for integrated evaluation."""
    file_path: Path = Path("app/models/skeletal_kinematics/test_generator_calibration.json")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    calibration_data: Dict[str, Dict[str, float]] = {
        "perceptual_jitter": {
            "very_low_threshold": 0.2,
            "low_threshold": 0.4,
            "high_threshold": 0.6,
            "very_high_threshold": 0.8
        },
        "temporal_smoothness": {
            "very_low_threshold": 0.1,
            "low_threshold": 0.3,
            "high_threshold": 0.5,
            "very_high_threshold": 0.7
        }
    }
    
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(calibration_data, file)

    yield file_path

    if file_path.exists():
        file_path.unlink()


@pytest.fixture
def active_config(valid_calibration_file: Path) -> SkeletalConfig:
    """Provides an active configuration bound to the test JSON file."""
    return SkeletalConfig(calibration_path=valid_calibration_file)


def test_evaluate_metric_expected_behaviour_for_very_low_quintile() -> None:
    """Tests proper routing for values strictly beneath the bottom boundary."""
    thresholds: Tuple[float, float, float, float] = (0.2, 0.4, 0.6, 0.8)
    cues: Tuple[str, str, str, str, str] = ("1", "2", "3", "4", "5")
    assert _evaluate_metric(0.1, thresholds, cues) == "1"


def test_evaluate_metric_expected_behaviour_for_high_quintile() -> None:
    """Tests proper routing for values bridging the middle-upper boundaries."""
    thresholds: Tuple[float, float, float, float] = (0.2, 0.4, 0.6, 0.8)
    cues: Tuple[str, str, str, str, str] = ("1", "2", "3", "4", "5")
    assert _evaluate_metric(0.7, thresholds, cues) == "4"


def test_evaluate_metric_expected_behaviour_for_very_high_quintile() -> None:
    """Tests proper routing for extreme values exceeding the top boundary."""
    thresholds: Tuple[float, float, float, float] = (0.2, 0.4, 0.6, 0.8)
    cues: Tuple[str, str, str, str, str] = ("1", "2", "3", "4", "5")
    assert _evaluate_metric(0.9, thresholds, cues) == "5"


def test_generate_behavioral_descriptor_expected_behaviour_for_valid_metrics(active_config: SkeletalConfig) -> None:
    """Tests successful translation logic given mathematically acceptable parameters."""
    result: str = generate_behavioral_descriptor(
        perceptual_jitter=0.5,
        temporal_smoothness=0.6,
        config=active_config
    )
    
    assert isinstance(result, str)
    assert "The candidate exhibits" in result
    assert "paired with" in result
    assert "moderate physical animation" in result
    assert "a composed, fluid posture" in result


def test_generate_behavioral_descriptor_expected_behaviour_for_extreme_values(active_config: SkeletalConfig) -> None:
    """Tests translation logic mapping boundary extremes simultaneously."""
    result: str = generate_behavioral_descriptor(
        perceptual_jitter=0.9,
        temporal_smoothness=0.01,
        config=active_config
    )
    
    assert "erratic physical shifts and postural swaying" in result
    assert "highly animated body language" in result


def test_generate_behavioral_descriptor_raises_for_invalid_jitter_type(active_config: SkeletalConfig) -> None:
    """Tests rejection of structural matrix typing violations."""
    with pytest.raises(TypeError, match="perceptual_jitter must be a numeric float"):
        generate_behavioral_descriptor(
            perceptual_jitter="0.5",  # type: ignore
            temporal_smoothness=0.5,
            config=active_config
        )


def test_generate_behavioral_descriptor_raises_for_invalid_smoothness_type(active_config: SkeletalConfig) -> None:
    """Tests rejection of boolean injection into the metric parameters."""
    with pytest.raises(TypeError, match="temporal_smoothness must be a numeric float"):
        generate_behavioral_descriptor(
            perceptual_jitter=0.5,
            temporal_smoothness=False,  # type: ignore
            config=active_config
        )


def test_generate_behavioral_descriptor_raises_for_negative_jitter(active_config: SkeletalConfig) -> None:
    """Tests absolute floor bounds against impossible geometric outputs."""
    with pytest.raises(ValueError, match="perceptual_jitter cannot be mathematically negative"):
        generate_behavioral_descriptor(
            perceptual_jitter=-0.1,
            temporal_smoothness=0.5,
            config=active_config
        )


def test_generate_behavioral_descriptor_raises_for_negative_smoothness(active_config: SkeletalConfig) -> None:
    """Tests absolute floor bounds against impossible geometric outputs."""
    with pytest.raises(ValueError, match="temporal_smoothness cannot be mathematically negative"):
        generate_behavioral_descriptor(
            perceptual_jitter=0.5,
            temporal_smoothness=-0.5,
            config=active_config
        )