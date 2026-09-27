import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

from app.src.visual_edge.skeletal_kinematics.config import SkeletalConfig


@pytest.fixture
def valid_calibration_data() -> Dict[str, Dict[str, float]]:
    """Provides mathematically ascending thresholds simulating Human-Omni metrics."""
    return {
        "perceptual_jitter": {
            "very_low_threshold": 4.0e-07,
            "low_threshold": 4.2e-07,
            "high_threshold": 4.4e-07,
            "very_high_threshold": 4.6e-07
        },
        "temporal_smoothness": {
            "very_low_threshold": 0.0006,
            "low_threshold": 0.0007,
            "high_threshold": 0.0008,
            "very_high_threshold": 0.0009
        }
    }


@pytest.fixture
def valid_calibration_file(valid_calibration_data: Dict[str, Dict[str, float]]) -> Generator[Path, None, None]:
    """Generates an actual JSON calibration file on the production path for integration testing."""
    file_path: Path = Path("app/models/visual/test_valid_calibration.json")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(valid_calibration_data, file)

    yield file_path

    if file_path.exists():
        file_path.unlink()


@pytest.fixture
def valid_weights_file() -> Generator[Path, None, None]:
    """Generates an empty physical file representing the localized .pth weights to pass structural existence checks."""
    file_path: Path = Path("src/models/weights/test_human_omni.pth")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()

    yield file_path

    if file_path.exists():
        file_path.unlink()


def test_SkeletalConfig_expected_behaviour_for_valid_instantiation(
    valid_calibration_file: Path, valid_weights_file: Path
) -> None:
    """Tests if the configuration securely loads nested JSON matrices and validates local model paths."""
    config = SkeletalConfig(
        calibration_path=valid_calibration_file,
        weights_path=valid_weights_file
    )

    assert config.temporal_window_size == 50
    assert config.feature_dim == 512
    assert config.num_joints == 33
    assert config.jitter_low == 4.2e-07
    assert config.smoothness_high == 0.0008


def test_SkeletalConfig_expected_behaviour_for_custom_instantiation(
    valid_calibration_file: Path, valid_weights_file: Path
) -> None:
    """Tests if the config assigns user-defined hardware and matrix boundaries properly."""
    config = SkeletalConfig(
        calibration_path=valid_calibration_file,
        weights_path=valid_weights_file,
        device="cuda:0",
        temporal_window_size=100,
        feature_dim=256,
        num_joints=17
    )

    assert config.device == "cuda:0"
    assert config.temporal_window_size == 100
    assert config.feature_dim == 256
    assert config.num_joints == 17
    assert config.smoothness_very_low == 0.0006


def test_SkeletalConfig_raises_for_missing_weights_file(valid_calibration_file: Path) -> None:
    """Tests failure state if the Human-Omni generation script hasn't output the weights."""
    missing_weights: Path = Path("src/models/weights/does_not_exist.pth")

    with pytest.raises(FileNotFoundError, match="Model weights missing"):
        SkeletalConfig(
            calibration_path=valid_calibration_file,
            weights_path=missing_weights
        )


def test_SkeletalConfig_raises_for_immutability(
    valid_calibration_file: Path, valid_weights_file: Path
) -> None:
    """Tests if SkeletalConfig strictly enforces the frozen structural state post-initialization."""
    config = SkeletalConfig(
        calibration_path=valid_calibration_file,
        weights_path=valid_weights_file
    )

    with pytest.raises(FrozenInstanceError):
        config.device = "gpu"  # type: ignore


def test_SkeletalConfig_raises_for_invalid_device(
    valid_calibration_file: Path, valid_weights_file: Path
) -> None:
    """Tests if blank strings are structurally rejected for compute hardware."""
    with pytest.raises(ValueError, match="device must be a non-empty string"):
        SkeletalConfig(
            calibration_path=valid_calibration_file,
            weights_path=valid_weights_file,
            device="   "
        )


def test_SkeletalConfig_raises_for_invalid_window_size_type(
    valid_calibration_file: Path, valid_weights_file: Path
) -> None:
    """Tests strict numeric isolation against boolean configuration injections."""
    with pytest.raises(TypeError, match="temporal_window_size must be an integer"):
        SkeletalConfig(
            calibration_path=valid_calibration_file,
            weights_path=valid_weights_file,
            temporal_window_size=True  # type: ignore
        )


def test_SkeletalConfig_raises_for_negative_window_size(
    valid_calibration_file: Path, valid_weights_file: Path
) -> None:
    """Tests physical rejection of zero or inverted frame buffer lengths."""
    with pytest.raises(ValueError, match="temporal_window_size must be strictly positive"):
        SkeletalConfig(
            calibration_path=valid_calibration_file,
            weights_path=valid_weights_file,
            temporal_window_size=0
        )


def test_SkeletalConfig_raises_for_missing_calibration_file(valid_weights_file: Path) -> None:
    """Tests safe abort sequencing when the targeted BeMERC thresholds do not exist."""
    missing_path: Path = Path("app/models/visual/does_not_exist.json")

    with pytest.raises(FileNotFoundError, match="Calibration file missing"):
        SkeletalConfig(
            calibration_path=missing_path,
            weights_path=valid_weights_file
        )


def test_SkeletalConfig_raises_for_invalid_json_format(valid_weights_file: Path) -> None:
    """Tests if corrupt serialization files trigger a clear framework halt."""
    corrupted_file: Path = Path("app/models/visual/test_corrupt.json")
    corrupted_file.parent.mkdir(parents=True, exist_ok=True)
    corrupted_file.write_text("invalid format")

    try:
        with pytest.raises(RuntimeError, match="Failed to load calibration JSON"):
            SkeletalConfig(
                calibration_path=corrupted_file,
                weights_path=valid_weights_file
            )
    finally:
        if corrupted_file.exists():
            corrupted_file.unlink()


def test_SkeletalConfig_raises_for_missing_parent_keys(valid_weights_file: Path) -> None:
    """Tests structural validation isolating the jitter and smoothness nodes."""
    missing_parent_file: Path = Path("app/models/visual/test_missing_parent.json")
    missing_parent_file.parent.mkdir(parents=True, exist_ok=True)
    with open(missing_parent_file, "w", encoding="utf-8") as file:
        json.dump({"perceptual_jitter": {}}, file)

    try:
        with pytest.raises(ValueError, match="missing required parent object"):
            SkeletalConfig(
                calibration_path=missing_parent_file,
                weights_path=valid_weights_file
            )
    finally:
        if missing_parent_file.exists():
            missing_parent_file.unlink()


def test_SkeletalConfig_raises_for_missing_threshold_keys(
    valid_calibration_data: Dict[str, Dict[str, float]], valid_weights_file: Path
) -> None:
    """Tests internal mapping integrity forcing all four quintiles to be present."""
    del valid_calibration_data["temporal_smoothness"]["very_low_threshold"]
    
    incomplete_file: Path = Path("app/models/visual/test_incomplete.json")
    incomplete_file.parent.mkdir(parents=True, exist_ok=True)
    with open(incomplete_file, "w", encoding="utf-8") as file:
        json.dump(valid_calibration_data, file)

    try:
        with pytest.raises(ValueError, match="missing required threshold key: 'very_low_threshold'"):
            SkeletalConfig(
                calibration_path=incomplete_file,
                weights_path=valid_weights_file
            )
    finally:
        if incomplete_file.exists():
            incomplete_file.unlink()


def test_SkeletalConfig_raises_for_invalid_threshold_types(
    valid_calibration_data: Dict[str, Any], valid_weights_file: Path
) -> None:
    """Tests runtime rejection of improperly formatted configuration scalars."""
    valid_calibration_data["perceptual_jitter"]["high_threshold"] = "0.44e-07"

    invalid_type_file: Path = Path("app/models/visual/test_invalid_type.json")
    invalid_type_file.parent.mkdir(parents=True, exist_ok=True)
    with open(invalid_type_file, "w", encoding="utf-8") as file:
        json.dump(valid_calibration_data, file)

    try:
        with pytest.raises(TypeError, match="must be a numerical float"):
            SkeletalConfig(
                calibration_path=invalid_type_file,
                weights_path=valid_weights_file
            )
    finally:
        if invalid_type_file.exists():
            invalid_type_file.unlink()


def test_SkeletalConfig_raises_for_unordered_thresholds(
    valid_calibration_data: Dict[str, Dict[str, float]], valid_weights_file: Path
) -> None:
    """Tests mathematical topology evaluating the sequential distribution."""
    valid_calibration_data["temporal_smoothness"]["low_threshold"] = 0.9999
    
    unordered_file: Path = Path("app/models/visual/test_unordered.json")
    unordered_file.parent.mkdir(parents=True, exist_ok=True)
    with open(unordered_file, "w", encoding="utf-8") as file:
        json.dump(valid_calibration_data, file)

    try:
        with pytest.raises(ValueError, match="must be strictly ascending"):
            SkeletalConfig(
                calibration_path=unordered_file,
                weights_path=valid_weights_file
            )
    finally:
        if unordered_file.exists():
            unordered_file.unlink()