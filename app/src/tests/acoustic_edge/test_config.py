import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Dict, Generator

import pytest

from app.src.acoustic_edge.config import AcousticConfig


@pytest.fixture
def valid_calibration_data() -> Dict[str, float]:
    """Provides a valid baseline configuration dictionary for thresholds."""
    return {
        "very_low_threshold": 0.20,
        "low_threshold": 0.40,
        "high_threshold": 0.60,
        "very_high_threshold": 0.80
    }


@pytest.fixture
def valid_calibration_file(valid_calibration_data: Dict[str, float]) -> Generator[Path, None, None]:
    """Generates an actual JSON calibration file on the production path for testing."""
    file_path: Path = Path("app/models/acoustic/test_valid_calibration.json")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(valid_calibration_data, file)
    
    yield file_path
    
    if file_path.exists():
        file_path.unlink()


def test_AcousticConfig_expected_behaviour_for_valid_instantiation(valid_calibration_file: Path) -> None:
    """Tests if the config successfully initializes and parses boundaries from a real JSON file."""
    config = AcousticConfig(calibration_path=valid_calibration_file)

    assert config.wavlm_repo_id == "microsoft/wavlm-large"
    assert config.very_low_threshold == 0.20
    assert config.low_threshold == 0.40
    assert config.high_threshold == 0.60
    assert config.very_high_threshold == 0.80


def test_AcousticConfig_expected_behaviour_for_custom_instantiation(valid_calibration_file: Path) -> None:
    """Tests if the configuration accurately assigns custom structural parameters."""
    custom_model_path: Path = Path("custom/path/ridge_model.joblib")
    custom_cache_dir: Path = Path("custom/cache/dir")

    config = AcousticConfig(
        wavlm_repo_id="custom/wavlm-base",
        wavlm_revision="main",
        ridge_model_path=custom_model_path,
        calibration_path=valid_calibration_file,
        device="cpu",
        use_float16=False,
        local_weights_dir=custom_cache_dir
    )

    assert config.wavlm_repo_id == "custom/wavlm-base"
    assert config.device == "cpu"
    assert config.use_float16 is False


def test_AcousticConfig_expected_behaviour_for_immutability(valid_calibration_file: Path) -> None:
    """Tests if AcousticConfig strictly enforces structural immutability."""
    config = AcousticConfig(calibration_path=valid_calibration_file)

    with pytest.raises(FrozenInstanceError):
        config.device = "cpu"  # type: ignore


def test_AcousticConfig_raises_for_missing_calibration_file() -> None:
    """Tests if the initialization aborts safely when the calibration file is missing."""
    missing_path: Path = Path("app/models/acoustic/does_not_exist.json")
    
    with pytest.raises(FileNotFoundError, match="Calibration file missing"):
        AcousticConfig(calibration_path=missing_path)


def test_AcousticConfig_raises_for_invalid_json_format() -> None:
    """Tests if the initialization wraps JSON parsing errors into a RuntimeError."""
    corrupted_file: Path = Path("app/models/acoustic/test_corrupted_calibration.json")
    corrupted_file.parent.mkdir(parents=True, exist_ok=True)
    corrupted_file.write_text("this is not valid json")

    try:
        with pytest.raises(RuntimeError, match="Failed to load calibration JSON"):
            AcousticConfig(calibration_path=corrupted_file)
    finally:
        if corrupted_file.exists():
            corrupted_file.unlink()


def test_AcousticConfig_raises_for_missing_threshold_keys() -> None:
    """Tests if the configuration strictly validates the presence of all required thresholds."""
    incomplete_file: Path = Path("app/models/acoustic/test_incomplete.json")
    incomplete_file.parent.mkdir(parents=True, exist_ok=True)
    with open(incomplete_file, "w", encoding="utf-8") as file:
        json.dump({"very_low_threshold": 0.20}, file)

    try:
        with pytest.raises(ValueError, match="missing required key"):
            AcousticConfig(calibration_path=incomplete_file)
    finally:
        if incomplete_file.exists():
            incomplete_file.unlink()


def test_AcousticConfig_raises_for_invalid_threshold_types() -> None:
    """Tests if the configuration enforces numeric data types for all threshold values."""
    invalid_type_file: Path = Path("app/models/acoustic/test_invalid_type.json")
    invalid_type_file.parent.mkdir(parents=True, exist_ok=True)
    with open(invalid_type_file, "w", encoding="utf-8") as file:
        json.dump({
            "very_low_threshold": "0.20",
            "low_threshold": 0.40,
            "high_threshold": 0.60,
            "very_high_threshold": 0.80
        }, file)

    try:
        with pytest.raises(TypeError, match="must be a numerical value"):
            AcousticConfig(calibration_path=invalid_type_file)
    finally:
        if invalid_type_file.exists():
            invalid_type_file.unlink()


def test_AcousticConfig_raises_for_unordered_thresholds() -> None:
    """Tests if the configuration mathematically validates strictly ascending boundaries."""
    unordered_file: Path = Path("app/models/acoustic/test_unordered.json")
    unordered_file.parent.mkdir(parents=True, exist_ok=True)
    with open(unordered_file, "w", encoding="utf-8") as file:
        json.dump({
            "very_low_threshold": 0.50,
            "low_threshold": 0.40,
            "high_threshold": 0.60,
            "very_high_threshold": 0.80
        }, file)

    try:
        with pytest.raises(ValueError, match="strictly ascending"):
            AcousticConfig(calibration_path=unordered_file)
    finally:
        if unordered_file.exists():
            unordered_file.unlink()


def test_AcousticConfig_raises_for_invalid_device(valid_calibration_file: Path) -> None:
    """Tests if empty strings are rejected for hardware device mapping."""
    with pytest.raises(ValueError, match="device must be a non-empty string"):
        AcousticConfig(calibration_path=valid_calibration_file, device="   ")