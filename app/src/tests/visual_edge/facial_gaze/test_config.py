from pathlib import Path

import pytest

from app.src.visual_edge.facial_gaze.config import FacialConfig

POSTER_PATH: Path = Path("app/models/visual/poster_plus_latent.pth").resolve()
REGRESSOR_PATH: Path = Path("app/models/visual/facial_poster_regressor.joblib").resolve()
CALIBRATION_PATH: Path = Path("app/models/visual/facial_calibration.json").resolve()


def test_facialconfig_expected_behaviour_for_default_instantiation() -> None:
    """Verifies that the configuration matrix instantiates successfully and parses calibration data."""
    config: FacialConfig = FacialConfig()
    
    assert config.poster_weights_path == POSTER_PATH
    assert config.regressor_weights_path == REGRESSOR_PATH
    assert config.calibration_path == CALIBRATION_PATH
    assert config.device == "cuda:0"
    assert config.inference_batch_size == 16
    assert config.feature_dim == 1024
    assert isinstance(config.calibration_thresholds, dict)
    assert len(config.calibration_thresholds) > 0


def test_facialconfig_expected_behaviour_for_custom_gpu_initialization() -> None:
    """Verifies that custom parameters override defaults correctly while preserving type structures."""
    config: FacialConfig = FacialConfig(
        device="cuda:1",
        inference_batch_size=32,
        feature_dim=1024
    )
    
    assert config.device == "cuda:1"
    assert config.inference_batch_size == 32


def test_facialconfig_raises_for_missing_poster_weights() -> None:
    """Ensures instantiation abruptly halts if the deep learning checkpoint is missing from disk."""
    invalid_path: Path = POSTER_PATH.parent / "missing_poster.pth"
    
    with pytest.raises(FileNotFoundError, match="POSTER\\+\\+ weights not found"):
        FacialConfig(poster_weights_path=invalid_path)


def test_facialconfig_raises_for_missing_regressor_weights() -> None:
    """Ensures instantiation abruptly halts if the mathematical regressor artifact is missing from disk."""
    invalid_path: Path = REGRESSOR_PATH.parent / "missing_model.joblib"
    
    with pytest.raises(FileNotFoundError, match="Regressor weights not found"):
        FacialConfig(regressor_weights_path=invalid_path)


def test_facialconfig_raises_for_missing_calibration_path() -> None:
    """Ensures instantiation abruptly halts if the calibration JSON artifact is missing from disk."""
    invalid_path: Path = CALIBRATION_PATH.parent / "missing_calibration.json"
    
    with pytest.raises(FileNotFoundError, match="Calibration artifact not found"):
        FacialConfig(calibration_path=invalid_path)


def test_facialconfig_raises_for_invalid_device_string() -> None:
    """Ensures unmapped hardware device requests are strictly rejected."""
    with pytest.raises(ValueError, match="device must be a valid compute string"):
        FacialConfig(device="tpu:0")


def test_facialconfig_raises_for_negative_batch_size() -> None:
    """Ensures batch processing parameters maintain mathematical impossibility guards."""
    with pytest.raises(ValueError, match="inference_batch_size must be strictly positive."):
        FacialConfig(inference_batch_size=0)


def test_facialconfig_raises_for_corrupted_calibration_json() -> None:
    """Ensures instantiation fails if the calibration JSON file is structurally invalid or unreadable."""
    corrupted_path: Path = CALIBRATION_PATH.parent / "corrupted_test_calibration.json"
    corrupted_path.write_text("invalid json format {")
    
    try:
        with pytest.raises(RuntimeError, match="Failed to load calibration JSON"):
            FacialConfig(calibration_path=corrupted_path)
    finally:
        if corrupted_path.exists():
            corrupted_path.unlink()