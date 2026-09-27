from pathlib import Path

import pytest
import torch

from app.src.stream_ingestion.config import IngestionConfig, IngestionMode


@pytest.fixture
def hardware_check() -> None:
    """Validates runtime environments but enforces CPU operations for structural configuration allocation."""
    _ = torch.cuda.is_available()


# =====================================================================
# EXPECTED BEHAVIOUR & EDGE CASES
# =====================================================================

def test_ingestion_config_expected_behaviour_for_live_mode(hardware_check: None) -> None:
    """Tests flawless initialization when targeting real-time hardware bounds."""
    config = IngestionConfig(
        mode=IngestionMode.LIVE,
        video_fps=30.0,
        audio_sample_rate=44100,
        camera_device_index=1,
        audio_device_index=2
    )
    
    assert config.mode == IngestionMode.LIVE
    assert config.video_fps == 30.0
    assert config.audio_sample_rate == 44100
    assert config.camera_device_index == 1
    assert config.audio_device_index == 2
    assert config.simulator_file_path is None


def test_ingestion_config_expected_behaviour_for_simulator_mode(hardware_check: None) -> None:
    """Tests flawless initialization when targeting a deterministic media file."""
    test_path: Path = Path("/tmp/test_interview.mp4")
    config = IngestionConfig(
        mode=IngestionMode.SIMULATOR,
        simulator_file_path=test_path
    )
    
    assert config.mode == IngestionMode.SIMULATOR
    assert config.simulator_file_path == test_path
    assert config.video_fps == 50.0  # Default validation
    assert config.audio_sample_rate == 16000  # Default validation


def test_ingestion_config_expected_behaviour_for_zero_device_indices() -> None:
    """Tests the boundary condition where OS hardware arrays sit exactly at index zero."""
    config = IngestionConfig(
        mode=IngestionMode.LIVE,
        camera_device_index=0,
        audio_device_index=0
    )
    assert config.camera_device_index == 0
    assert config.audio_device_index == 0


# =====================================================================
# ERROR CASES: TYPE POLLUTION
# =====================================================================

def test_ingestion_config_raises_for_invalid_mode_type() -> None:
    """Tests structural rejection of unmapped string strategies."""
    with pytest.raises(TypeError, match="mode must be a valid IngestionMode enum."):
        IngestionConfig(mode="live")  # type: ignore


def test_ingestion_config_raises_for_invalid_video_fps_type() -> None:
    """Tests structural rejection of integers passed to floating-point strict fields."""
    with pytest.raises(TypeError, match="video_fps must be a strictly typed float."):
        IngestionConfig(mode=IngestionMode.LIVE, video_fps=50)  # type: ignore


def test_ingestion_config_raises_for_invalid_sample_rate_type() -> None:
    """Tests structural rejection of floats or booleans masquerading as integers."""
    with pytest.raises(TypeError, match="audio_sample_rate must be a strictly typed integer."):
        IngestionConfig(mode=IngestionMode.LIVE, audio_sample_rate=16000.0)  # type: ignore

    with pytest.raises(TypeError, match="audio_sample_rate must be a strictly typed integer."):
        IngestionConfig(mode=IngestionMode.LIVE, audio_sample_rate=True)  # type: ignore


def test_ingestion_config_raises_for_invalid_camera_index_type() -> None:
    """Tests structural rejection of invalid pointer types for hardware arrays."""
    with pytest.raises(TypeError, match="camera_device_index must be a strictly typed integer."):
        IngestionConfig(mode=IngestionMode.LIVE, camera_device_index="0")  # type: ignore


def test_ingestion_config_raises_for_invalid_simulator_path_type() -> None:
    """Tests structural rejection of raw strings masking as pathlib objects."""
    with pytest.raises(TypeError, match="simulator_file_path must be a valid pathlib.Path object."):
        IngestionConfig(
            mode=IngestionMode.SIMULATOR,
            simulator_file_path="/tmp/video.mp4"  # type: ignore
        )


# =====================================================================
# ERROR CASES: VALUE BOUNDARY VIOLATIONS
# =====================================================================

def test_ingestion_config_raises_for_zero_video_fps() -> None:
    """Tests rejection of impossible visual extraction frequencies."""
    with pytest.raises(ValueError, match="video_fps must be strictly positive."):
        IngestionConfig(mode=IngestionMode.LIVE, video_fps=0.0)


def test_ingestion_config_raises_for_negative_video_fps() -> None:
    """Tests rejection of inverted visual extraction frequencies."""
    with pytest.raises(ValueError, match="video_fps must be strictly positive."):
        IngestionConfig(mode=IngestionMode.LIVE, video_fps=-24.0)


def test_ingestion_config_raises_for_zero_sample_rate() -> None:
    """Tests rejection of mathematically impossible acoustic sampling."""
    with pytest.raises(ValueError, match="audio_sample_rate must be strictly positive."):
        IngestionConfig(mode=IngestionMode.LIVE, audio_sample_rate=0)


def test_ingestion_config_raises_for_negative_camera_index() -> None:
    """Tests rejection of out-of-bounds negative hardware pointers."""
    with pytest.raises(ValueError, match="camera_device_index must be a non-negative integer."):
        IngestionConfig(mode=IngestionMode.LIVE, camera_device_index=-1)


def test_ingestion_config_raises_for_negative_audio_index() -> None:
    """Tests rejection of out-of-bounds negative hardware pointers."""
    with pytest.raises(ValueError, match="audio_device_index must be a non-negative integer."):
        IngestionConfig(mode=IngestionMode.LIVE, audio_device_index=-1)


def test_ingestion_config_raises_for_missing_simulator_path() -> None:
    """Tests strategy dependency enforcement when simulator mode is activated without a target."""
    with pytest.raises(ValueError, match="simulator_file_path is explicitly required when mode is SIMULATOR."):
        IngestionConfig(mode=IngestionMode.SIMULATOR, simulator_file_path=None)