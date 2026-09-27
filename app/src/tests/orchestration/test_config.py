import pytest

from app.src.orchestration.config import OrchestratorConfig


def test_OrchestratorConfig_expected_behaviour_for_valid_instance() -> None:
    """
    Validates that a correctly initialized configuration passes validation seamlessly.
    """
    # Instantiation should succeed without raising any errors
    OrchestratorConfig(
        audio_sample_rate=16000,
        video_fps=50.0,
        window_duration_sec=2.0,
        window_stride_sec=0.5,
        queue_timeout_sec=5.0
    )


def test_OrchestratorConfig_raises_for_invalid_field_type() -> None:
    """
    Validates type enforcement catching integers passed to float fields upon creation.
    """
    with pytest.raises(TypeError) as exc_info:
        OrchestratorConfig(
            audio_sample_rate=16000,
            video_fps=50,  # type: ignore
            window_duration_sec=2.0,
            window_stride_sec=0.5,
            queue_timeout_sec=5.0
        )

    assert "video_fps must be float" in str(exc_info.value)


def test_OrchestratorConfig_raises_for_negative_duration() -> None:
    """
    Validates constraint enforcement preventing mathematically impossible time boundaries.
    """
    with pytest.raises(ValueError) as exc_info:
        OrchestratorConfig(
            audio_sample_rate=16000,
            video_fps=50.0,
            window_duration_sec=-2.0,
            window_stride_sec=0.5,
            queue_timeout_sec=5.0
        )

    assert "window_duration_sec must be positive" in str(exc_info.value)


def test_OrchestratorConfig_raises_for_zero_sample_rate() -> None:
    """
    Validates constraint enforcement preventing mathematically impossible signal rates.
    """
    with pytest.raises(ValueError) as exc_info:
        OrchestratorConfig(
            audio_sample_rate=0,
            video_fps=50.0,
            window_duration_sec=2.0,
            window_stride_sec=0.5,
            queue_timeout_sec=5.0
        )

    assert "audio_sample_rate must be positive" in str(exc_info.value)