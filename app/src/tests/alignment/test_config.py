from pathlib import Path
import pytest

from app.src.alignment.config import AlignmentConfig


def test_AlignmentConfig_expected_behaviour_for_default_instance() -> None:
    """
    Validates that the out-of-the-box default configuration passes all structural checks.
    """
    AlignmentConfig()


def test_AlignmentConfig_expected_behaviour_for_custom_valid_instance() -> None:
    """
    Validates that a correctly initialized custom configuration passes seamlessly.
    """
    AlignmentConfig(
        model_repo_id="nvidia/custom-ctc",
        model_filename="custom.nemo",
        local_weights_dir=Path("/tmp/alignment"),
        device="cpu",
        use_float16=False,
        video_fps=60,
        time_stride_sec=0.05
    )


def test_AlignmentConfig_raises_for_invalid_repo_id_type() -> None:
    """
    Validates type safety guard preventing incorrect types assigned to repo id.
    """
    with pytest.raises(TypeError) as exc_info:
        AlignmentConfig(model_repo_id=123)  # type: ignore

    assert "model_repo_id must be a string" in str(exc_info.value)


def test_AlignmentConfig_raises_for_empty_filename() -> None:
    """
    Validates constraint enforcement preventing empty strings for model filenames.
    """
    with pytest.raises(ValueError) as exc_info:
        AlignmentConfig(model_filename="   ")

    assert "model_filename must be a non-empty string" in str(exc_info.value)


def test_AlignmentConfig_raises_for_invalid_weights_dir_type() -> None:
    """
    Validates type enforcement catching strings passed to Path fields.
    """
    with pytest.raises(TypeError) as exc_info:
        AlignmentConfig(local_weights_dir="/invalid/path/type")  # type: ignore

    assert "local_weights_dir must be a Path object" in str(exc_info.value)


def test_AlignmentConfig_raises_for_zero_video_fps() -> None:
    """
    Validates mathematical constraints preventing zero or negative framerates.
    """
    with pytest.raises(ValueError) as exc_info:
        AlignmentConfig(video_fps=0)

    assert "video_fps must be strictly positive" in str(exc_info.value)


def test_AlignmentConfig_raises_for_negative_time_stride() -> None:
    """
    Validates mathematical constraints preventing zero or negative time strides.
    """
    with pytest.raises(ValueError) as exc_info:
        AlignmentConfig(time_stride_sec=-0.04)

    assert "time_stride_sec must be strictly positive" in str(exc_info.value)