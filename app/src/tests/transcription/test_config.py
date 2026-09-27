from pathlib import Path

import pytest

from app.src.transcription.config import TranscriptionConfig


def test_TranscriptionConfig_expected_behaviour_for_default_instance() -> None:
    """
    Validates that the out-of-the-box default configuration passes all structural checks.
    """
    # Should execute without raising any exceptions
    TranscriptionConfig()


def test_TranscriptionConfig_expected_behaviour_for_custom_valid_instance() -> None:
    """
    Validates that a correctly initialized custom configuration passes seamlessly.
    """
    TranscriptionConfig(
        model_repo_id="nvidia/parakeet-tdt-1.1b",
        model_filename="custom-model.nemo",
        local_weights_dir=Path("/tmp/models"),
        device="cpu",
        use_float16=False
    )


def test_TranscriptionConfig_raises_for_invalid_repo_id_type() -> None:
    """
    Validates type enforcement catching integers passed to string fields.
    """
    with pytest.raises(ValueError) as exc_info:
        TranscriptionConfig(model_repo_id=12345)  # type: ignore

    assert "must be a non-empty string" in str(exc_info.value)


def test_TranscriptionConfig_raises_for_empty_repo_id() -> None:
    """
    Validates constraint enforcement preventing empty strings for repository IDs.
    """
    with pytest.raises(ValueError) as exc_info:
        TranscriptionConfig(model_repo_id="   ")

    assert "must be a non-empty string" in str(exc_info.value)


def test_TranscriptionConfig_raises_for_empty_filename() -> None:
    """
    Validates constraint enforcement preventing empty strings for model filenames.
    """
    with pytest.raises(ValueError) as exc_info:
        TranscriptionConfig(model_filename="")

    assert "must be a non-empty string" in str(exc_info.value)


def test_TranscriptionConfig_raises_for_invalid_weights_dir_type() -> None:
    """
    Validates type enforcement catching strings passed to Path fields.
    """
    with pytest.raises(TypeError) as exc_info:
        TranscriptionConfig(local_weights_dir="app/models")  # type: ignore

    assert "local_weights_dir must be a Path" in str(exc_info.value)


def test_TranscriptionConfig_raises_for_empty_device() -> None:
    """
    Validates constraint enforcement preventing empty strings for compute devices.
    """
    with pytest.raises(ValueError) as exc_info:
        TranscriptionConfig(device=" ")

    assert "must be a non-empty string" in str(exc_info.value)


def test_TranscriptionConfig_raises_for_invalid_float16_type() -> None:
    """
    Validates type enforcement catching integers passed to boolean flags.
    """
    with pytest.raises(TypeError) as exc_info:
        TranscriptionConfig(use_float16=1)  # type: ignore

    assert "use_float16 must be a boolean" in str(exc_info.value)