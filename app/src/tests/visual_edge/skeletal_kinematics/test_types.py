from dataclasses import FrozenInstanceError

import pytest

from app.src.visual_edge.skeletal_kinematics.types import SkeletalDescriptorPayload


def test_SkeletalDescriptorPayload_expected_behaviour_for_valid_instantiation() -> None:
    """Tests successful creation with mathematically valid parameters."""
    payload = SkeletalDescriptorPayload(
        segment_id=42,
        start_time_sec=10.5,
        end_time_sec=12.5,
        behavioral_descriptor="displays fluid posture",
        perceptual_jitter=0.00045,
        temporal_smoothness=0.00085
    )

    assert payload.segment_id == 42
    assert payload.start_time_sec == 10.5
    assert payload.end_time_sec == 12.5
    assert payload.behavioral_descriptor == "displays fluid posture"
    assert payload.perceptual_jitter == 0.00045
    assert payload.temporal_smoothness == 0.00085


def test_SkeletalDescriptorPayload_expected_behaviour_for_integer_time_cast() -> None:
    """Tests if integer temporal values are securely accepted via the numeric type union."""
    payload = SkeletalDescriptorPayload(
        segment_id=1,
        start_time_sec=10,
        end_time_sec=11,
        behavioral_descriptor="displays fluid posture",
        perceptual_jitter=0.0,
        temporal_smoothness=0.0
    )

    assert payload.start_time_sec == 10
    assert payload.end_time_sec == 11


def test_SkeletalDescriptorPayload_raises_for_immutability() -> None:
    """Tests strict enforcement of frozen dataclass constraints."""
    payload = SkeletalDescriptorPayload(
        segment_id=1,
        start_time_sec=0.0,
        end_time_sec=1.0,
        behavioral_descriptor="valid",
        perceptual_jitter=0.0,
        temporal_smoothness=0.0
    )

    with pytest.raises(FrozenInstanceError):
        payload.segment_id = 2  # type: ignore


def test_SkeletalDescriptorPayload_raises_for_invalid_segment_id_type() -> None:
    """Tests structural rejection of non-integer identifier types."""
    with pytest.raises(TypeError, match="segment_id must be an integer"):
        SkeletalDescriptorPayload(
            segment_id="1",  # type: ignore
            start_time_sec=0.0,
            end_time_sec=1.0,
            behavioral_descriptor="valid",
            perceptual_jitter=0.0,
            temporal_smoothness=0.0
        )


def test_SkeletalDescriptorPayload_raises_for_negative_segment_id() -> None:
    """Tests rejection of mathematically invalid chronologies."""
    with pytest.raises(ValueError, match="segment_id must be non-negative"):
        SkeletalDescriptorPayload(
            segment_id=-5,
            start_time_sec=0.0,
            end_time_sec=1.0,
            behavioral_descriptor="valid",
            perceptual_jitter=0.0,
            temporal_smoothness=0.0
        )


def test_SkeletalDescriptorPayload_raises_for_invalid_start_time_type() -> None:
    """Tests strict numeric enforcement on the temporal start boundary."""
    with pytest.raises(TypeError, match="start_time_sec must be a numeric float"):
        SkeletalDescriptorPayload(
            segment_id=1,
            start_time_sec="0.0",  # type: ignore
            end_time_sec=1.0,
            behavioral_descriptor="valid",
            perceptual_jitter=0.0,
            temporal_smoothness=0.0
        )


def test_SkeletalDescriptorPayload_raises_for_negative_start_time() -> None:
    """Tests logical validation against negative temporal alignment."""
    with pytest.raises(ValueError, match="start_time_sec cannot be negative"):
        SkeletalDescriptorPayload(
            segment_id=1,
            start_time_sec=-1.5,
            end_time_sec=1.0,
            behavioral_descriptor="valid",
            perceptual_jitter=0.0,
            temporal_smoothness=0.0
        )


def test_SkeletalDescriptorPayload_raises_for_invalid_end_time_type() -> None:
    """Tests strict numeric enforcement on the temporal end boundary."""
    with pytest.raises(TypeError, match="end_time_sec must be a numeric float"):
        SkeletalDescriptorPayload(
            segment_id=1,
            start_time_sec=0.0,
            end_time_sec=None,  # type: ignore
            behavioral_descriptor="valid",
            perceptual_jitter=0.0,
            temporal_smoothness=0.0
        )


def test_SkeletalDescriptorPayload_raises_for_reversed_temporal_window() -> None:
    """Tests mathematical logic preventing inverted or zero-length time sequences."""
    with pytest.raises(ValueError, match="strictly greater than start_time_sec"):
        SkeletalDescriptorPayload(
            segment_id=1,
            start_time_sec=5.0,
            end_time_sec=4.0,
            behavioral_descriptor="valid",
            perceptual_jitter=0.0,
            temporal_smoothness=0.0
        )


def test_SkeletalDescriptorPayload_raises_for_zero_length_temporal_window() -> None:
    """Tests mathematical logic preventing zero-length temporal windows."""
    with pytest.raises(ValueError, match="strictly greater than start_time_sec"):
        SkeletalDescriptorPayload(
            segment_id=1,
            start_time_sec=5.0,
            end_time_sec=5.0,
            behavioral_descriptor="valid",
            perceptual_jitter=0.0,
            temporal_smoothness=0.0
        )


def test_SkeletalDescriptorPayload_raises_for_invalid_descriptor_type() -> None:
    """Tests rejection of non-string structures in the translation string."""
    with pytest.raises(TypeError, match="behavioral_descriptor must be a string"):
        SkeletalDescriptorPayload(
            segment_id=1,
            start_time_sec=0.0,
            end_time_sec=1.0,
            behavioral_descriptor=["invalid array"],  # type: ignore
            perceptual_jitter=0.0,
            temporal_smoothness=0.0
        )


def test_SkeletalDescriptorPayload_raises_for_empty_descriptor() -> None:
    """Tests rejection of semantically empty matrices flowing downstream."""
    with pytest.raises(ValueError, match="behavioral_descriptor cannot be empty"):
        SkeletalDescriptorPayload(
            segment_id=1,
            start_time_sec=0.0,
            end_time_sec=1.0,
            behavioral_descriptor="   ",
            perceptual_jitter=0.0,
            temporal_smoothness=0.0
        )


def test_SkeletalDescriptorPayload_raises_for_invalid_jitter_type() -> None:
    """Tests rejection of non-numeric jitter parameters."""
    with pytest.raises(TypeError, match="perceptual_jitter must be a numeric float"):
        SkeletalDescriptorPayload(
            segment_id=1,
            start_time_sec=0.0,
            end_time_sec=1.0,
            behavioral_descriptor="valid",
            perceptual_jitter=False,  # type: ignore
            temporal_smoothness=0.0
        )


def test_SkeletalDescriptorPayload_raises_for_invalid_smoothness_type() -> None:
    """Tests rejection of non-numeric smoothness parameters."""
    with pytest.raises(TypeError, match="temporal_smoothness must be a numeric float"):
        SkeletalDescriptorPayload(
            segment_id=1,
            start_time_sec=0.0,
            end_time_sec=1.0,
            behavioral_descriptor="valid",
            perceptual_jitter=0.0,
            temporal_smoothness="0.0"  # type: ignore
        )