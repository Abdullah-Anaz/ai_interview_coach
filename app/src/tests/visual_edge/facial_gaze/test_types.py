import pytest

from app.src.visual_edge.facial_gaze.types import FacialDescriptorPayload


def test_facialdescriptorpayload_expected_behaviour_for_valid_instantiation() -> None:
    """Verifies that the expanded OCEAN payload instantiates correctly when provided with structurally valid data."""
    payload: FacialDescriptorPayload = FacialDescriptorPayload(
        segment_id="SEG_001",
        start_time_sec=2.5,
        end_time_sec=5.0,
        behavioral_descriptor="Displays focused affect.",
        extraversion_score=0.72,
        neuroticism_score=0.45,
        agreeableness_score=0.60,
        conscientiousness_score=0.85,
        openness_score=0.55
    )
    
    assert payload.segment_id == "SEG_001"
    assert payload.start_time_sec == 2.5
    assert payload.end_time_sec == 5.0
    assert payload.behavioral_descriptor == "Displays focused affect."
    assert payload.extraversion_score == 0.72
    assert payload.neuroticism_score == 0.45
    assert payload.agreeableness_score == 0.60
    assert payload.conscientiousness_score == 0.85
    assert payload.openness_score == 0.55


def test_facialdescriptorpayload_expected_behaviour_for_zero_start_time() -> None:
    """Verifies that the payload correctly handles boundary limits such as a zero-second start time."""
    payload: FacialDescriptorPayload = FacialDescriptorPayload(
        segment_id="SEG_000",
        start_time_sec=0.0,
        end_time_sec=1.0,
        behavioral_descriptor="Initial frame.",
        extraversion_score=0.0,
        neuroticism_score=0.0,
        agreeableness_score=0.0,
        conscientiousness_score=0.0,
        openness_score=0.0
    )
    
    assert payload.start_time_sec == 0.0
    assert payload.end_time_sec == 1.0


def test_facialdescriptorpayload_raises_for_empty_segment_id() -> None:
    """Ensures instantiation fails if the segment identifier is structurally empty."""
    with pytest.raises(ValueError, match="segment_id cannot be an empty string."):
        FacialDescriptorPayload(
            segment_id="   ",
            start_time_sec=0.0,
            end_time_sec=2.0,
            behavioral_descriptor="Valid text.",
            extraversion_score=0.5,
            neuroticism_score=0.5,
            agreeableness_score=0.5,
            conscientiousness_score=0.5,
            openness_score=0.5
        )


def test_facialdescriptorpayload_raises_for_negative_start_time() -> None:
    """Ensures instantiation fails if the temporal start boundary is mathematically impossible."""
    with pytest.raises(ValueError, match="start_time_sec cannot be a negative value."):
        FacialDescriptorPayload(
            segment_id="SEG_002",
            start_time_sec=-1.5,
            end_time_sec=2.0,
            behavioral_descriptor="Valid text.",
            extraversion_score=0.5,
            neuroticism_score=0.5,
            agreeableness_score=0.5,
            conscientiousness_score=0.5,
            openness_score=0.5
        )


def test_facialdescriptorpayload_raises_for_inverted_temporal_boundaries() -> None:
    """Ensures instantiation fails if the end time does not strictly follow the start time."""
    with pytest.raises(ValueError, match="end_time_sec must be strictly greater than start_time_sec."):
        FacialDescriptorPayload(
            segment_id="SEG_003",
            start_time_sec=5.0,
            end_time_sec=2.0,
            behavioral_descriptor="Valid text.",
            extraversion_score=0.5,
            neuroticism_score=0.5,
            agreeableness_score=0.5,
            conscientiousness_score=0.5,
            openness_score=0.5
        )


def test_facialdescriptorpayload_raises_for_identical_temporal_boundaries() -> None:
    """Ensures instantiation fails if the temporal segment lacks depth (duration of zero)."""
    with pytest.raises(ValueError, match="end_time_sec must be strictly greater than start_time_sec."):
        FacialDescriptorPayload(
            segment_id="SEG_004",
            start_time_sec=3.0,
            end_time_sec=3.0,
            behavioral_descriptor="Valid text.",
            extraversion_score=0.5,
            neuroticism_score=0.5,
            agreeableness_score=0.5,
            conscientiousness_score=0.5,
            openness_score=0.5
        )


def test_facialdescriptorpayload_raises_for_invalid_descriptor_type() -> None:
    """Ensures instantiation fails if the translation payload is not a valid text string."""
    with pytest.raises(TypeError, match="behavioral_descriptor must be a string."):
        FacialDescriptorPayload(
            segment_id="SEG_005",
            start_time_sec=0.0,
            end_time_sec=2.0,
            behavioral_descriptor=None,  # type: ignore
            extraversion_score=0.5,
            neuroticism_score=0.5,
            agreeableness_score=0.5,
            conscientiousness_score=0.5,
            openness_score=0.5
        )


def test_facialdescriptorpayload_raises_for_invalid_metric_type() -> None:
    """Ensures instantiation fails if any of the regressed OCEAN scores are not numeric."""
    with pytest.raises(TypeError, match="neuroticism_score must be a numeric value."):
        FacialDescriptorPayload(
            segment_id="SEG_006",
            start_time_sec=0.0,
            end_time_sec=2.0,
            behavioral_descriptor="Valid text.",
            extraversion_score=0.5,
            neuroticism_score="High",  # type: ignore
            agreeableness_score=0.5,
            conscientiousness_score=0.5,
            openness_score=0.5
        )