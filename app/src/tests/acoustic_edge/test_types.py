import pytest

from app.src.acoustic_edge.types import AcousticDescriptorPayload


def test_acoustic_descriptor_payload_expected_behaviour_for_valid_instantiation() -> None:
    """Tests the successful instantiation of the payload with correct data types, boundaries, and raw scores."""
    payload = AcousticDescriptorPayload(
        segment_id=1,
        start_time_sec=0.0,
        end_time_sec=5.0,
        vocal_cues=("high extraversion", "low neuroticism"),
        extraversion_score=0.8,
        neuroticism_score=0.2,
        agreeableness_score=0.5,
        conscientiousness_score=0.9,
        openness_score=0.6
    )
    
    assert payload.segment_id == 1
    assert payload.start_time_sec == 0.0
    assert payload.end_time_sec == 5.0
    assert isinstance(payload.vocal_cues, tuple)
    assert len(payload.vocal_cues) == 2
    assert payload.extraversion_score == 0.8
    assert payload.conscientiousness_score == 0.9


def test_acoustic_descriptor_payload_expected_behaviour_for_edge_case_zero_duration() -> None:
    """Tests the boundary condition where end_time_sec is fractionally larger than start_time_sec."""
    payload = AcousticDescriptorPayload(
        segment_id=0,
        start_time_sec=1.123,
        end_time_sec=1.124,
        vocal_cues=("moderate levels",),
        extraversion_score=0.5,
        neuroticism_score=0.5,
        agreeableness_score=0.5,
        conscientiousness_score=0.5,
        openness_score=0.5
    )
    
    assert payload.end_time_sec > payload.start_time_sec


def test_acoustic_descriptor_payload_raises_for_negative_segment_id() -> None:
    """Tests if the dataclass strictly prohibits negative segment identifiers."""
    with pytest.raises(ValueError, match="segment_id must be a non-negative integer."):
        AcousticDescriptorPayload(
            segment_id=-1,
            start_time_sec=0.0,
            end_time_sec=1.0,
            vocal_cues=("high extraversion",),
            extraversion_score=0.5,
            neuroticism_score=0.5,
            agreeableness_score=0.5,
            conscientiousness_score=0.5,
            openness_score=0.5
        )


def test_acoustic_descriptor_payload_raises_for_inverted_time_boundaries() -> None:
    """Tests if the dataclass strictly prohibits end times that occur before start times."""
    with pytest.raises(ValueError, match="end_time_sec must be strictly greater than start_time_sec."):
        AcousticDescriptorPayload(
            segment_id=1,
            start_time_sec=5.0,
            end_time_sec=2.0,
            vocal_cues=("high extraversion",),
            extraversion_score=0.5,
            neuroticism_score=0.5,
            agreeableness_score=0.5,
            conscientiousness_score=0.5,
            openness_score=0.5
        )


def test_acoustic_descriptor_payload_raises_for_invalid_cues_type() -> None:
    """Tests if the dataclass strictly enforces the tuple[str, ...] type for cues."""
    with pytest.raises(TypeError, match="vocal_cues must be a tuple of strings."):
        AcousticDescriptorPayload(
            segment_id=1,
            start_time_sec=0.0,
            end_time_sec=1.0,
            vocal_cues=["high extraversion"],  # type: ignore
            extraversion_score=0.5,
            neuroticism_score=0.5,
            agreeableness_score=0.5,
            conscientiousness_score=0.5,
            openness_score=0.5
        )


def test_acoustic_descriptor_payload_raises_for_empty_strings_in_cues() -> None:
    """Tests if the dataclass prohibits empty strings within the cues tuple."""
    with pytest.raises(ValueError, match="Vocal cues cannot be empty strings."):
        AcousticDescriptorPayload(
            segment_id=1,
            start_time_sec=0.0,
            end_time_sec=1.0,
            vocal_cues=("high extraversion", "   "),
            extraversion_score=0.5,
            neuroticism_score=0.5,
            agreeableness_score=0.5,
            conscientiousness_score=0.5,
            openness_score=0.5
        )


def test_acoustic_descriptor_payload_raises_for_invalid_score_type() -> None:
    """Tests if the dataclass structurally rejects non-numerical raw telemetry."""
    with pytest.raises(TypeError, match="OCEAN scores must be numerical values."):
        AcousticDescriptorPayload(
            segment_id=1,
            start_time_sec=0.0,
            end_time_sec=1.0,
            vocal_cues=("high extraversion",),
            extraversion_score="0.8",  # type: ignore
            neuroticism_score=0.5,
            agreeableness_score=0.5,
            conscientiousness_score=0.5,
            openness_score=0.5
        )