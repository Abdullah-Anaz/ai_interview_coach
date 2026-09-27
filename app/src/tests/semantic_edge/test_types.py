import pytest

from app.src.semantic_edge.types import SemanticDescriptorPayload


def test_semantic_descriptor_payload_expected_behaviour_for_standard_instantiation() -> None:
    """Verifies that the payload correctly initializes and retains data under valid parameters."""
    payload = SemanticDescriptorPayload(
        segment_id=42,
        start_time_sec=1.5,
        end_time_sec=3.0,
        transcript_text="I led the database migration project."
    )
    
    assert payload is not None
    assert payload.segment_id == 42
    assert payload.start_time_sec == 1.5
    assert payload.end_time_sec == 3.0
    assert payload.transcript_text == "I led the database migration project."


def test_semantic_descriptor_payload_expected_behaviour_for_zero_start_time() -> None:
    """Verifies that a starting boundary of exactly absolute zero is mathematically accepted."""
    payload = SemanticDescriptorPayload(
        segment_id=0,
        start_time_sec=0.0,
        end_time_sec=5.2,
        transcript_text="Hello."
    )
    
    assert payload.start_time_sec == 0.0
    assert payload.segment_id == 0


def test_semantic_descriptor_payload_expected_behaviour_for_empty_transcript() -> None:
    """Verifies that an empty transcript string is structurally accepted for silent periods."""
    payload = SemanticDescriptorPayload(
        segment_id=5,
        start_time_sec=10.0,
        end_time_sec=12.5,
        transcript_text=""
    )
    
    assert payload.transcript_text == ""


def test_semantic_descriptor_payload_raises_for_invalid_segment_id_type() -> None:
    """Ensures structural rejection if the segment sequence ID deviates from strict integer typing."""
    with pytest.raises(TypeError, match="segment_id must be an integer."):
        SemanticDescriptorPayload(
            segment_id="12",  # type: ignore
            start_time_sec=0.0,
            end_time_sec=1.0,
            transcript_text="Test"
        )


def test_semantic_descriptor_payload_raises_for_negative_segment_id() -> None:
    """Ensures mathematical bounds block negative sequential identifiers."""
    with pytest.raises(ValueError, match="segment_id cannot be a negative integer."):
        SemanticDescriptorPayload(
            segment_id=-1,
            start_time_sec=0.0,
            end_time_sec=1.0,
            transcript_text="Test"
        )


def test_semantic_descriptor_payload_raises_for_invalid_start_time_type() -> None:
    """Ensures strict float/int type safety on the temporal lower boundary."""
    with pytest.raises(TypeError, match="start_time_sec must be a numeric value."):
        SemanticDescriptorPayload(
            segment_id=1,
            start_time_sec="0.0",  # type: ignore
            end_time_sec=1.0,
            transcript_text="Test"
        )


def test_semantic_descriptor_payload_raises_for_invalid_end_time_type() -> None:
    """Ensures strict float/int type safety on the temporal upper boundary."""
    with pytest.raises(TypeError, match="end_time_sec must be a numeric value."):
        SemanticDescriptorPayload(
            segment_id=1,
            start_time_sec=0.0,
            end_time_sec="1.0",  # type: ignore
            transcript_text="Test"
        )


def test_semantic_descriptor_payload_raises_for_negative_start_time() -> None:
    """Ensures absolute timestamps cannot cross into negative durations."""
    with pytest.raises(ValueError, match="start_time_sec cannot be a negative value."):
        SemanticDescriptorPayload(
            segment_id=1,
            start_time_sec=-2.5,
            end_time_sec=1.0,
            transcript_text="Test"
        )


def test_semantic_descriptor_payload_raises_for_inverted_temporal_boundaries() -> None:
    """Ensures the dataclass structurally rejects reversed time windows."""
    with pytest.raises(ValueError, match="end_time_sec must be strictly greater than start_time_sec."):
        SemanticDescriptorPayload(
            segment_id=1,
            start_time_sec=5.0,
            end_time_sec=2.0,
            transcript_text="Test"
        )


def test_semantic_descriptor_payload_raises_for_zero_duration_boundaries() -> None:
    """Ensures a window slice mathematically possesses a non-zero time delta."""
    with pytest.raises(ValueError, match="end_time_sec must be strictly greater than start_time_sec."):
        SemanticDescriptorPayload(
            segment_id=1,
            start_time_sec=3.0,
            end_time_sec=3.0,
            transcript_text="Test"
        )


def test_semantic_descriptor_payload_raises_for_invalid_transcript_type() -> None:
    """Ensures structural rejection if the semantic payload receives non-string data types."""
    with pytest.raises(TypeError, match="transcript_text must be a string."):
        SemanticDescriptorPayload(
            segment_id=1,
            start_time_sec=0.0,
            end_time_sec=1.0,
            transcript_text=None  # type: ignore
        )