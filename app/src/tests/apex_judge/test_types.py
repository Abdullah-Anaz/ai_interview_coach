import pytest
import torch

from app.src.apex_judge.types import CoachingFeedbackPayload


@pytest.fixture
def hardware_check() -> None:
    """Validates runtime environments but enforces CPU operations for structural data allocation."""
    _ = torch.cuda.is_available()


# =====================================================================
# EXPECTED BEHAVIOUR & EDGE CASES
# =====================================================================

def test_coaching_feedback_payload_expected_behaviour_for_valid_instantiation(hardware_check: None) -> None:
    """Tests the successful instantiation of the payload with standard valid data."""
    payload = CoachingFeedbackPayload(
        segment_id=42,
        feedback_text="### 🎯 Executive Summary\nExcellent structural pacing."
    )
    
    assert payload.segment_id == 42
    assert payload.feedback_text == "### 🎯 Executive Summary\nExcellent structural pacing."
    assert isinstance(payload.segment_id, int)
    assert isinstance(payload.feedback_text, str)


def test_coaching_feedback_payload_expected_behaviour_for_zero_segment_id() -> None:
    """Tests the boundary condition where the sequence identifier is exactly zero."""
    payload = CoachingFeedbackPayload(
        segment_id=0,
        feedback_text="Valid feedback."
    )
    
    assert payload.segment_id == 0
    assert payload.feedback_text == "Valid feedback."


def test_coaching_feedback_payload_expected_behaviour_for_multiline_markdown() -> None:
    """Tests if the data contract safely handles complex, multi-line markdown with special characters."""
    complex_text: str = (
        "### 🎭 Multimodal Congruence Analysis\n"
        "- **Semantic Rewrite:** Use 'I engineered' instead of 'I did'.\n"
        "- **Delivery Correction:** 🗣️ Lower your vocal pitch."
    )
    payload = CoachingFeedbackPayload(
        segment_id=1,
        feedback_text=complex_text
    )
    
    assert "### 🎭" in payload.feedback_text
    assert "🗣️" in payload.feedback_text
    assert "\n" in payload.feedback_text


# =====================================================================
# ERROR CASES (TYPE POLLUTION & VALUE BOUNDARIES)
# =====================================================================

def test_coaching_feedback_payload_raises_for_negative_segment_id() -> None:
    """Tests if the dataclass strictly prohibits negative segment identifiers."""
    with pytest.raises(ValueError, match="segment_id must be a non-negative integer."):
        CoachingFeedbackPayload(
            segment_id=-1,
            feedback_text="Actionable advice."
        )


def test_coaching_feedback_payload_raises_for_float_segment_id() -> None:
    """Tests if the dataclass structurally rejects floating-point numbers for sequence identifiers."""
    with pytest.raises(TypeError, match="segment_id must be a strictly typed integer."):
        CoachingFeedbackPayload(
            segment_id=1.0,  # type: ignore
            feedback_text="Actionable advice."
        )


def test_coaching_feedback_payload_raises_for_boolean_segment_id() -> None:
    """Tests if the dataclass strictly traps boolean values masking as integers."""
    with pytest.raises(TypeError, match="segment_id must be a strictly typed integer."):
        CoachingFeedbackPayload(
            segment_id=False,  # type: ignore
            feedback_text="Actionable advice."
        )


def test_coaching_feedback_payload_raises_for_string_segment_id() -> None:
    """Tests if the dataclass structurally rejects string types for sequence identifiers."""
    with pytest.raises(TypeError, match="segment_id must be a strictly typed integer."):
        CoachingFeedbackPayload(
            segment_id="1",  # type: ignore
            feedback_text="Actionable advice."
        )


def test_coaching_feedback_payload_raises_for_non_string_feedback() -> None:
    """Tests if the dataclass structurally rejects non-string data injected into the feedback field."""
    with pytest.raises(TypeError, match="feedback_text must be a string."):
        CoachingFeedbackPayload(
            segment_id=1,
            feedback_text={"summary": "Actionable advice."}  # type: ignore
        )


def test_coaching_feedback_payload_raises_for_boolean_feedback() -> None:
    """Tests if the dataclass strictly rejects booleans in the text field."""
    with pytest.raises(TypeError, match="feedback_text must be a string."):
        CoachingFeedbackPayload(
            segment_id=1,
            feedback_text=True  # type: ignore
        )


def test_coaching_feedback_payload_raises_for_empty_feedback() -> None:
    """Tests if the dataclass prohibits completely empty strings."""
    with pytest.raises(ValueError, match="feedback_text cannot be empty or whitespace."):
        CoachingFeedbackPayload(
            segment_id=1,
            feedback_text=""
        )


def test_coaching_feedback_payload_raises_for_whitespace_feedback() -> None:
    """Tests if the dataclass prohibits strings containing only spaces, tabs, and newlines."""
    with pytest.raises(ValueError, match="feedback_text cannot be empty or whitespace."):
        CoachingFeedbackPayload(
            segment_id=1,
            feedback_text="   \n \t  "
        )