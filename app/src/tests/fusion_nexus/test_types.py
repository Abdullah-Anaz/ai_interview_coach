import time

import pytest

from app.src.acoustic_edge.types import AcousticDescriptorPayload
from app.src.fusion_nexus.types import FusedPromptPayload, SegmentBufferState
from app.src.semantic_edge.types import SemanticDescriptorPayload
from app.src.visual_edge.facial_gaze.types import FacialDescriptorPayload
from app.src.visual_edge.skeletal_kinematics.types import SkeletalDescriptorPayload


@pytest.fixture
def real_semantic_payload() -> SemanticDescriptorPayload:
    """Provides a mathematically valid semantic payload."""
    return SemanticDescriptorPayload(
        segment_id=1,
        start_time_sec=0.0,
        end_time_sec=2.5,
        transcript_text="I managed the deployment."
    )


@pytest.fixture
def real_facial_payload() -> FacialDescriptorPayload:
    """Provides a mathematically valid facial gaze payload reflecting true OCEAN mappings."""
    return FacialDescriptorPayload(
        segment_id="1", 
        start_time_sec=0.0,
        end_time_sec=2.5,
        behavioral_descriptor="The candidate's facial micro-expressions indicate high extraversion, low neuroticism, high agreeableness, very high conscientiousness, and medium openness.",
        extraversion_score=0.8,
        neuroticism_score=0.2,
        agreeableness_score=0.7,
        conscientiousness_score=0.9,
        openness_score=0.6
    )


@pytest.fixture
def real_skeletal_payload() -> SkeletalDescriptorPayload:
    """Provides a mathematically valid skeletal kinematics payload reflecting BeMERC descriptors."""
    return SkeletalDescriptorPayload(
        segment_id=1,
        start_time_sec=0.0,
        end_time_sec=2.5,
        behavioral_descriptor="The candidate exhibited very low perceptual jitter and high temporal smoothness.",
        perceptual_jitter=0.05,
        temporal_smoothness=0.95
    )


@pytest.fixture
def real_acoustic_payload() -> AcousticDescriptorPayload:
    """Provides a mathematically valid acoustic payload reflecting true OCEAN mappings."""
    return AcousticDescriptorPayload(
        segment_id=1,
        start_time_sec=0.0,
        end_time_sec=2.5,
        vocal_cues=("The candidate's profile indicates high extraversion, low neuroticism, medium agreeableness, high conscientiousness, and medium openness.",),
        extraversion_score=0.8,
        neuroticism_score=0.2,
        agreeableness_score=0.5,
        conscientiousness_score=0.8,
        openness_score=0.5
    )


def test_segment_buffer_state_expected_behaviour_for_instantiation() -> None:
    """Verifies that the buffer state correctly initializes and sets the creation timestamp."""
    state = SegmentBufferState(segment_id=42)
    
    assert state.segment_id == 42
    assert isinstance(state.created_at, float)
    assert state.created_at <= time.time()
    assert state.semantic_payload is None
    assert state.facial_payload is None
    assert state.skeletal_payload is None
    assert state.acoustic_payload is None
    assert state.is_complete() is False


def test_segment_buffer_state_expected_behaviour_for_completion_check(
    real_semantic_payload: SemanticDescriptorPayload,
    real_facial_payload: FacialDescriptorPayload,
    real_skeletal_payload: SkeletalDescriptorPayload,
    real_acoustic_payload: AcousticDescriptorPayload
) -> None:
    """Verifies the boolean state trigger accurately switches when all modalities arrive."""
    state = SegmentBufferState(segment_id=1)
    
    state.semantic_payload = real_semantic_payload
    assert state.is_complete() is False
    
    state.facial_payload = real_facial_payload
    state.skeletal_payload = real_skeletal_payload
    assert state.is_complete() is False
    
    state.acoustic_payload = real_acoustic_payload
    assert state.is_complete() is True


def test_segment_buffer_state_raises_for_invalid_segment_id_type() -> None:
    """Ensures structural rejection if the sequence ID deviates from strict integer typing."""
    with pytest.raises(TypeError, match="segment_id must be an integer."):
        SegmentBufferState(segment_id="1")  # type: ignore


def test_segment_buffer_state_raises_for_negative_segment_id() -> None:
    """Ensures mathematical bounds block negative sequential identifiers."""
    with pytest.raises(ValueError, match="segment_id cannot be a negative integer."):
        SegmentBufferState(segment_id=-1)


def test_segment_buffer_state_raises_for_invalid_created_at_type() -> None:
    """Ensures strict timestamp safety to prevent TTL calculation crashes."""
    with pytest.raises(TypeError, match="created_at must be a float representing Unix epoch time."):
        SegmentBufferState(segment_id=1, created_at=1672531200)  # type: ignore (int instead of float)


def test_fused_prompt_payload_expected_behaviour_for_standard_instantiation() -> None:
    """Verifies that the immutable output payload correctly secures the final template string."""
    payload = FusedPromptPayload(
        segment_id=10,
        compiled_prompt="[SYSTEM] You are an expert. [CONTEXT] Hello."
    )
    
    assert payload.segment_id == 10
    assert payload.compiled_prompt == "[SYSTEM] You are an expert. [CONTEXT] Hello."


def test_fused_prompt_payload_raises_for_invalid_segment_id_type() -> None:
    """Ensures structural rejection for non-integer segment identifiers."""
    with pytest.raises(TypeError, match="segment_id must be an integer."):
        FusedPromptPayload(
            segment_id=10.5,  # type: ignore
            compiled_prompt="Test."
        )


def test_fused_prompt_payload_raises_for_negative_segment_id() -> None:
    """Ensures sequence identifiers are mathematically valid integers."""
    with pytest.raises(ValueError, match="segment_id cannot be a negative integer."):
        FusedPromptPayload(
            segment_id=-5,
            compiled_prompt="Test."
        )


def test_fused_prompt_payload_raises_for_invalid_prompt_type() -> None:
    """Ensures the text pipeline rejects structural deviations like raw dictionaries."""
    with pytest.raises(TypeError, match="compiled_prompt must be a strictly typed string."):
        FusedPromptPayload(
            segment_id=1,
            compiled_prompt={"prompt": "Test."}  # type: ignore
        )


def test_fused_prompt_payload_raises_for_empty_prompt() -> None:
    """Ensures the final LLM payload mathematically contains valid structural tokens."""
    with pytest.raises(ValueError, match="compiled_prompt cannot be an empty string."):
        FusedPromptPayload(
            segment_id=1,
            compiled_prompt="   "
        )