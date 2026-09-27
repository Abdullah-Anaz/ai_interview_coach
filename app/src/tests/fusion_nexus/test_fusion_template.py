import pytest
import torch

from app.src.acoustic_edge.types import AcousticDescriptorPayload
from app.src.fusion_nexus.fusion_template import (
    _format_acoustic_knowledge,
    _format_facial_knowledge,
    _format_semantic_context,
    _format_skeletal_knowledge,
    compile_instruct_erc_prompt,
)
from app.src.semantic_edge.types import SemanticDescriptorPayload
from app.src.visual_edge.facial_gaze.types import FacialDescriptorPayload
from app.src.visual_edge.skeletal_kinematics.types import SkeletalDescriptorPayload


@pytest.fixture
def hardware_check() -> None:
    """Validates runtime environments but enforces CPU operations for string manipulation."""
    _ = torch.cuda.is_available()


@pytest.fixture
def full_semantic_payload() -> SemanticDescriptorPayload:
    return SemanticDescriptorPayload(
        segment_id=1,
        start_time_sec=0.0,
        end_time_sec=2.5,
        transcript_text="I successfully deployed the database migration."
    )


@pytest.fixture
def full_acoustic_payload() -> AcousticDescriptorPayload:
    return AcousticDescriptorPayload(
        segment_id=1,
        start_time_sec=0.0,
        end_time_sec=2.5,
        vocal_cues=("The candidate's profile indicates high extraversion and low neuroticism.",),
        extraversion_score=0.8,
        neuroticism_score=0.2,
        agreeableness_score=0.5,
        conscientiousness_score=0.9,
        openness_score=0.6
    )


@pytest.fixture
def full_facial_payload() -> FacialDescriptorPayload:
    return FacialDescriptorPayload(
        segment_id="1",
        start_time_sec=0.0,
        end_time_sec=2.5,
        behavioral_descriptor="Maintained strong eye contact with positive valence.",
        extraversion_score=0.8,
        neuroticism_score=0.2,
        agreeableness_score=0.7,
        conscientiousness_score=0.9,
        openness_score=0.6
    )


@pytest.fixture
def full_skeletal_payload() -> SkeletalDescriptorPayload:
    return SkeletalDescriptorPayload(
        segment_id=1,
        start_time_sec=0.0,
        end_time_sec=2.5,
        behavioral_descriptor="Exhibited very low perceptual jitter.",
        perceptual_jitter=0.05,
        temporal_smoothness=0.95
    )


# =====================================================================
# HELPER FUNCTION TESTS: EXPECTED BEHAVIOUR & EDGE CASES
# =====================================================================

def test_format_semantic_context_expected_behaviour_for_whitespace() -> None:
    """Tests if a payload with only spaces safely collapses to UNAVAILABLE."""
    payload = SemanticDescriptorPayload.__new__(SemanticDescriptorPayload)
    object.__setattr__(payload, "transcript_text", "   \n  ")
    assert _format_semantic_context(payload) == "UNAVAILABLE"


def test_format_acoustic_knowledge_expected_behaviour_for_empty_tuple() -> None:
    """Tests if an acoustic payload with an empty cues tuple gracefully degrades."""
    payload = AcousticDescriptorPayload.__new__(AcousticDescriptorPayload)
    object.__setattr__(payload, "vocal_cues", ())
    assert _format_acoustic_knowledge(payload) == "UNAVAILABLE"


def test_format_facial_knowledge_expected_behaviour_for_whitespace() -> None:
    """Tests if a facial payload with empty descriptors gracefully degrades."""
    payload = FacialDescriptorPayload.__new__(FacialDescriptorPayload)
    object.__setattr__(payload, "behavioral_descriptor", "   ")
    assert _format_facial_knowledge(payload) == "UNAVAILABLE"


def test_format_skeletal_knowledge_expected_behaviour_for_whitespace() -> None:
    """Tests if a skeletal payload with empty descriptors gracefully degrades, bypassing DTO validation to test compiler fallback."""
    payload = SkeletalDescriptorPayload.__new__(SkeletalDescriptorPayload)
    object.__setattr__(payload, "behavioral_descriptor", "   ")
    assert _format_skeletal_knowledge(payload) == "UNAVAILABLE"


# =====================================================================
# HELPER FUNCTION TESTS: ERROR ISOLATION (TYPE CHECKING)
# =====================================================================

def test_format_semantic_context_raises_for_invalid_type() -> None:
    """Proves the helper directly traps type violations."""
    with pytest.raises(TypeError, match="semantic must be an instance"):
        _format_semantic_context({"transcript_text": "Spoofed"})  # type: ignore


def test_format_acoustic_knowledge_raises_for_invalid_type() -> None:
    """Proves the helper directly traps type violations."""
    with pytest.raises(TypeError, match="acoustic must be an instance"):
        _format_acoustic_knowledge({"vocal_cues": ("Spoofed",)})  # type: ignore


def test_format_facial_knowledge_raises_for_invalid_type() -> None:
    """Proves the helper directly traps type violations."""
    with pytest.raises(TypeError, match="facial must be an instance"):
        _format_facial_knowledge({"behavioral_descriptor": "Spoofed"})  # type: ignore


def test_format_skeletal_knowledge_raises_for_invalid_type() -> None:
    """Proves the helper directly traps type violations."""
    with pytest.raises(TypeError, match="skeletal must be an instance"):
        _format_skeletal_knowledge({"behavioral_descriptor": "Spoofed"})  # type: ignore


# =====================================================================
# COMPILER TESTS: EXPECTED BEHAVIOUR & DEGRADATION PATHWAYS
# =====================================================================

def test_compile_instruct_erc_prompt_expected_behaviour_for_all_modalities(
    hardware_check: None,
    full_semantic_payload: SemanticDescriptorPayload,
    full_acoustic_payload: AcousticDescriptorPayload,
    full_facial_payload: FacialDescriptorPayload,
    full_skeletal_payload: SkeletalDescriptorPayload
) -> None:
    """Verifies flawless compilation when the asynchronous buffer collects all parallel payloads."""
    prompt: str = compile_instruct_erc_prompt(
        semantic=full_semantic_payload,
        acoustic=full_acoustic_payload,
        facial=full_facial_payload,
        skeletal=full_skeletal_payload
    )

    assert "<transcript>\nI successfully deployed the database migration.\n</transcript>" in prompt
    assert "<vocal_delivery>\nThe candidate's profile indicates high extraversion" in prompt
    assert "<facial_microexpressions>\nMaintained strong eye contact" in prompt
    assert "<posture_kinematics>\nExhibited very low perceptual jitter." in prompt
    assert "<style_guidelines>" in prompt
    assert "### 🎯 Executive Summary" in prompt


def test_compile_instruct_erc_prompt_expected_behaviour_for_only_semantic(
    full_semantic_payload: SemanticDescriptorPayload
) -> None:
    """Tests degradation when camera and microphone hardware fail entirely, leaving only transcript."""
    prompt: str = compile_instruct_erc_prompt(
        semantic=full_semantic_payload,
        acoustic=None,
        facial=None,
        skeletal=None
    )

    assert "<transcript>\nI successfully deployed the database migration.\n</transcript>" in prompt
    assert "<vocal_delivery>\nUNAVAILABLE\n</vocal_delivery>" in prompt
    assert "<facial_microexpressions>\nUNAVAILABLE\n</facial_microexpressions>" in prompt
    assert "<posture_kinematics>\nUNAVAILABLE\n</posture_kinematics>" in prompt


def test_compile_instruct_erc_prompt_expected_behaviour_for_missing_semantic(
    full_acoustic_payload: AcousticDescriptorPayload,
    full_facial_payload: FacialDescriptorPayload,
    full_skeletal_payload: SkeletalDescriptorPayload
) -> None:
    """Tests degradation when speech-to-text fails but multimodal telemetry succeeds."""
    prompt: str = compile_instruct_erc_prompt(
        semantic=None,
        acoustic=full_acoustic_payload,
        facial=full_facial_payload,
        skeletal=full_skeletal_payload
    )

    assert "<transcript>\nUNAVAILABLE\n</transcript>" in prompt
    assert "<vocal_delivery>\nThe candidate's profile" in prompt
    assert "<facial_microexpressions>\nMaintained strong eye contact" in prompt
    assert "<posture_kinematics>\nExhibited very low perceptual jitter." in prompt


def test_compile_instruct_erc_prompt_expected_behaviour_for_total_degradation() -> None:
    """Tests if the compiler safely falls back on all tags if the orchestrator fails catastrophically."""
    prompt: str = compile_instruct_erc_prompt(
        semantic=None,
        acoustic=None,
        facial=None,
        skeletal=None
    )

    assert "<transcript>\nUNAVAILABLE\n</transcript>" in prompt
    assert "<vocal_delivery>\nUNAVAILABLE\n</vocal_delivery>" in prompt
    assert "<facial_microexpressions>\nUNAVAILABLE\n</facial_microexpressions>" in prompt
    assert "<posture_kinematics>\nUNAVAILABLE\n</posture_kinematics>" in prompt
    assert "### 🛠️ High-ROI Coaching Adjustments" in prompt


# =====================================================================
# COMPILER TESTS: ERROR CASCADES (TYPE POLLUTION)
# =====================================================================

def test_compile_instruct_erc_prompt_raises_for_malformed_semantic() -> None:
    """Tests graceful error trapping if an invalid semantic payload bypasses queue constraints."""
    with pytest.raises(RuntimeError, match="InstructERC prompt compilation failed"):
        compile_instruct_erc_prompt(
            semantic={"transcript_text": "Spoofed data"},  # type: ignore
            acoustic=None,
            facial=None,
            skeletal=None
        )


def test_compile_instruct_erc_prompt_raises_for_malformed_acoustic(
    full_semantic_payload: SemanticDescriptorPayload
) -> None:
    """Tests graceful error trapping if an invalid acoustic payload bypasses queue constraints."""
    with pytest.raises(RuntimeError, match="InstructERC prompt compilation failed"):
        compile_instruct_erc_prompt(
            semantic=full_semantic_payload,
            acoustic={"vocal_cues": ("Spoofed",)},  # type: ignore
            facial=None,
            skeletal=None
        )


def test_compile_instruct_erc_prompt_raises_for_malformed_facial(
    full_semantic_payload: SemanticDescriptorPayload
) -> None:
    """Tests graceful error trapping if an invalid facial payload bypasses queue constraints."""
    with pytest.raises(RuntimeError, match="InstructERC prompt compilation failed"):
        compile_instruct_erc_prompt(
            semantic=full_semantic_payload,
            acoustic=None,
            facial={"behavioral_descriptor": "Spoofed"},  # type: ignore
            skeletal=None
        )


def test_compile_instruct_erc_prompt_raises_for_malformed_skeletal(
    full_semantic_payload: SemanticDescriptorPayload
) -> None:
    """Tests graceful error trapping if an invalid skeletal payload bypasses queue constraints."""
    with pytest.raises(RuntimeError, match="InstructERC prompt compilation failed"):
        compile_instruct_erc_prompt(
            semantic=full_semantic_payload,
            acoustic=None,
            facial=None,
            skeletal={"behavioral_descriptor": "Spoofed"}  # type: ignore
        )