from typing import Optional

from app.src.acoustic_edge.types import AcousticDescriptorPayload
from app.src.semantic_edge.types import SemanticDescriptorPayload
from app.src.visual_edge.facial_gaze.types import FacialDescriptorPayload
from app.src.visual_edge.skeletal_kinematics.types import SkeletalDescriptorPayload


def _format_semantic_context(semantic: Optional[SemanticDescriptorPayload]) -> str:
    """Extracts the verbatim transcript, falls back gracefully, or raises on type violations."""
    if semantic is None:
        return "UNAVAILABLE"
    if not isinstance(semantic, SemanticDescriptorPayload):
        raise TypeError("semantic must be an instance of SemanticDescriptorPayload.")
        
    if semantic.transcript_text.strip():
        return semantic.transcript_text.strip()
    return "UNAVAILABLE"


def _format_acoustic_knowledge(acoustic: Optional[AcousticDescriptorPayload]) -> str:
    """Extracts the vocal delivery cues, falls back gracefully, or raises on type violations."""
    if acoustic is None:
        return "UNAVAILABLE"
    if not isinstance(acoustic, AcousticDescriptorPayload):
        raise TypeError("acoustic must be an instance of AcousticDescriptorPayload.")
        
    if acoustic.vocal_cues and acoustic.vocal_cues[0].strip():
        return acoustic.vocal_cues[0].strip()
    return "UNAVAILABLE"


def _format_facial_knowledge(facial: Optional[FacialDescriptorPayload]) -> str:
    """Extracts the facial behavioral descriptors, falls back gracefully, or raises on type violations."""
    if facial is None:
        return "UNAVAILABLE"
    if not isinstance(facial, FacialDescriptorPayload):
        raise TypeError("facial must be an instance of FacialDescriptorPayload.")
        
    if facial.behavioral_descriptor.strip():
        return facial.behavioral_descriptor.strip()
    return "UNAVAILABLE"


def _format_skeletal_knowledge(skeletal: Optional[SkeletalDescriptorPayload]) -> str:
    """Extracts the kinematic posture descriptors, falls back gracefully, or raises on type violations."""
    if skeletal is None:
        return "UNAVAILABLE"
    if not isinstance(skeletal, SkeletalDescriptorPayload):
        raise TypeError("skeletal must be an instance of SkeletalDescriptorPayload.")
        
    if skeletal.behavioral_descriptor.strip():
        return skeletal.behavioral_descriptor.strip()
    return "UNAVAILABLE"


def compile_instruct_erc_prompt(
    semantic: Optional[SemanticDescriptorPayload],
    acoustic: Optional[AcousticDescriptorPayload],
    facial: Optional[FacialDescriptorPayload],
    skeletal: Optional[SkeletalDescriptorPayload]
) -> str:
    """
    Constructs a highly engineered InstructERC prompt tailored for the 'Tell me about yourself' 
    interview question, enforcing multimodal congruence analysis and structured output schemas.

    Args:
        semantic (Optional[SemanticDescriptorPayload]): The verbatim text transcript.
        acoustic (Optional[AcousticDescriptorPayload]): The OCEAN vocal delivery descriptors.
        facial (Optional[FacialDescriptorPayload]): The OCEAN facial micro-expression descriptors.
        skeletal (Optional[SkeletalDescriptorPayload]): The kinematic BeMERC descriptors.

    Returns:
        str: The fully compiled prompt formatted for the Apex LLM Judge.

    Raises:
        RuntimeError: If the compilation process fails structurally.
    """
    try:
        transcript: str = _format_semantic_context(semantic)
        vocal_cues: str = _format_acoustic_knowledge(acoustic)
        facial_cues: str = _format_facial_knowledge(facial)
        posture_cues: str = _format_skeletal_knowledge(skeletal)

        prompt: str = (
            "<system_persona>\n"
            "You are an elite Executive Interview Coach and Senior HR Manager. "
            "Your expertise combines behavioral psychology, executive presence, and empathetic leadership development. "
            "You deliver radically candid, grounded, and undeniably authoritative feedback designed to elevate candidates to their highest potential. You speak naturally, like a human mentor, never like a robot analyzing data.\n"
            "</system_persona>\n\n"
            
            "<telemetry_context>\n"
            "- <vocal_delivery> and <facial_microexpressions> are scored against personality traits.\n"
            "- <posture_kinematics> measures physical stability.\n"
            "CRITICAL: You are receiving backend data, but your output MUST sound like a human conversation. NEVER use terms like 'telemetry', 'kinematics', 'microexpressions', 'multimodal', 'erratic', 'sway', or 'OCEAN'. Instead of 'posture kinematics reveal swaying,' say 'I noticed you shifting in your seat.' Instead of 'facial microexpressions,' say 'your facial expressions' or 'your smile'.\n"
            "</telemetry_context>\n\n"
            
            "<style_guidelines>\n"
            "1. NATURAL, HUMAN CONVERSATION: Write exactly as a high-level executive coach would speak to a client in a one-on-one session. Avoid stiff, robotic, or overly academic phrasing completely.\n"
            "2. CONSTRUCTIVE FEEDFORWARD (HOW TO IMPROVE): Never point out a flaw without immediately providing a practical, natural way to fix it. Coach them step-by-step on exactly what to do differently next time.\n"
            "3. ZERO-HALLUCINATION REALITY CHECK: You MUST NOT invent emotions, energy, or passion that do not exist in the telemetry. If the telemetry indicates a flat, calm, neutral, or low-energy delivery, you must state that truthfully and constructively.\n"
            "4. VERBATIM EXTRACTION ONLY: You are strictly forbidden from paraphrasing, altering, or inventing quotes. Every single word inside quotation marks MUST be copied character-for-character, word-for-word directly out of the <transcript> block above. If you cannot find an exact match, do not use quotes.\n"
            "5. ROLE AGNOSTIC: Do not assume or reference any specific job title, seniority level, or target company.\n"
            "6. ACCESSIBILITY: Explain everything in remarkably simple, actionable, and conversational language.\n"
            "</style_guidelines>\n\n"

            "<primary_objective>\n"
            "Evaluate the candidate's 'Tell me about yourself' elevator pitch. Assess their Present-Past-Future narrative structure and the congruence between their spoken words and non-verbal delivery. Show them exactly how to improve.\n"
            "</primary_objective>\n\n"
            
            "<candidate_telemetry>\n"
            f"<transcript>\n{transcript}\n</transcript>\n"
            f"<vocal_delivery>\n{vocal_cues}\n</vocal_delivery>\n"
            f"<facial_microexpressions>\n{facial_cues}\n</facial_microexpressions>\n"
            f"<posture_kinematics>\n{posture_cues}\n</posture_kinematics>\n"
            "</candidate_telemetry>\n\n"
            
            "<output_schema>\n"
            "Output EXACTLY according to this Markdown schema. No introductory or concluding filler.\n\n"
            "### 🎯 Executive Summary\n"
            "[A single, highly specific paragraph summarizing their overall presence in a natural, conversational tone. Speak directly to them (e.g., 'You walked into this interview...'). Base this STRICTLY on their actual demeanor, whether it was highly energetic or very subdued. End by telling them the #1 thing they need to focus on to improve.]\n\n"
            "### 🏗️ Narrative Structure (Present-Past-Future)\n"
            "**Present:** [Critique their opening naturally. You MUST copy and paste an exact, verbatim phrase from the <transcript> inside quotation marks to prove your point. Tell them exactly how to make this section stronger.]\n"
            "**Past:** [Critique their highlights. You MUST copy and paste an exact, verbatim phrase or achievement from the <transcript> inside quotation marks. Explain precisely how they can better connect that specific past achievement to their value.]\n"
            "**Future:** [Critique their close. You MUST copy and paste their exact final words from the <transcript> inside quotation marks. Give them the exact strategy to end on a more confident, forward-looking note.]\n\n"  
            "### 🎭 Delivery and Presence\n"
            "[Speak naturally about how their voice, face, and body language impacted their message. Talk to them like a human mentor. DO NOT use technical words. If they shifted in their seat, tell them how to ground themselves. If their voice was flat, tell them exactly where and how to inject energy.]\n\n"
            "### 🛠️ High-ROI Coaching Adjustments\n"
            "**Semantic Rewrite:** [Provide an exact, improved script revision for their weakest phrasing so they know what excellence looks like.]\n\n"
            "**Delivery Correction:** [Provide one precise, conversational physical or vocal command they can practice immediately to project calm authority.]\n"
            "</output_schema>"
        )
        return prompt

    except Exception as exc:
        raise RuntimeError(f"InstructERC prompt compilation failed: {exc}") from exc