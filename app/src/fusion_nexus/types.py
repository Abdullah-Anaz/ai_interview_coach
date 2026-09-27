import time
from dataclasses import dataclass, field
from typing import Optional

from app.src.acoustic_edge.types import AcousticDescriptorPayload
from app.src.semantic_edge.types import SemanticDescriptorPayload
from app.src.visual_edge.facial_gaze.types import FacialDescriptorPayload
from app.src.visual_edge.skeletal_kinematics.types import SkeletalDescriptorPayload


@dataclass
class SegmentBufferState:
    """
    Mutable state tracker for a temporal segment during asynchronous multimodal aggregation.
    Monitors the arrival of edge payloads and manages Time-to-Live (TTL) synchronization.
    """
    segment_id: int
    created_at: float = field(default_factory=time.time)
    semantic_payload: Optional[SemanticDescriptorPayload] = None
    facial_payload: Optional[FacialDescriptorPayload] = None
    skeletal_payload: Optional[SkeletalDescriptorPayload] = None
    acoustic_payload: Optional[AcousticDescriptorPayload] = None

    def __post_init__(self) -> None:
        """
        Executes strict runtime validation to ensure structural integrity of the buffer state.

        Raises:
            TypeError: If attribute types deviate from the defined schema.
            ValueError: If the sequence identifier is mathematically invalid.
        """
        if not isinstance(self.segment_id, int):
            raise TypeError("segment_id must be an integer.")
        if self.segment_id < 0:
            raise ValueError("segment_id cannot be a negative integer.")
        if not isinstance(self.created_at, float):
            raise TypeError("created_at must be a float representing Unix epoch time.")

    def is_complete(self) -> bool:
        """
        Evaluates whether all required parallel modalities have successfully populated the buffer.

        Returns:
            bool: True if all modality payloads are present, False otherwise.
        """
        return all([
            self.semantic_payload is not None,
            self.facial_payload is not None,
            self.skeletal_payload is not None,
            self.acoustic_payload is not None
        ])


@dataclass(frozen=True)
class FusedPromptPayload:
    """
    An immutable transport structure containing the final compiled InstructERC 
    prompt string ready for language model synthesis.
    """
    segment_id: int
    compiled_prompt: str

    def __post_init__(self) -> None:
        """
        Executes strict runtime validation to ensure structural integrity of the LLM prompt.

        Raises:
            TypeError: If attribute types deviate from the defined schema.
            ValueError: If the prompt is structurally empty or the sequence identifier is invalid.
        """
        if not isinstance(self.segment_id, int):
            raise TypeError("segment_id must be an integer.")
        if self.segment_id < 0:
            raise ValueError("segment_id cannot be a negative integer.")

        if not isinstance(self.compiled_prompt, str):
            raise TypeError("compiled_prompt must be a strictly typed string.")
        if not self.compiled_prompt.strip():
            raise ValueError("compiled_prompt cannot be an empty string.")