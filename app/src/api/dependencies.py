import threading
from dataclasses import dataclass, field
from queue import Queue
from typing import List


@dataclass(frozen=True)
class PipelineState:
    """
    Centralized memory registry for asynchronous communication buffers and thread handles.

    Attributes:
        q_transcription (Queue): Ingress buffer for raw audio chunks.
        q_window (Queue): Ingress buffer for synchronized temporal windows.
        q_acoustic (Queue): Egress buffer routing to the acoustic edge.
        q_semantic (Queue): Egress buffer routing to the semantic edge.
        q_visual_facial (Queue): Egress buffer routing to the facial kinematics edge.
        q_visual_skeletal (Queue): Egress buffer routing to the skeletal kinematics edge.
        q_nexus_out (Queue): Egress buffer routing fused payloads to the LLM judge.
        q_judge_out (Queue): Final egress buffer holding the coaching feedback.
        active_threads (List[threading.Thread]): Registry of currently executing background loops.
    """
    q_transcription: Queue = field(default_factory=Queue)
    q_window: Queue = field(default_factory=Queue)
    q_acoustic: Queue = field(default_factory=Queue)
    q_semantic: Queue = field(default_factory=Queue)
    q_visual_facial: Queue = field(default_factory=Queue)
    q_visual_skeletal: Queue = field(default_factory=Queue)
    q_nexus_out: Queue = field(default_factory=Queue)
    q_judge_out: Queue = field(default_factory=Queue)
    active_threads: List[threading.Thread] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Validates structural types of the queue and thread matrices.

        Raises:
            TypeError: If any attribute is allocated with an invalid type or signature.
        """
        queue_fields: List[str] = [
            "q_transcription", "q_window", "q_acoustic", "q_semantic",
            "q_visual_facial", "q_visual_skeletal", "q_nexus_out", "q_judge_out"
        ]
        
        for q_name in queue_fields:
            val = getattr(self, q_name)
            if not isinstance(val, Queue):
                raise TypeError(f"{q_name} must be a queue.Queue instance.")
                
        if not isinstance(self.active_threads, list):
            raise TypeError("active_threads must be a list instance.")
            
        for thread_obj in self.active_threads:
            if not isinstance(thread_obj, threading.Thread):
                raise TypeError("Elements of active_threads must be threading.Thread instances.")


# Global state instance bridging the ASGI application to the background edge matrix
global_pipeline_state = PipelineState()