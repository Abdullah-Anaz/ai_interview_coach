import logging
import queue
import time
from typing import Any, Dict, List

from app.src.acoustic_edge.types import AcousticDescriptorPayload
from app.src.fusion_nexus.fusion_template import compile_instruct_erc_prompt
from app.src.fusion_nexus.types import FusedPromptPayload, SegmentBufferState
from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.semantic_edge.types import SemanticDescriptorPayload
from app.src.visual_edge.facial_gaze.types import FacialDescriptorPayload
from app.src.visual_edge.skeletal_kinematics.types import SkeletalDescriptorPayload

logger: logging.Logger = logging.getLogger(__name__)


def _compile_segment_state(state: SegmentBufferState) -> str:
    """
    Compiles a populated or partially degraded segment state into an InstructERC prompt string.

    Args:
        state (SegmentBufferState): The buffered state containing available edge DTOs.

    Returns:
        str: The compiled InstructERC prompt.

    Raises:
        RuntimeError: If template compilation fails.
    """
    try:
        return compile_instruct_erc_prompt(
            semantic=state.semantic_payload,
            acoustic=state.acoustic_payload,
            facial=state.facial_payload,
            skeletal=state.skeletal_payload
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to compile InstructERC prompt for segment {state.segment_id}: {exc}") from exc


def _emit_terminal_payload(
    compiled_prompts: List[str],
    output_q: queue.Queue,
    poll_timeout: float
) -> None:
    """
    Executes late fusion by merging chronologically compiled segments and emitting a single 
    FusedPromptPayload, followed immediately by a terminal sentinel.

    Args:
        compiled_prompts (List[str]): Sequence of compiled segment prompts.
        output_q (queue.Queue): Egress buffer for compiled FusedPromptPayload objects.
        poll_timeout (float): Block duration for queue insertion.

    Raises:
        RuntimeError: If the downstream queue is saturated.
    """
    try:
        if compiled_prompts:
            final_merged_prompt: str = "\n\n=== NEXT TEMPORAL SEGMENT ===\n\n".join(compiled_prompts)
            payload = FusedPromptPayload(
                segment_id=0,
                compiled_prompt=final_merged_prompt
            )
            output_q.put(payload, timeout=poll_timeout)
            logger.info("[NEXUS] Executed late fusion: Successfully emitted aggregated payload to Judge.")
            
        output_q.put(OrchestratorTerminalSentinel(), timeout=poll_timeout)
    except queue.Full as exc:
        raise RuntimeError("Nexus output queue saturated during terminal emission.") from exc


def run_fusion_nexus_node(
    semantic_q: queue.Queue,
    acoustic_q: queue.Queue,
    facial_q: queue.Queue,
    skeletal_q: queue.Queue,
    output_q: queue.Queue,
    ttl_seconds: float = 15.0,
    poll_timeout: float = 0.05
) -> None:
    """
    Executes the asynchronous aggregation matrix, buffering multimodal payloads.
    Enforces TTL degradation for fault tolerance and performs temporal late fusion.

    Args:
        semantic_q (queue.Queue): Ingress buffer for raw verbatim transcripts.
        acoustic_q (queue.Queue): Ingress buffer for vocal OCEAN descriptors.
        facial_q (queue.Queue): Ingress buffer for visual OCEAN descriptors.
        skeletal_q (queue.Queue): Ingress buffer for kinematic BeMERC descriptors.
        output_q (queue.Queue): Egress buffer for compiled FusedPromptPayload objects.
        ttl_seconds (float): Maximum buffer duration before a segment is forcibly compiled.
        poll_timeout (float): Micro-polling duration to ensure non-blocking rotation.

    Raises:
        TypeError: If queue interfaces violate structural constraints.
    """
    queues: Dict[str, queue.Queue] = {
        "semantic": semantic_q,
        "acoustic": acoustic_q,
        "facial": facial_q,
        "skeletal": skeletal_q
    }

    for name, q in queues.items():
        if not isinstance(q, queue.Queue):
            raise TypeError(f"{name} ingress must be a strictly typed queue.Queue.")
    if not isinstance(output_q, queue.Queue):
        raise TypeError("output_q egress must be a strictly typed queue.Queue.")

    buffer_matrix: Dict[int, SegmentBufferState] = {}
    compiled_prompts_buffer: List[str] = []
    active_sentinels: int = 0

    while True:
        current_time: float = time.time()

        expired_ids: List[int] = [
            sid for sid, state in buffer_matrix.items() 
            if (current_time - getattr(state, "created_at", current_time)) > ttl_seconds
        ]
        
        for sid in expired_ids:
            expired_state: SegmentBufferState = buffer_matrix.pop(sid)
            logger.warning("[NEXUS] Segment %d triggered TTL degradation. Proceeding with partial data.", sid)
            try:
                compiled_prompts_buffer.append(_compile_segment_state(expired_state))
            except Exception as exc:
                logger.error("Skipping expired segment %d: %s", sid, exc)

        if active_sentinels == 4:
            for sid in sorted(buffer_matrix.keys()):
                partial_state: SegmentBufferState = buffer_matrix[sid]
                try:
                    compiled_prompts_buffer.append(_compile_segment_state(partial_state))
                except Exception as exc:
                    logger.error("Skipping partial segment %d during flush: %s", sid, exc)
            
            buffer_matrix.clear()
            _emit_terminal_payload(compiled_prompts_buffer, output_q, poll_timeout)
            
            compiled_prompts_buffer.clear()
            active_sentinels = 0
            continue

        for source, q in queues.items():
            try:
                item: Any = q.get(timeout=poll_timeout)
            except queue.Empty:
                continue

            if isinstance(item, OrchestratorTerminalSentinel):
                active_sentinels += 1
                continue

            segment_id: int = getattr(item, "segment_id", -1)
            
            if isinstance(segment_id, str):
                try:
                    segment_id = int(segment_id)
                except ValueError:
                    continue
            
            if segment_id < 0:
                continue

            if segment_id not in buffer_matrix:
                buffer_matrix[segment_id] = SegmentBufferState(segment_id=segment_id)
            
            target_state: SegmentBufferState = buffer_matrix[segment_id]

            if source == "semantic" and isinstance(item, SemanticDescriptorPayload):
                target_state.semantic_payload = item
            elif source == "acoustic" and isinstance(item, AcousticDescriptorPayload):
                target_state.acoustic_payload = item
            elif source == "facial" and isinstance(item, FacialDescriptorPayload):
                target_state.facial_payload = item
            elif source == "skeletal" and isinstance(item, SkeletalDescriptorPayload):
                target_state.skeletal_payload = item

            if target_state.is_complete():
                completed_state: SegmentBufferState = buffer_matrix.pop(segment_id)
                try:
                    compiled_prompts_buffer.append(_compile_segment_state(completed_state))
                except Exception as exc:
                    logger.error("Skipping segment %d: %s", segment_id, exc)