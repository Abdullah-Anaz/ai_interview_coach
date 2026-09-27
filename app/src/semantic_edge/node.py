import logging
import queue
from typing import Any, List, Tuple

from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.orchestration.types import AlignedWord, SynchronizedSegment
from app.src.semantic_edge.types import SemanticDescriptorPayload

logger: logging.Logger = logging.getLogger(__name__)


def _extract_transcript(words: Tuple[AlignedWord, ...]) -> str:
    """
    Concatenates a temporally aligned sequence of words into a single, whitespace-normalized string.

    Args:
        words (Tuple[AlignedWord, ...]): The sequence of strict text alignment dataclases.

    Returns:
        str: The fully reconstructed, verbatim spoken utterance.

    Raises:
        TypeError: If the input is not a tuple or contains non-AlignedWord objects.
    """
    if not isinstance(words, tuple):
        raise TypeError("Words must be provided as a strictly typed tuple.")
    
    extracted: List[str] = []
    for word_obj in words:
        if not isinstance(word_obj, AlignedWord):
            raise TypeError("All items in the tuple must be AlignedWord instances.")
            
        clean_word: str = word_obj.word.strip()
        if clean_word:
            extracted.append(clean_word)
            
    return " ".join(extracted)


def _process_segment(item: SynchronizedSegment) -> SemanticDescriptorPayload:
    """
    Extracts semantic text and boundaries from a temporal window to generate a transport payload.

    Args:
        item (SynchronizedSegment): The tightly-coupled multimodal payload from the Temporal Aligner.

    Returns:
        SemanticDescriptorPayload: The immutable data transfer object mapped to the specific window.

    Raises:
        TypeError: If the input object structurally deviates from a SynchronizedSegment.
        RuntimeError: If data extraction or payload instantiation fails mathematically.
    """
    if not isinstance(item, SynchronizedSegment):
        raise TypeError("Input must be a strictly typed SynchronizedSegment.")

    try:
        transcript: str = _extract_transcript(item.words)
        
        return SemanticDescriptorPayload(
            segment_id=int(item.segment_id),
            start_time_sec=float(item.audio_slice_start_sec),
            end_time_sec=float(item.audio_slice_end_sec),
            transcript_text=transcript
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to process semantic segment {getattr(item, 'segment_id', 'unknown')}: {exc}") from exc


def _aggregate_payloads(payloads: List[SemanticDescriptorPayload]) -> SemanticDescriptorPayload:
    """
    Reduces a sequence of semantic segment payloads into a singular concatenated text matrix.

    Args:
        payloads (List[SemanticDescriptorPayload]): Accumulated text inference payloads.

    Returns:
        SemanticDescriptorPayload: Unified terminal payload containing the complete transcript.

    Raises:
        ValueError: If the payload list is explicitly empty.
    """
    if not payloads:
        raise ValueError("Aggregation requires a strictly populated sequence of payloads.")

    full_transcript: str = " ".join(
        p.transcript_text.strip() 
        for p in payloads 
        if p.transcript_text and p.transcript_text.strip()
    )

    return SemanticDescriptorPayload(
        segment_id=0,
        start_time_sec=payloads[0].start_time_sec,
        end_time_sec=payloads[-1].end_time_sec,
        transcript_text=full_transcript
    )


def run_semantic_node(
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    timeout_sec: float = 1.0
) -> None:
    """
    Executes the isolated multiprocessing loop for the Semantic Edge text extraction pipeline 
    using a Stream-Reduce aggregation pattern.

    Args:
        input_queue (queue.Queue): Upstream buffer supplying temporally aligned segments.
        output_queue (queue.Queue): Downstream buffer routing payloads to the Late Fusion Nexus.
        timeout_sec (float): Polling duration for the queue lock. Defaults to 1.0.

    Raises:
        TypeError: If the provided communication buffers are not valid Queue instances.
        RuntimeError: If the downstream queue saturates or segment processing fatally collapses.
    """
    if not isinstance(input_queue, queue.Queue) or not isinstance(output_queue, queue.Queue):
        raise TypeError("Communication buffers must be instances of queue.Queue.")

    history_buffer: List[SemanticDescriptorPayload] = []

    while True:
        try:
            item: Any = input_queue.get(timeout=timeout_sec)
        except queue.Empty:
            continue

        if isinstance(item, OrchestratorTerminalSentinel):
            if history_buffer:
                try:
                    final_payload: SemanticDescriptorPayload = _aggregate_payloads(history_buffer)
                    object.__setattr__(final_payload, "segment_id", 0)
                    output_queue.put(final_payload, timeout=timeout_sec)
                    logger.info("[SEMANTIC SUCCESS] Emitted global aggregated payload to Fusion Nexus.")
                except queue.Full as exc:
                    raise RuntimeError(f"Semantic aggregation failure: queue saturated. {exc}") from exc
                except Exception as exc:
                    try:
                        output_queue.put(OrchestratorTerminalSentinel(error=exc), timeout=timeout_sec)
                    except queue.Full:
                        pass  # Ignore secondary saturation crash
                    raise RuntimeError(f"Semantic aggregation failure: {exc}") from exc

            try:
                output_queue.put(item, timeout=timeout_sec)
                history_buffer.clear()
                continue
            except queue.Full as exc:
                raise RuntimeError("Output queue saturated during termination cascade.") from exc

        if not isinstance(item, SynchronizedSegment):
            continue

        try:
            payload: SemanticDescriptorPayload = _process_segment(item)
            history_buffer.append(payload)
            logger.info("[SEMANTIC STREAM] Processed segment %d into aggregation buffer.", payload.segment_id)
        except Exception as exc:
            try:
                output_queue.put(OrchestratorTerminalSentinel(error=exc), timeout=timeout_sec)
            except queue.Full:
                pass
            logger.error("Semantic processing failure on segment %s: %s", getattr(item, 'segment_id', 'unknown'), exc)
            raise RuntimeError(f"Semantic pipeline execution failed: {exc}") from exc