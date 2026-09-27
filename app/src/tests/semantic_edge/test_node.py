import threading
from queue import Queue
from typing import Any, List, Tuple

import pytest
import torch

from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.orchestration.types import AlignedWord, SynchronizedSegment
from app.src.semantic_edge.node import (
    _aggregate_payloads,
    _extract_transcript,
    _process_segment,
    run_semantic_node,
)
from app.src.semantic_edge.types import SemanticDescriptorPayload


@pytest.fixture
def hardware_check() -> None:
    """Validates runtime environments but enforces CPU operations for string manipulation."""
    _ = torch.cuda.is_available()


@pytest.fixture
def valid_words_tuple() -> Tuple[AlignedWord, ...]:
    """Generates a mathematically valid sequence of acoustic-aligned text tokens."""
    return (
        AlignedWord(word="I", start_time_sec=0.0, end_time_sec=0.2, start_frame_idx=0, end_frame_idx=10, confidence=0.99),
        AlignedWord(word=" deployed", start_time_sec=0.2, end_time_sec=0.6, start_frame_idx=10, end_frame_idx=30, confidence=0.95),
        AlignedWord(word="the ", start_time_sec=0.6, end_time_sec=0.8, start_frame_idx=30, end_frame_idx=40, confidence=0.98),
        AlignedWord(word="pipeline.", start_time_sec=0.8, end_time_sec=1.5, start_frame_idx=40, end_frame_idx=75, confidence=0.99)
    )


@pytest.fixture
def whitespace_words_tuple() -> Tuple[AlignedWord, ...]:
    """Generates an edge-case sequence containing erratic spacing to test normalization."""
    return (
        AlignedWord(word="   Yes,  ", start_time_sec=0.0, end_time_sec=0.5, start_frame_idx=0, end_frame_idx=25, confidence=0.90),
        AlignedWord(word="   ", start_time_sec=0.5, end_time_sec=1.0, start_frame_idx=25, end_frame_idx=50, confidence=0.0),
        AlignedWord(word=" absolutely. ", start_time_sec=1.0, end_time_sec=2.0, start_frame_idx=50, end_frame_idx=100, confidence=0.99)
    )


@pytest.fixture
def standard_segment(valid_words_tuple: Tuple[AlignedWord, ...]) -> SynchronizedSegment:
    """Constructs a fully populated, temporally aligned window from the orchestrator."""
    return SynchronizedSegment(
        segment_id=77,
        words=valid_words_tuple,
        audio_slice_start_sec=0.0,
        audio_slice_end_sec=1.5,
        visual_frame_start_idx=0,
        visual_frame_end_idx=75,
        is_final=False
    )


@pytest.fixture
def secondary_segment(whitespace_words_tuple: Tuple[AlignedWord, ...]) -> SynchronizedSegment:
    """Constructs a secondary temporal window to evaluate sequence aggregation."""
    return SynchronizedSegment(
        segment_id=78,
        words=whitespace_words_tuple,
        audio_slice_start_sec=1.5,
        audio_slice_end_sec=3.5,
        visual_frame_start_idx=75,
        visual_frame_end_idx=175,
        is_final=False
    )


def test_extract_transcript_expected_behaviour_for_valid_tuple(valid_words_tuple: Tuple[AlignedWord, ...]) -> None:
    """Verifies that the extraction algorithm strictly joins and trims a sequence of tokens."""
    result: str = _extract_transcript(valid_words_tuple)
    assert result == "I deployed the pipeline."


def test_extract_transcript_expected_behaviour_for_whitespace_tuple(whitespace_words_tuple: Tuple[AlignedWord, ...]) -> None:
    """Verifies that the mathematical extraction cleanly normalizes extreme space anomalies."""
    result: str = _extract_transcript(whitespace_words_tuple)
    assert result == "Yes, absolutely."


def test_extract_transcript_expected_behaviour_for_empty_tuple() -> None:
    """Verifies that silent periods with zero acoustic words return empty structural strings."""
    result: str = _extract_transcript(tuple())
    assert result == ""


def test_extract_transcript_raises_for_invalid_type() -> None:
    """Ensures architectural constraints successfully block standard Python lists."""
    with pytest.raises(TypeError, match="Words must be provided as a strictly typed tuple."):
        _extract_transcript(["not", "a", "tuple"])  # type: ignore


def test_extract_transcript_raises_for_invalid_elements() -> None:
    """Ensures extraction aborts if the temporal tuple is polluted with raw strings."""
    invalid_tuple = (
        AlignedWord(word="Test", start_time_sec=0.0, end_time_sec=0.5, start_frame_idx=0, end_frame_idx=25, confidence=1.0),
        "string_pollution"
    )
    with pytest.raises(TypeError, match="All items in the tuple must be AlignedWord instances."):
        _extract_transcript(invalid_tuple)  # type: ignore


def test_process_segment_expected_behaviour_for_valid_segment(standard_segment: SynchronizedSegment) -> None:
    """Tests the complete end-to-end payload extraction from a raw synchronized orchestrator segment."""
    payload: SemanticDescriptorPayload = _process_segment(standard_segment)
    
    assert isinstance(payload, SemanticDescriptorPayload)
    assert payload.segment_id == 77
    assert payload.start_time_sec == 0.0
    assert payload.end_time_sec == 1.5
    assert payload.transcript_text == "I deployed the pipeline."


def test_process_segment_raises_for_invalid_segment_type() -> None:
    """Ensures the wrapper gracefully intercepts and rejects spoofed data models."""
    with pytest.raises(TypeError, match="Input must be a strictly typed SynchronizedSegment."):
        _process_segment({"segment_id": 1, "words": []})  # type: ignore


def test_aggregate_payloads_expected_behaviour_for_multiple_segments() -> None:
    """Tests functional reduction of a chronological text buffer into a single string matrix."""
    payloads: List[SemanticDescriptorPayload] = [
        SemanticDescriptorPayload(1, 0.0, 2.0, "This is the first sentence."),
        SemanticDescriptorPayload(2, 2.0, 4.0, "This is the second sentence.")
    ]
    
    aggregated: SemanticDescriptorPayload = _aggregate_payloads(payloads)
    
    assert isinstance(aggregated, SemanticDescriptorPayload)
    assert aggregated.segment_id == 0
    assert aggregated.start_time_sec == 0.0
    assert aggregated.end_time_sec == 4.0
    assert aggregated.transcript_text == "This is the first sentence. This is the second sentence."


def test_aggregate_payloads_raises_for_empty_sequence() -> None:
    """Tests that text averaging blocks empty sequence inputs."""
    with pytest.raises(ValueError, match="Aggregation requires a strictly populated sequence of payloads."):
        _aggregate_payloads([])


def test_run_semantic_node_expected_behaviour_for_clean_termination() -> None:
    """Tests deterministic queue shutdown upon sentinel payload ingestion."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    in_q.put(OrchestratorTerminalSentinel())
    
    worker = threading.Thread(target=run_semantic_node, args=(in_q, out_q), kwargs={"timeout_sec": 0.1}, daemon=True)
    worker.start()

    assert isinstance(out_q.get(timeout=1.0), OrchestratorTerminalSentinel)


def test_run_semantic_node_expected_behaviour_for_processing_valid_stream(
    standard_segment: SynchronizedSegment,
    secondary_segment: SynchronizedSegment
) -> None:
    """Tests holistic stream-reduce lifecycle computing sequential transcript concatenations."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    in_q.put(standard_segment)
    in_q.put(secondary_segment)
    in_q.put(OrchestratorTerminalSentinel())

    worker = threading.Thread(target=run_semantic_node, args=(in_q, out_q), kwargs={"timeout_sec": 0.1}, daemon=True)
    worker.start()

    payload: Any = out_q.get(timeout=2.0)
    assert isinstance(payload, SemanticDescriptorPayload)
    assert payload.segment_id == 0
    assert payload.start_time_sec == 0.0
    assert payload.end_time_sec == 3.5
    assert payload.transcript_text == "I deployed the pipeline. Yes, absolutely."
    
    sentinel: Any = out_q.get(timeout=2.0)
    assert isinstance(sentinel, OrchestratorTerminalSentinel)


def test_run_semantic_node_raises_for_invalid_input_queue_type() -> None:
    """Tests structural validation on cross-process communication bounds."""
    with pytest.raises(TypeError, match="Communication buffers must be instances of queue.Queue."):
        run_semantic_node(input_queue=[], output_queue=Queue())  # type: ignore


def test_run_semantic_node_raises_for_invalid_output_queue_type() -> None:
    """Tests structural validation on cross-process communication bounds."""
    with pytest.raises(TypeError, match="Communication buffers must be instances of queue.Queue."):
        run_semantic_node(input_queue=Queue(), output_queue=None)  # type: ignore


def test_run_semantic_node_raises_for_saturated_downstream_queue(standard_segment: SynchronizedSegment) -> None:
    """Tests process crash isolation if the Late Fusion Nexus bottlenecks terminal markers."""
    in_q: Queue = Queue()
    out_q: Queue = Queue(maxsize=1)

    out_q.put("block_the_queue")
    in_q.put(standard_segment)
    in_q.put(OrchestratorTerminalSentinel())

    with pytest.raises(RuntimeError, match="Semantic aggregation failure"):
        run_semantic_node(input_queue=in_q, output_queue=out_q, timeout_sec=0.1)


def test_run_semantic_node_raises_for_processing_failure() -> None:
    """Tests failure propagation and downstream sentinel transmission during matrix collapse."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    corrupted_segment = SynchronizedSegment(
        segment_id=1,
        words=("this_string_will_crash_the_extractor",),  # type: ignore
        audio_slice_start_sec=0.0,
        audio_slice_end_sec=2.0,  
        visual_frame_start_idx=0,
        visual_frame_end_idx=1,
        is_final=True
    )

    in_q.put(corrupted_segment)

    with pytest.raises(RuntimeError, match="Semantic pipeline execution failed"):
        run_semantic_node(input_queue=in_q, output_queue=out_q, timeout_sec=2.0)

    sentinel: Any = out_q.get_nowait()
    assert isinstance(sentinel, OrchestratorTerminalSentinel)