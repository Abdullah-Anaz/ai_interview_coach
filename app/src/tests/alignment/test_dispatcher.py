from queue import Queue

import pytest

from app.src.alignment.dispatcher import dispatch_synchronized_segment
from app.src.orchestration.types import SynchronizedSegment


def _create_dummy_segment() -> SynchronizedSegment:
    """
    Helper to generate a structurally valid SynchronizedSegment for routing tests.
    """
    return SynchronizedSegment(
        segment_id=1,
        words=tuple(),
        audio_slice_start_sec=0.0,
        audio_slice_end_sec=1.0,
        visual_frame_start_idx=0,
        visual_frame_end_idx=50,
        is_final=False
    )


def test_dispatch_synchronized_segment_expected_behaviour_for_valid_routing() -> None:
    """
    Validates seamless fan-out of a segment to all three target queues.
    """
    acoustic_q: Queue = Queue()
    visual_q: Queue = Queue()
    semantic_q: Queue = Queue()
    segment: SynchronizedSegment = _create_dummy_segment()
    
    dispatch_synchronized_segment(
        segment=segment,
        acoustic_queue=acoustic_q,
        visual_queue=visual_q,
        semantic_queue=semantic_q,
        timeout_sec=0.1
    )
    
    # Ensure all queues received exactly one payload
    assert acoustic_q.qsize() == 1
    assert visual_q.qsize() == 1
    assert semantic_q.qsize() == 1
    
    # Verify the exact object reference was passed without mutation
    assert acoustic_q.get_nowait() is segment
    assert visual_q.get_nowait() is segment
    assert semantic_q.get_nowait() is segment


def test_dispatch_synchronized_segment_raises_for_acoustic_queue_saturation() -> None:
    """
    Error Case: Validates the dispatcher aborts if the Acoustic Edge is bottlenecked.
    """
    acoustic_q: Queue = Queue(maxsize=1)
    visual_q: Queue = Queue()
    semantic_q: Queue = Queue()
    
    # Pre-fill to force a saturation block
    acoustic_q.put("blocker")
    
    with pytest.raises(RuntimeError) as exc_info:
        dispatch_synchronized_segment(
            segment=_create_dummy_segment(),
            acoustic_queue=acoustic_q,
            visual_queue=visual_q,
            semantic_queue=semantic_q,
            timeout_sec=0.1
        )
        
    assert "Acoustic Edge queue saturation" in str(exc_info.value)


def test_dispatch_synchronized_segment_raises_for_visual_queue_saturation() -> None:
    """
    Error Case: Validates the dispatcher aborts if the Visual Edge is bottlenecked.
    """
    acoustic_q: Queue = Queue()
    visual_q: Queue = Queue(maxsize=1)
    semantic_q: Queue = Queue()
    
    # Pre-fill to force a saturation block
    visual_q.put("blocker")
    
    with pytest.raises(RuntimeError) as exc_info:
        dispatch_synchronized_segment(
            segment=_create_dummy_segment(),
            acoustic_queue=acoustic_q,
            visual_queue=visual_q,
            semantic_queue=semantic_q,
            timeout_sec=0.1
        )
        
    assert "Visual Edge queue saturation" in str(exc_info.value)


def test_dispatch_synchronized_segment_raises_for_semantic_queue_saturation() -> None:
    """
    Error Case: Validates the dispatcher aborts if the Semantic Edge is bottlenecked.
    """
    acoustic_q: Queue = Queue()
    visual_q: Queue = Queue()
    semantic_q: Queue = Queue(maxsize=1)
    
    # Pre-fill to force a saturation block
    semantic_q.put("blocker")
    
    with pytest.raises(RuntimeError) as exc_info:
        dispatch_synchronized_segment(
            segment=_create_dummy_segment(),
            acoustic_queue=acoustic_q,
            visual_queue=visual_q,
            semantic_queue=semantic_q,
            timeout_sec=0.1
        )
        
    assert "Semantic Edge queue saturation" in str(exc_info.value)