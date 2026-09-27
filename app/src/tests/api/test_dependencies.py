import queue
import threading

import pytest
import torch

from app.src.api.dependencies import PipelineState


@pytest.fixture
def hardware_check() -> None:
    """Validates GPU/CPU tensor accessibility to mirror runtime orchestration constraints."""
    _ = torch.cuda.is_available()


# =====================================================================
# EXPECTED BEHAVIOUR
# =====================================================================

def test_pipeline_state_expected_behaviour_for_default_initialization(hardware_check: None) -> None:
    """Verifies that the default constructor safely allocates independent queues for all attributes."""
    state = PipelineState()
    
    assert isinstance(state.q_transcription, queue.Queue)
    assert isinstance(state.q_window, queue.Queue)
    assert isinstance(state.q_acoustic, queue.Queue)
    assert isinstance(state.q_semantic, queue.Queue)
    assert isinstance(state.q_visual_facial, queue.Queue)
    assert isinstance(state.q_visual_skeletal, queue.Queue)
    assert isinstance(state.q_nexus_out, queue.Queue)
    assert isinstance(state.q_judge_out, queue.Queue)
    
    assert isinstance(state.active_threads, list)
    assert len(state.active_threads) == 0
    
    # Verify strict memory isolation (no shared references across defaults)
    assert state.q_transcription is not state.q_window
    assert state.q_acoustic is not state.q_semantic


def test_pipeline_state_expected_behaviour_for_injected_allocation(hardware_check: None) -> None:
    """Verifies state registry properly binds to pre-allocated thread and queue references."""
    custom_q: queue.Queue = queue.Queue(maxsize=10)
    
    def _dummy_worker() -> None:
        pass
        
    dummy_thread = threading.Thread(target=_dummy_worker)
    
    state = PipelineState(
        q_judge_out=custom_q,
        active_threads=[dummy_thread]
    )
    
    assert state.q_judge_out is custom_q
    assert state.q_judge_out.maxsize == 10
    assert len(state.active_threads) == 1
    assert state.active_threads[0] is dummy_thread


# =====================================================================
# ERROR CASES & EDGE CONDITIONS
# =====================================================================

def test_pipeline_state_raises_for_invalid_queue_type() -> None:
    """Verifies structural rejection of unmapped lists masquerading as asynchronous buffers."""
    with pytest.raises(TypeError, match="q_transcription must be a queue.Queue instance."):
        PipelineState(q_transcription=[])  # type: ignore


def test_pipeline_state_raises_for_invalid_thread_registry_type() -> None:
    """Verifies structural rejection of non-list objects passed to the thread registry."""
    with pytest.raises(TypeError, match="active_threads must be a list instance."):
        PipelineState(active_threads=tuple())  # type: ignore


def test_pipeline_state_raises_for_invalid_thread_element_type() -> None:
    """Verifies strict python typing constraints on objects stored inside the thread registry."""
    with pytest.raises(TypeError, match="Elements of active_threads must be threading.Thread instances."):
        PipelineState(active_threads=["thread_1_string"])  # type: ignore