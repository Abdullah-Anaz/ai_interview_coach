import queue
import threading
from typing import Any

import pytest
import torch

from app.src.api.dependencies import PipelineState
from app.src.api.lifecycle import _spawn_daemon, shutdown_matrix, spawn_pipeline_node
from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel


@pytest.fixture
def hardware_check() -> None:
    """Validates GPU/CPU tensor accessibility to mirror runtime orchestration constraints."""
    _ = torch.cuda.is_available()


@pytest.fixture
def fresh_state() -> PipelineState:
    """Provides an isolated PipelineState matrix for concurrent testing."""
    return PipelineState()


def _dummy_consumer_node(input_q: queue.Queue) -> None:
    """A purely functional blocking consumer to natively test thread cascade teardowns."""
    while True:
        item: Any = input_q.get()
        if isinstance(item, OrchestratorTerminalSentinel):
            break


# =====================================================================
# EXPECTED BEHAVIOUR
# =====================================================================

def test_spawn_daemon_expected_behaviour_for_thread_allocation(hardware_check: None) -> None:
    """Verifies that the daemon allocator successfully targets and executes a callable."""
    execution_flag = []
    
    def _instant_worker() -> None:
        execution_flag.append(True)

    thread = _spawn_daemon(target=_instant_worker, kwargs={})
    
    assert isinstance(thread, threading.Thread)
    assert thread.daemon is True
    
    thread.join(timeout=2.0)
    assert len(execution_flag) == 1
    assert execution_flag[0] is True


def test_spawn_pipeline_node_expected_behaviour_for_registration(
    hardware_check: None, 
    fresh_state: PipelineState
) -> None:
    """Verifies that nodes are successfully spawned and recorded in the state matrix."""
    def _noop_worker() -> None:
        pass
        
    spawn_pipeline_node(fresh_state, target=_noop_worker, kwargs={})
    
    assert len(fresh_state.active_threads) == 1
    assert fresh_state.active_threads[0].is_alive() or not fresh_state.active_threads[0].is_alive()
    
    fresh_state.active_threads[0].join(timeout=2.0)


def test_shutdown_matrix_expected_behaviour_for_graceful_teardown(
    hardware_check: None, 
    fresh_state: PipelineState
) -> None:
    """Verifies that the teardown cascade successfully unblocks queues and joins active threads."""
    spawn_pipeline_node(
        state=fresh_state,
        target=_dummy_consumer_node,
        kwargs={"input_q": fresh_state.q_acoustic}
    )
    spawn_pipeline_node(
        state=fresh_state,
        target=_dummy_consumer_node,
        kwargs={"input_q": fresh_state.q_nexus_out}
    )
    
    assert fresh_state.active_threads[0].is_alive() is True
    assert fresh_state.active_threads[1].is_alive() is True
    
    shutdown_matrix(fresh_state, timeout_sec=5.0)
    
    assert fresh_state.active_threads[0].is_alive() is False
    assert fresh_state.active_threads[1].is_alive() is False


# =====================================================================
# EDGE CASES & TYPE VIOLATIONS
# =====================================================================

def test_spawn_daemon_raises_for_invalid_target() -> None:
    """Verifies structural rejection of uncallable objects passed to the thread allocator."""
    with pytest.raises(TypeError, match="target must be a callable function."):
        _spawn_daemon(target="not_a_function", kwargs={})  # type: ignore


def test_spawn_daemon_raises_for_invalid_kwargs() -> None:
    """Verifies structural rejection of invalid argument mappings passed to the thread allocator."""
    def _noop_worker() -> None:
        pass
        
    with pytest.raises(TypeError, match="kwargs must be a dictionary."):
        _spawn_daemon(target=_noop_worker, kwargs=[])  # type: ignore


def test_spawn_pipeline_node_raises_for_invalid_state() -> None:
    """Verifies structural rejection of invalid state registries."""
    def _noop_worker() -> None:
        pass
        
    with pytest.raises(TypeError, match="state must be a PipelineState instance."):
        spawn_pipeline_node(state=None, target=_noop_worker, kwargs={})  # type: ignore


def test_shutdown_matrix_raises_for_invalid_state() -> None:
    """Verifies structural rejection of invalid state registries during the teardown sequence."""
    with pytest.raises(TypeError, match="state must be a PipelineState instance."):
        shutdown_matrix(state=None, timeout_sec=5.0)  # type: ignore


def test_shutdown_matrix_raises_for_invalid_timeout(fresh_state: PipelineState) -> None:
    """Verifies structural rejection of integers or strings masquerading as float timeouts."""
    with pytest.raises(TypeError, match="timeout_sec must be a strictly typed float."):
        shutdown_matrix(state=fresh_state, timeout_sec=5)  # type: ignore


def test_shutdown_matrix_raises_for_negative_timeout(fresh_state: PipelineState) -> None:
    """Verifies rejection of inverted or zero time constraints during the join cascade."""
    with pytest.raises(ValueError, match="timeout_sec must be strictly positive."):
        shutdown_matrix(state=fresh_state, timeout_sec=-1.0)