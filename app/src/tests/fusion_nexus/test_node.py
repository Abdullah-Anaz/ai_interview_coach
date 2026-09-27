import queue
import threading
import time
from typing import Any, List

import pytest

from app.src.acoustic_edge.types import AcousticDescriptorPayload
from app.src.fusion_nexus.node import (
    _compile_segment_state,
    _emit_terminal_payload,
    run_fusion_nexus_node,
)
from app.src.fusion_nexus.types import FusedPromptPayload, SegmentBufferState
from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.semantic_edge.types import SemanticDescriptorPayload
from app.src.visual_edge.facial_gaze.types import FacialDescriptorPayload
from app.src.visual_edge.skeletal_kinematics.types import SkeletalDescriptorPayload


@pytest.fixture
def base_semantic_payload() -> SemanticDescriptorPayload:
    return SemanticDescriptorPayload(1, 0.0, 2.5, "I delivered the project.")


@pytest.fixture
def base_acoustic_payload() -> AcousticDescriptorPayload:
    return AcousticDescriptorPayload(1, 0.0, 2.5, ("high extraversion",), 0.8, 0.2, 0.5, 0.9, 0.6)


@pytest.fixture
def base_facial_payload() -> FacialDescriptorPayload:
    return FacialDescriptorPayload("1", 0.0, 2.5, "strong eye contact", 0.8, 0.2, 0.7, 0.9, 0.6)


@pytest.fixture
def base_skeletal_payload() -> SkeletalDescriptorPayload:
    return SkeletalDescriptorPayload(1, 0.0, 2.5, "low jitter", 0.05, 0.95)


def test_compile_segment_state_expected_behaviour_for_successful_compilation(
    base_semantic_payload: SemanticDescriptorPayload
) -> None:
    """Verifies that a buffer state correctly serializes into a fused LLM prompt string."""
    state = SegmentBufferState(segment_id=42)
    state.semantic_payload = base_semantic_payload

    result: str = _compile_segment_state(state)
    
    assert isinstance(result, str)
    assert "I delivered the project." in result


def test_emit_terminal_payload_expected_behaviour() -> None:
    """Verifies that compiled segments are merged and emitted followed by a terminal sentinel."""
    out_q: queue.Queue = queue.Queue()
    prompts: List[str] = ["Segment A", "Segment B"]
    
    _emit_terminal_payload(prompts, out_q, poll_timeout=1.0)
    
    payload: Any = out_q.get(timeout=1.0)
    assert isinstance(payload, FusedPromptPayload)
    assert payload.segment_id == 0
    assert "Segment A" in payload.compiled_prompt
    assert "Segment B" in payload.compiled_prompt
    
    sentinel: Any = out_q.get(timeout=1.0)
    assert isinstance(sentinel, OrchestratorTerminalSentinel)


def test_emit_terminal_payload_raises_for_saturated_queue() -> None:
    """Verifies protective encapsulation when the downstream LLM queue hits memory limits."""
    out_q: queue.Queue = queue.Queue(maxsize=1)
    out_q.put("block")
    
    with pytest.raises(RuntimeError, match="Nexus output queue saturated"):
        _emit_terminal_payload(["Prompt"], out_q, poll_timeout=0.01)


def test_run_fusion_nexus_node_expected_behaviour_for_synchronous_completion(
    base_semantic_payload: SemanticDescriptorPayload,
    base_acoustic_payload: AcousticDescriptorPayload,
    base_facial_payload: FacialDescriptorPayload,
    base_skeletal_payload: SkeletalDescriptorPayload
) -> None:
    """Tests flawless synchronous integration when all four parallel sub-systems deliver payloads simultaneously."""
    sem_q, ac_q, fac_q, skel_q, out_q = [queue.Queue() for _ in range(5)]
    
    sem_q.put(base_semantic_payload)
    ac_q.put(base_acoustic_payload)
    fac_q.put(base_facial_payload)
    skel_q.put(base_skeletal_payload)

    for q in (sem_q, ac_q, fac_q, skel_q):
        q.put(OrchestratorTerminalSentinel())

    worker = threading.Thread(
        target=run_fusion_nexus_node,
        args=(sem_q, ac_q, fac_q, skel_q, out_q),
        kwargs={"ttl_seconds": 5.0, "poll_timeout": 0.05},
        daemon=True
    )
    worker.start()

    result = out_q.get(timeout=2.0)
    assert isinstance(result, FusedPromptPayload)
    assert result.segment_id == 0
    
    sentinel = out_q.get(timeout=2.0)
    assert isinstance(sentinel, OrchestratorTerminalSentinel)


def test_run_fusion_nexus_node_expected_behaviour_for_ttl_degradation(
    base_semantic_payload: SemanticDescriptorPayload,
    base_facial_payload: FacialDescriptorPayload
) -> None:
    """Tests asynchronous Time-To-Live logic gracefully degrading and compiling missing streams for late fusion."""
    sem_q, ac_q, fac_q, skel_q, out_q = [queue.Queue() for _ in range(5)]
    
    sem_q.put(base_semantic_payload)
    fac_q.put(base_facial_payload)

    time.sleep(0.1)

    for q in (sem_q, ac_q, fac_q, skel_q):
        q.put(OrchestratorTerminalSentinel())

    worker = threading.Thread(
        target=run_fusion_nexus_node,
        args=(sem_q, ac_q, fac_q, skel_q, out_q),
        kwargs={"ttl_seconds": 0.05, "poll_timeout": 0.05},
        daemon=True
    )
    worker.start()

    result = out_q.get(timeout=2.0)
    assert isinstance(result, FusedPromptPayload)
    assert result.segment_id == 0
    assert "<posture_kinematics>\nUNAVAILABLE\n</posture_kinematics>" in result.compiled_prompt


def test_run_fusion_nexus_node_expected_behaviour_for_out_of_order_arrival(
    base_semantic_payload: SemanticDescriptorPayload,
    base_acoustic_payload: AcousticDescriptorPayload,
    base_facial_payload: FacialDescriptorPayload,
    base_skeletal_payload: SkeletalDescriptorPayload
) -> None:
    """Tests the aggregation matrix resolving unordered DTOs dynamically before flushing the terminal sequence."""
    sem_q, ac_q, fac_q, skel_q, out_q = [queue.Queue() for _ in range(5)]
    
    ac_q.put(base_acoustic_payload)
    skel_q.put(base_skeletal_payload)
    fac_q.put(base_facial_payload)
    sem_q.put(base_semantic_payload)

    for q in (sem_q, ac_q, fac_q, skel_q):
        q.put(OrchestratorTerminalSentinel())

    worker = threading.Thread(
        target=run_fusion_nexus_node,
        args=(sem_q, ac_q, fac_q, skel_q, out_q),
        kwargs={"ttl_seconds": 5.0, "poll_timeout": 0.05},
        daemon=True
    )
    worker.start()

    result = out_q.get(timeout=2.0)
    assert isinstance(result, FusedPromptPayload)
    assert result.segment_id == 0


def test_run_fusion_nexus_node_expected_behaviour_for_ignoring_invalid_items() -> None:
    """Tests strict routing loops dropping unmapped objects or missing schema signatures without crashing."""
    sem_q, ac_q, fac_q, skel_q, out_q = [queue.Queue() for _ in range(5)]
    
    sem_q.put("invalid_data")
    ac_q.put({"segment_id": "invalid_alpha"})
    fac_q.put({"segment_id": -5})
    
    for q in (sem_q, ac_q, fac_q, skel_q):
        q.put(OrchestratorTerminalSentinel())

    worker = threading.Thread(
        target=run_fusion_nexus_node,
        args=(sem_q, ac_q, fac_q, skel_q, out_q),
        kwargs={"ttl_seconds": 5.0, "poll_timeout": 0.05},
        daemon=True
    )
    worker.start()

    sentinel = out_q.get(timeout=2.0)
    assert isinstance(sentinel, OrchestratorTerminalSentinel)
    assert out_q.empty()


def test_run_fusion_nexus_node_raises_for_invalid_queue_types() -> None:
    """Tests strictly typed instantiation blocking dynamically incorrect buffer allocations."""
    sem_q, ac_q, fac_q, skel_q, out_q = [queue.Queue() for _ in range(5)]
    
    with pytest.raises(TypeError, match="semantic ingress must be a strictly typed queue.Queue."):
        run_fusion_nexus_node([], ac_q, fac_q, skel_q, out_q)  # type: ignore

    with pytest.raises(TypeError, match="output_q egress must be a strictly typed queue.Queue."):
        run_fusion_nexus_node(sem_q, ac_q, fac_q, skel_q, None)  # type: ignore