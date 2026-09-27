import queue
import threading
import time
from typing import Any

import pytest
import torch

from app.src.apex_judge.config import JudgeConfig, ModelProvider
from app.src.apex_judge.node import run_llm_judge_node
from app.src.apex_judge.types import CoachingFeedbackPayload
from app.src.fusion_nexus.types import FusedPromptPayload
from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel


@pytest.fixture(scope="module")
def local_gpu_config() -> JudgeConfig:
    """Provides a local HuggingFace configuration for unmocked GPU execution."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA architecture is unavailable. Skipping local LLM node tests.")
        
    return JudgeConfig(
        provider=ModelProvider.LLAMA,
        model_name="distilgpt2",
        temperature=0.1,
        max_tokens=10,
        device="cuda:0"
    )


# =====================================================================
# EXPECTED BEHAVIOUR & QUEUE LIFECYCLE
# =====================================================================

def test_run_llm_judge_node_expected_behaviour_for_processing_payload(local_gpu_config: JudgeConfig) -> None:
    """Verifies holistic queue lifecycle processing using the actual local Transformer model."""
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()

    payload = FusedPromptPayload(segment_id=1, compiled_prompt="Evaluate this candidate.")
    in_q.put(payload)
    in_q.put(OrchestratorTerminalSentinel())

    run_llm_judge_node(input_q=in_q, output_q=out_q, config=local_gpu_config, timeout_sec=5.0)

    result: Any = out_q.get(timeout=2.0)
    assert isinstance(result, CoachingFeedbackPayload)
    assert result.segment_id == 1
    assert len(result.feedback_text) > 0
    
    sentinel: Any = out_q.get(timeout=2.0)
    assert isinstance(sentinel, OrchestratorTerminalSentinel)


def test_run_llm_judge_node_expected_behaviour_for_ignoring_invalid_items(local_gpu_config: JudgeConfig) -> None:
    """Verifies the consumer loop actively drops unknown structural types without crashing the thread."""
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()

    in_q.put("corrupt_string_data")
    in_q.put({"segment_id": 99})
    in_q.put(OrchestratorTerminalSentinel())

    run_llm_judge_node(input_q=in_q, output_q=out_q, config=local_gpu_config, timeout_sec=2.0)

    sentinel: Any = out_q.get(timeout=2.0)
    assert isinstance(sentinel, OrchestratorTerminalSentinel)
    assert out_q.empty()


def test_run_llm_judge_node_expected_behaviour_for_empty_queue_timeout_recovery(local_gpu_config: JudgeConfig) -> None:
    """Verifies the loop handles queue.Empty exceptions cleanly and continues polling until a sentinel arrives."""
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    
    def delayed_insertion() -> None:
        """Injects a sentinel natively from a parallel thread after a simulated delay."""
        time.sleep(0.1)
        in_q.put(OrchestratorTerminalSentinel())
        
    threading.Thread(target=delayed_insertion, daemon=True).start()
    
    # The timeout is shorter than the thread delay, forcing the node to recover from queue.Empty at least once
    run_llm_judge_node(input_q=in_q, output_q=out_q, config=local_gpu_config, timeout_sec=0.01)
    
    assert isinstance(out_q.get(timeout=1.0), OrchestratorTerminalSentinel)


# =====================================================================
# ERROR CASES & TYPE POLLUTION
# =====================================================================

def test_run_llm_judge_node_raises_for_invalid_input_queue_type(local_gpu_config: JudgeConfig) -> None:
    """Verifies strict architectural type checking on the ingress queue parameter."""
    with pytest.raises(TypeError, match="input_q must be an instance of queue.Queue."):
        run_llm_judge_node(input_q=[], output_q=queue.Queue(), config=local_gpu_config)  # type: ignore


def test_run_llm_judge_node_raises_for_invalid_output_queue_type(local_gpu_config: JudgeConfig) -> None:
    """Verifies strict architectural type checking on the egress queue parameter."""
    with pytest.raises(TypeError, match="output_q must be an instance of queue.Queue."):
        run_llm_judge_node(input_q=queue.Queue(), output_q=None, config=local_gpu_config)  # type: ignore


def test_run_llm_judge_node_raises_for_initialization_failure() -> None:
    """Verifies catastrophic initialization failure sequentially propagates an error sentinel downstream."""
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    
    broken_config = JudgeConfig.__new__(JudgeConfig)
    object.__setattr__(broken_config, "provider", "INVALID_PROVIDER_ENUM")
    
    with pytest.raises(RuntimeError, match="Judge node initialization failed"):
        run_llm_judge_node(input_q=in_q, output_q=out_q, config=broken_config)
        
    assert isinstance(out_q.get_nowait(), OrchestratorTerminalSentinel)


def test_run_llm_judge_node_raises_for_saturated_output_queue_on_termination(local_gpu_config: JudgeConfig) -> None:
    """Verifies protective encapsulation when the downstream UI queue saturates during teardown."""
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue(maxsize=1)
    
    out_q.put("blocking_element")
    in_q.put(OrchestratorTerminalSentinel())
    
    with pytest.raises(RuntimeError, match="Output queue saturated during terminal sequence."):
        run_llm_judge_node(input_q=in_q, output_q=out_q, config=local_gpu_config, timeout_sec=0.05)


def test_run_llm_judge_node_raises_for_saturated_output_queue_on_payload(local_gpu_config: JudgeConfig) -> None:
    """Verifies protective encapsulation when the downstream UI queue saturates during active processing."""
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue(maxsize=1)
    
    out_q.put("blocking_element")
    in_q.put(FusedPromptPayload(segment_id=42, compiled_prompt="Valid prompt."))
    
    # Because the exception handler in node.py attempts a SECOND put() which will also trigger queue.Full
    with pytest.raises((RuntimeError, queue.Full)):
        run_llm_judge_node(input_q=in_q, output_q=out_q, config=local_gpu_config, timeout_sec=0.05)


def test_run_llm_judge_node_raises_for_processing_failure(local_gpu_config: JudgeConfig) -> None:
    """Verifies internal inference collapse routes an error sentinel downstream and raises a RuntimeError."""
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    
    # Bypass the strict DTO validation to inject a native NoneType, forcing HF to crash natively
    poison_payload = FusedPromptPayload.__new__(FusedPromptPayload)
    object.__setattr__(poison_payload, "segment_id", 99)
    object.__setattr__(poison_payload, "compiled_prompt", None)
    
    in_q.put(poison_payload)
    
    with pytest.raises(RuntimeError, match="Judge evaluation failure on segment 99"):
        run_llm_judge_node(input_q=in_q, output_q=out_q, config=local_gpu_config, timeout_sec=1.0)
        
    sentinel: Any = out_q.get_nowait()
    assert isinstance(sentinel, OrchestratorTerminalSentinel)