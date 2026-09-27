import queue
from typing import Any

from app.src.apex_judge.config import JudgeConfig
from app.src.apex_judge.router import execute_evaluation, load_llm_backend
from app.src.apex_judge.types import CoachingFeedbackPayload
from app.src.fusion_nexus.types import FusedPromptPayload
from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel


def run_llm_judge_node(
    input_q: queue.Queue,
    output_q: queue.Queue,
    config: JudgeConfig,
    timeout_sec: float = 5.0
) -> None:
    """
    Executes the continuous Apex LLM consumer loop.

    Args:
        input_q (queue.Queue): Ingress buffer supplying FusedPromptPayload objects.
        output_q (queue.Queue): Egress buffer receiving CoachingFeedbackPayload objects.
        config (JudgeConfig): Backend configuration targeting GPT, Gemini, or local LLaMA.
        timeout_sec (float): Block timeout in seconds for queue retrieval.

    Raises:
        TypeError: If queue parameters violate structural constraints.
        RuntimeError: If model initialization fails or processing crashes.
    """
    if not isinstance(input_q, queue.Queue):
        raise TypeError("input_q must be an instance of queue.Queue.")
    if not isinstance(output_q, queue.Queue):
        raise TypeError("output_q must be an instance of queue.Queue.")

    try:
        client = load_llm_backend(config)
    except Exception as exc:
        output_q.put(OrchestratorTerminalSentinel())
        raise RuntimeError(f"Judge node initialization failed: {exc}") from exc

    while True:
        try:
            item: Any = input_q.get(timeout=timeout_sec)
        except queue.Empty:
            continue

        if isinstance(item, OrchestratorTerminalSentinel):
            try:
                output_q.put(item, timeout=timeout_sec)
            except queue.Full as exc:
                raise RuntimeError("Output queue saturated during terminal sequence.") from exc
            break

        if not isinstance(item, FusedPromptPayload):
            continue

        try:
            feedback_str: str = execute_evaluation(item.compiled_prompt, client)
            payload = CoachingFeedbackPayload(
                segment_id=item.segment_id,
                feedback_text=feedback_str
            )
            output_q.put(payload, timeout=timeout_sec)
        except queue.Full as exc:
            output_q.put(OrchestratorTerminalSentinel(), timeout=timeout_sec)
            raise RuntimeError("Judge output queue saturated.") from exc
        except Exception as exc:
            output_q.put(OrchestratorTerminalSentinel(), timeout=timeout_sec)
            raise RuntimeError(f"Judge evaluation failure on segment {item.segment_id}: {exc}") from exc