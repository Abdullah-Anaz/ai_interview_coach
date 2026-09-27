import logging
import queue
import threading
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, List

from fastapi import FastAPI

from app.src.acoustic_edge.config import AcousticConfig
from app.src.acoustic_edge.node import run_acoustic_node
from app.src.api.dependencies import PipelineState, global_pipeline_state
from app.src.apex_judge.config import JudgeConfig, ModelProvider
from app.src.apex_judge.node import run_llm_judge_node
from app.src.fusion_nexus.node import run_fusion_nexus_node
from app.src.orchestration.orchestrator import (
    OrchestratorTerminalSentinel,
    run_orchestrator_node,
)
from app.src.semantic_edge.node import run_semantic_node
from app.src.transcription.config import TranscriptionConfig
from app.src.transcription.transcriber import run_transcription_node
from app.src.visual_edge.facial_gaze.config import FacialConfig
from app.src.visual_edge.facial_gaze.node import run_facial_node
from app.src.visual_edge.skeletal_kinematics.config import SkeletalConfig
from app.src.visual_edge.skeletal_kinematics.node import run_skeletal_node

logger = logging.getLogger(__name__)


def _safe_thread_wrapper(target_func: Callable[..., Any], **kwargs: Any) -> None:
    """
    Executes a target function safely, catching and logging unhandled exceptions.
    Attempts to inject a terminal sentinel into egress queues upon catastrophic failure.

    Args:
        target_func (Callable[..., Any]): The core node loop to execute.
        **kwargs: The parameter map injected into the node.
    """
    try:
        target_func(**kwargs)
    except Exception as exc:
        logger.error("Catastrophic failure in pipeline node '%s': %s", target_func.__name__, exc)
        logger.error(traceback.format_exc())
        
        for key, value in kwargs.items():
            if "queue" in key.lower() or key.endswith("_q") or key.endswith("_out"):
                try:
                    if hasattr(value, "put"):
                        value.put(OrchestratorTerminalSentinel(error=exc))
                except Exception:
                    pass


def _spawn_daemon(target: Callable[..., Any], kwargs: Dict[str, Any]) -> threading.Thread:
    """
    Allocates and executes a continuous execution loop on an isolated background thread.

    Args:
        target (Callable[..., Any]): The microservice node function to execute.
        kwargs (Dict[str, Any]): The parameter map injected into the node.

    Returns:
        threading.Thread: The initialized and running thread handle.

    Raises:
        TypeError: If the target is not callable or kwargs is not a dictionary.
    """
    if not callable(target):
        raise TypeError("target must be a callable function.")
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dictionary.")

    thread = threading.Thread(target=target, kwargs=kwargs, daemon=True)
    thread.start()
    return thread


def spawn_pipeline_node(
    state: PipelineState,
    target: Callable[..., Any],
    kwargs: Dict[str, Any]
) -> None:
    """
    Spawns a pipeline node and registers its thread in the global state matrix.

    Args:
        state (PipelineState): The central registry for the pipeline.
        target (Callable[..., Any]): The node function to execute.
        kwargs (Dict[str, Any]): Arguments injected into the node.

    Raises:
        TypeError: If state is not a PipelineState instance.
    """
    if not isinstance(state, PipelineState):
        raise TypeError("state must be a PipelineState instance.")
    
    thread = _spawn_daemon(target, kwargs)
    state.active_threads.append(thread)


def shutdown_matrix(state: PipelineState, timeout_sec: float = 120.0) -> None:
    """
    Executes a graceful teardown cascade across all active pipeline threads.

    Injects terminal sentinels into all network buffers to unblock waiting queues,
    then systematically joins all registered threads to prevent memory leaks.

    Args:
        state (PipelineState): The central registry containing active buffers and threads.
        timeout_sec (float): The maximum time in seconds to wait for a thread to join.

    Raises:
        TypeError: If the state or timeout violates strict python types.
        ValueError: If timeout_sec is negative.
    """
    if not isinstance(state, PipelineState):
        raise TypeError("state must be a PipelineState instance.")
    if not isinstance(timeout_sec, float):
        raise TypeError("timeout_sec must be a strictly typed float.")
    if timeout_sec <= 0.0:
        raise ValueError("timeout_sec must be strictly positive.")

    sentinel = OrchestratorTerminalSentinel()
    
    queues = [
        state.q_transcription,
        state.q_window,
        state.q_acoustic,
        state.q_semantic,
        state.q_visual_facial,
        state.q_visual_skeletal,
        state.q_nexus_out,
        state.q_judge_out
    ]

    for q in queues:
        try:
            q.put_nowait(sentinel)
        except Exception:
            pass

    for thread in state.active_threads:
        if thread.is_alive():
            thread.join(timeout=timeout_sec)


@asynccontextmanager
async def manage_pipeline_lifecycle(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    ASGI lifecycle hook governing the startup allocation and teardown of the execution matrix.

    Args:
        app (FastAPI): The ASGI application instance.

    Yields:
        None: Suspends execution while the ASGI server processes active connections.
    """
    asr_to_semantic_queue: queue.Queue = queue.Queue()
    nexus_semantic_in: queue.Queue = queue.Queue()
    nexus_acoustic_in: queue.Queue = queue.Queue()
    nexus_facial_in: queue.Queue = queue.Queue()
    nexus_skeletal_in: queue.Queue = queue.Queue()

    logger.info("Booting Orchestrator Node...")
    spawn_pipeline_node(
        state=global_pipeline_state,
        target=_safe_thread_wrapper,
        kwargs={
            "target_func": run_orchestrator_node,
            "q_transcription_in": global_pipeline_state.q_transcription,
            "q_window_in": global_pipeline_state.q_window,
            "q_acoustic_out": global_pipeline_state.q_acoustic,
            "q_semantic_out": global_pipeline_state.q_semantic,
            "q_visual_facial_out": global_pipeline_state.q_visual_facial,
            "q_visual_skeletal_out": global_pipeline_state.q_visual_skeletal,
        }
    )

    logger.info("Booting Transcription Node (NeMo Parakeet)...")
    spawn_pipeline_node(
        state=global_pipeline_state,
        target=_safe_thread_wrapper,
        kwargs={
            "target_func": run_transcription_node,
            "input_queue": global_pipeline_state.q_semantic, 
            "output_queue": asr_to_semantic_queue,            
            "config": TranscriptionConfig()
        }
    )
    time.sleep(3.0)

    logger.info("Booting Semantic Node...")
    spawn_pipeline_node(
        state=global_pipeline_state,
        target=_safe_thread_wrapper,
        kwargs={
            "target_func": run_semantic_node,
            "input_queue": asr_to_semantic_queue,             
            "output_queue": nexus_semantic_in                 
        }
    )

    logger.info("Booting Acoustic Node...")
    spawn_pipeline_node(
        state=global_pipeline_state,
        target=_safe_thread_wrapper,
        kwargs={
            "target_func": run_acoustic_node,
            "input_queue": global_pipeline_state.q_acoustic,
            "output_queue": nexus_acoustic_in,
            "config": AcousticConfig()
        }
    )
    time.sleep(1.0)

    logger.info("Booting Skeletal Node...")
    spawn_pipeline_node(
        state=global_pipeline_state,
        target=_safe_thread_wrapper,
        kwargs={
            "target_func": run_skeletal_node,
            "input_queue": global_pipeline_state.q_visual_skeletal,
            "output_queue": nexus_skeletal_in,
            "config": SkeletalConfig()
        }
    )
    time.sleep(1.0)

    logger.info("Booting Facial Node...")
    spawn_pipeline_node(
        state=global_pipeline_state,
        target=_safe_thread_wrapper,
        kwargs={
            "target_func": run_facial_node, 
            "input_queue": global_pipeline_state.q_visual_facial,
            "output_queue": nexus_facial_in,
            "config": FacialConfig()
        }
    )
    time.sleep(1.0)

    logger.info("Booting Fusion & Judge Nodes...")
    spawn_pipeline_node(
        state=global_pipeline_state,
        target=_safe_thread_wrapper,
        kwargs={
            "target_func": run_fusion_nexus_node,
            "semantic_q": nexus_semantic_in,
            "acoustic_q": nexus_acoustic_in,
            "facial_q": nexus_facial_in,
            "skeletal_q": nexus_skeletal_in,
            "output_q": global_pipeline_state.q_nexus_out,
            "ttl_seconds": 600.0 
        }
    )

    judge_config = JudgeConfig(
        provider=ModelProvider.GEMINI,
        model_name="gemini-3.8-flash",
        temperature=0.1,
        max_tokens=256,
        device="cpu"
    )
    spawn_pipeline_node(
        state=global_pipeline_state,
        target=_safe_thread_wrapper,
        kwargs={
            "target_func": run_llm_judge_node,
            "input_q": global_pipeline_state.q_nexus_out,
            "output_q": global_pipeline_state.q_judge_out,
            "config": judge_config,
            "timeout_sec": 120.0
        }
    )

    yield

    logger.info("Initiating graceful shutdown matrix...")
    shutdown_matrix(state=global_pipeline_state, timeout_sec=0.1)