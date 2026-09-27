import logging
import queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from app.src.acoustic_edge.config import AcousticConfig
from app.src.acoustic_edge.node import run_acoustic_node
from app.src.api.dependencies import global_pipeline_state
from app.src.api.lifecycle import shutdown_matrix, spawn_pipeline_node
from app.src.apex_judge.config import JudgeConfig, ModelProvider
from app.src.apex_judge.node import run_llm_judge_node
from app.src.apex_judge.types import CoachingFeedbackPayload
from app.src.fusion_nexus.node import run_fusion_nexus_node
from app.src.orchestration.orchestrator import (
    OrchestratorTerminalSentinel,
    run_orchestrator_node,
)
from app.src.semantic_edge.node import run_semantic_node
from app.src.stream_ingestion.config import IngestionConfig, IngestionMode
from app.src.stream_ingestion.stream_simulator import execute_simulator_ingestion
from app.src.visual_edge.facial_gaze.config import FacialConfig
from app.src.visual_edge.facial_gaze.node import run_facial_node
from app.src.visual_edge.skeletal_kinematics.config import SkeletalConfig
from app.src.visual_edge.skeletal_kinematics.node import run_skeletal_node
from app.src.transcription.config import TranscriptionConfig
from app.src.transcription.transcriber import run_transcription_node

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def _verify_video_target(target_path: Path) -> None:
    """
    Validates the existence of the target simulation video file.

    Args:
        target_path (Path): The file path to the target media container.

    Raises:
        FileNotFoundError: If the designated file does not exist on disk.
    """
    if not target_path.exists() or not target_path.is_file():
        raise FileNotFoundError(f"Simulation file target not found: {target_path}")


def _safe_thread_wrapper(target_func: Callable[..., Any], **kwargs: Any) -> None:
    """
    Executes a background node target with explicit exception trapping and stack trace dumping.

    Args:
        target_func (Callable[..., Any]): The microservice function to run inside the thread.
        kwargs (Any): Parameter map injected into the target function.
    """
    try:
        target_func(**kwargs)
    except Exception as exc:
        print(f"\n[FATAL THREAD CRASH] {target_func.__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise


def _ignite_pipeline_nodes() -> None:
    """
    Initializes configuration matrices and allocates all background edge nodes to the global state.
    """
    asr_to_semantic_queue = queue.Queue()
    nexus_semantic_in: queue.Queue = queue.Queue()
    nexus_acoustic_in: queue.Queue = queue.Queue()
    nexus_facial_in: queue.Queue = queue.Queue()
    nexus_skeletal_in: queue.Queue = queue.Queue()

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

    print("-> Booting Transcription Node (NeMo Parakeet)...")
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

    print("-> Booting Semantic Node...")
    spawn_pipeline_node(
        state=global_pipeline_state,
        target=_safe_thread_wrapper,
        kwargs={
            "target_func": run_semantic_node,
            "input_queue": asr_to_semantic_queue,             
            "output_queue": nexus_semantic_in                 
        }
    )

    print("-> Booting Acoustic Node...")
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

    # (Removed the duplicate Semantic Node boot that was causing race conditions)

    print("-> Booting Skeletal Node...")
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

    print("-> Booting Facial Node...")
    spawn_pipeline_node(
        state=global_pipeline_state,
        target=_safe_thread_wrapper,
        kwargs={
            "target_func": run_facial_node, # <--- Direct reference
            "input_queue": global_pipeline_state.q_visual_facial,
            "output_queue": nexus_facial_in,
            "config": FacialConfig()
        }
    )
    time.sleep(1.0)

    print("-> Booting Fusion & Judge Nodes...")
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
        max_tokens=5000,
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
            "timeout_sec": 30.0
        }
    )

def _poll_evaluation_results() -> None:
    """
    Polls the terminal egress queue, cleans the response format, and exits.
    """
    while True:
        try:
            # Increased timeout to 60s to accommodate NeMo Parakeet's cold-boot
            payload: Any = global_pipeline_state.q_judge_out.get()
            
            if isinstance(payload, OrchestratorTerminalSentinel):
                print("\n[SYSTEM] Reached End of Stream.")
                break
                
            if isinstance(payload, CoachingFeedbackPayload):
                print(f"\n==========================================")
                print(f"🎯 AI COACHING EVALUATION (Segment {payload.segment_id})")
                print(f"==========================================")
                
                raw_text = payload.feedback_text
                if isinstance(raw_text, list):
                    clean_text = "".join([chunk.get("text", "") for chunk in raw_text if isinstance(chunk, dict)])
                    print(clean_text)
                else:
                    print(raw_text)
                    
                print(f"==========================================\n")
            break
                
        except queue.Empty:
            print("\n[SYSTEM] Timeout waiting for judge evaluation.")
            break


def main() -> None:
    """
    Master execution entry point orchestrating the local deterministic simulation pipeline.
    """
    target_video: Path = Path("app/src/simulation/video/test_2.mp4")
    
    try:
        _verify_video_target(target_video)

        print("[SYSTEM] Igniting pipeline nodes...")
        _ignite_pipeline_nodes()

        print("[SYSTEM] Warming up neural matrices... (Waiting 70s)")
        time.sleep(70.0) 

        sim_config = IngestionConfig(
            mode=IngestionMode.SIMULATOR,
            simulator_file_path=target_video,
            video_fps=50.0,
            audio_sample_rate=16000
        )

        print(f"[SYSTEM] Pumping {target_video.name} into the matrix...")
        execute_simulator_ingestion(
            config=sim_config,
            transcription_queue=global_pipeline_state.q_transcription,
            window_queue=global_pipeline_state.q_window
        )

        print("[SYSTEM] Awaiting AI Judge Evaluation...\n")
        _poll_evaluation_results()

    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Manual override detected.")
    except Exception as exc:
        print(f"\n[ERROR] Catastrophic pipeline failure: {exc}")
    finally:
        print("[SYSTEM] Executing graceful teardown cascade...")
        shutdown_matrix(global_pipeline_state)
        print("[SYSTEM] Teardown complete. Exiting.")


if __name__ == "__main__":
    main()