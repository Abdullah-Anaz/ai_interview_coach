import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import aiofiles
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.src.api.dependencies import global_pipeline_state
from app.src.api.lifecycle import manage_pipeline_lifecycle
from app.src.apex_judge.types import CoachingFeedbackPayload
from app.src.orchestration.config import OrchestratorConfig
from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel, route_media_stream

logger: logging.Logger = logging.getLogger(__name__)

app: FastAPI = FastAPI(lifespan=manage_pipeline_lifecycle)


async def _transmit_feedback_loop(websocket: WebSocket, pipeline_finished: asyncio.Event) -> None:
    """
    Asynchronously polls the egress queue and pushes serialized JSON feedback to the client.

    Args:
        websocket (WebSocket): The active client connection.
        pipeline_finished (asyncio.Event): Synchronization flag to coordinate termination.
    """
    try:
        while not pipeline_finished.is_set():
            await asyncio.sleep(0.1)
            
            if global_pipeline_state.q_judge_out.empty():
                continue

            payload: Any = global_pipeline_state.q_judge_out.get_nowait()
            
            if isinstance(payload, OrchestratorTerminalSentinel):
                logger.info("[TRANSMIT] Received terminal sentinel on q_judge_out. Stream complete.")
                pipeline_finished.set()
                break
                
            if isinstance(payload, CoachingFeedbackPayload):
                response: Dict[str, Any] = {
                    "segment_id": payload.segment_id,
                    "feedback": payload.feedback_text
                }
                logger.info("[TRANSMIT] Sending coaching feedback segment %s to client.", payload.segment_id)
                await websocket.send_json(response)

    except Exception as exc:
        logger.error("[TRANSMIT ERROR] Error during feedback transmission: %s", exc)
        pipeline_finished.set()


async def _receive_media_loop_live(
    websocket: WebSocket, 
    pipeline_finished: asyncio.Event, 
    fifo_path: Path
) -> None:
    """
    Asynchronously reads binary network blobs and streams them directly into a named pipe.

    Args:
        websocket (WebSocket): The active client connection.
        pipeline_finished (asyncio.Event): Synchronization flag to coordinate termination.
        fifo_path (Path): The file path to the UNIX named pipe.
    """
    try:
        async with aiofiles.open(fifo_path, mode="wb") as fifo_file:
            while not pipeline_finished.is_set():
                try:
                    message: Dict[str, Any] = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                msg_type: Any = message.get("type")
                
                if msg_type == "websocket.receive":
                    if "text" in message:
                        try:
                            data: Dict[str, Any] = json.loads(message["text"])
                            action: Any = data.get("action")
                            legacy_type: Any = data.get("type")
                            
                            if legacy_type == "EOS" or action == "TERMINATE_STREAM":
                                logger.info("Received End-Of-Stream sentinel. Closing live stream.")
                                break
                        except json.JSONDecodeError:
                            pass
                        except Exception as route_exc:
                            logger.error("Error processing stream termination: %s", route_exc)
                        continue

                    binary_payload: Any = message.get("bytes")
                    if binary_payload and isinstance(binary_payload, bytes):
                        await fifo_file.write(binary_payload)
                        await fifo_file.flush()

                elif msg_type == "websocket.disconnect":
                    logger.info("Client disconnected gracefully.")
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client.")
    except Exception as exc:
        logger.error("WebSocket ingress failure: %s", exc, exc_info=True)
    finally:
        if not pipeline_finished.is_set():
            global_pipeline_state.q_transcription.put(OrchestratorTerminalSentinel())
            global_pipeline_state.q_window.put(OrchestratorTerminalSentinel())
            pipeline_finished.set()


@app.websocket("/ws/interview")
async def interview_stream_endpoint(websocket: WebSocket) -> None:
    """
    Establishes a full-duplex WebSocket connection to ingest live media and emit coaching feedback.

    Args:
        websocket (WebSocket): The active client connection protocol.
    """
    await websocket.accept()
    while not global_pipeline_state.q_judge_out.empty():
        global_pipeline_state.q_judge_out.get_nowait()
        
    logger.info(">>> [DEBUG] LIVE STREAMING SERVER ACTIVE <<<")

    pipeline_finished: asyncio.Event = asyncio.Event()
    temp_dir: str = tempfile.mkdtemp()
    fifo_path: Path = Path(temp_dir) / "live_stream.webm"
    
    try:
        os.mkfifo(str(fifo_path))
        
        orch_config: OrchestratorConfig = OrchestratorConfig(
            audio_sample_rate=16000,
            video_fps=50.0,
            window_duration_sec=2.0,
            window_stride_sec=0.5,
            queue_timeout_sec=5.0
        )

        routing_task: asyncio.Task = asyncio.create_task(
            asyncio.to_thread(
                route_media_stream,
                source=fifo_path,
                config=orch_config,
                transcription_queue=global_pipeline_state.q_transcription,
                window_queue=global_pipeline_state.q_window,
                frame_buffer_capacity=500
            )
        )

        await asyncio.gather(
            _transmit_feedback_loop(websocket, pipeline_finished),
            _receive_media_loop_live(websocket, pipeline_finished, fifo_path),
            routing_task
        )

    except Exception as exc:
        logger.error("Critical failure in WebSocket endpoint setup: %s", exc, exc_info=True)
    finally:
        if fifo_path.exists():
            try:
                fifo_path.unlink()
            except Exception as cleanup_exc:
                logger.error("Failed to remove FIFO path: %s", cleanup_exc)
        if Path(temp_dir).exists():
            try:
                Path(temp_dir).rmdir()
            except Exception as dir_cleanup_exc:
                logger.error("Failed to remove temp directory: %s", dir_cleanup_exc)