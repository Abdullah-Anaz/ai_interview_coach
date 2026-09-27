from pathlib import Path
from queue import Queue

from app.src.orchestration.config import OrchestratorConfig
from app.src.orchestration.orchestrator import route_media_stream
from app.src.stream_ingestion.config import IngestionConfig, IngestionMode


def execute_simulator_ingestion(
    config: IngestionConfig,
    transcription_queue: Queue,
    window_queue: Queue,
    frame_buffer_capacity: int = 500,
    window_duration_sec: float = 2.0,
    window_stride_sec: float = 0.5,
    queue_timeout_sec: float = 5.0
) -> None:
    """
    Validates a pre-recorded simulation target and routes it through the orchestration pipeline.
    Leverages natural downstream queue saturation to enforce processing backpressure.

    Args:
        config (IngestionConfig): Validated configuration bound to SIMULATOR mode.
        transcription_queue (Queue): Egress route for normalized audio chunks.
        window_queue (Queue): Egress route for synchronized multimodal temporal slices.
        frame_buffer_capacity (int): Maximum frames retained in the visual ring buffer.
        window_duration_sec (float): Total length of each temporal window slice.
        window_stride_sec (float): Temporal step size between consecutive windows.
        queue_timeout_sec (float): Maximum block time when pushing to a full downstream queue.

    Raises:
        ValueError: If the configuration mode is strictly LIVE or lacks a file target.
        FileNotFoundError: If the designated simulator file does not exist.
        IsADirectoryError: If the designated simulator file path points to a directory.
    """
    if config.mode != IngestionMode.SIMULATOR:
        raise ValueError("execute_simulator_ingestion strictly requires IngestionMode.SIMULATOR.")

    if config.simulator_file_path is None:
        raise ValueError("simulator_file_path cannot be None when in SIMULATOR mode.")

    target_path: Path = config.simulator_file_path

    if not target_path.exists():
        raise FileNotFoundError(f"Simulation file target not found: {target_path}")

    if not target_path.is_file():
        raise IsADirectoryError(f"Simulation target is a directory, not a file: {target_path}")

    orch_config = OrchestratorConfig(
        audio_sample_rate=config.audio_sample_rate,
        video_fps=config.video_fps,
        window_duration_sec=window_duration_sec,
        window_stride_sec=window_stride_sec,
        queue_timeout_sec=queue_timeout_sec
    )

    route_media_stream(
        source=target_path,
        config=orch_config,
        transcription_queue=transcription_queue,
        window_queue=window_queue,
        frame_buffer_capacity=frame_buffer_capacity
    )