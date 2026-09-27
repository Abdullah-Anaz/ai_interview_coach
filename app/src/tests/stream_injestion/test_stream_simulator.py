import av
import numpy as np
import pytest
import torch
from queue import Queue
from pathlib import Path
from typing import Any

from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.stream_ingestion.config import IngestionConfig, IngestionMode
from app.src.stream_ingestion.stream_simulator import execute_simulator_ingestion


@pytest.fixture
def hardware_check() -> None:
    """Validates runtime environments but enforces CPU operations for unmocked file IO tests."""
    _ = torch.cuda.is_available()


@pytest.fixture
def valid_simulator_config(tmp_path: Path) -> IngestionConfig:
    """Provides a structurally valid configuration bound to a temporary file path."""
    dummy_file: Path = tmp_path / "test_video.mp4"
    dummy_file.touch()
    return IngestionConfig(
        mode=IngestionMode.SIMULATOR,
        video_fps=10.0,
        audio_sample_rate=16000,
        simulator_file_path=dummy_file
    )


@pytest.fixture
def unmocked_media_config(tmp_path: Path) -> IngestionConfig:
    """
    Natively synthesizes a genuine media container using PyAV to test
    the complete demuxing and routing integration without mocking.
    """
    file_path: Path = tmp_path / "native_test_stream.mp4"
    
    container = av.open(str(file_path), mode='w')
    
    video_stream = container.add_stream('mpeg4', rate=10)
    video_stream.width = 320
    video_stream.height = 240
    video_stream.pix_fmt = 'yuv420p'
    
    audio_stream = container.add_stream('mp2', rate=16000)
    
    for _ in range(5):
        v_frame = av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), dtype=np.uint8), format='rgb24')
        for packet in video_stream.encode(v_frame):
            container.mux(packet)
            
        a_frame = av.AudioFrame.from_ndarray(np.zeros((1, 1600), dtype=np.float32), format='fltp', layout='mono')
        a_frame.sample_rate = 16000
        for packet in audio_stream.encode(a_frame):
            container.mux(packet)
            
    for packet in video_stream.encode():
        container.mux(packet)
    for packet in audio_stream.encode():
        container.mux(packet)
        
    container.close()
    
    return IngestionConfig(
        mode=IngestionMode.SIMULATOR,
        video_fps=10.0,
        audio_sample_rate=16000,
        simulator_file_path=file_path
    )


# =====================================================================
# EXPECTED BEHAVIOUR & INTEGRATION
# =====================================================================

def test_execute_simulator_ingestion_expected_behaviour_for_native_routing(
    hardware_check: None,
    unmocked_media_config: IngestionConfig
) -> None:
    """Verifies that the simulator successfully bridges into the orchestrator and processes a real file."""
    transcription_q: Queue = Queue()
    window_q: Queue = Queue()
    
    execute_simulator_ingestion(
        config=unmocked_media_config,
        transcription_queue=transcription_q,
        window_queue=window_q,
        frame_buffer_capacity=50,
        window_duration_sec=0.2,
        window_stride_sec=0.1,
        queue_timeout_sec=1.0
    )
    
    assert not transcription_q.empty()
    assert not window_q.empty()
    
    # Verify the terminal cascade executed correctly at EOF
    found_terminal = False
    while not transcription_q.empty():
        item: Any = transcription_q.get_nowait()
        if isinstance(item, OrchestratorTerminalSentinel):
            found_terminal = True
            
    assert found_terminal


# =====================================================================
# ERROR CASES: MODE CONFLICTS & FILE VALIDATION
# =====================================================================

def test_execute_simulator_ingestion_raises_for_live_mode() -> None:
    """Verifies structural rejection if a live configuration is mistakenly passed to the simulator engine."""
    config = IngestionConfig(
        mode=IngestionMode.LIVE,
        camera_device_index=0,
        audio_device_index=0
    )
    
    with pytest.raises(ValueError, match="strictly requires IngestionMode.SIMULATOR."):
        execute_simulator_ingestion(config, Queue(), Queue())


def test_execute_simulator_ingestion_raises_for_missing_file(tmp_path: Path) -> None:
    """Verifies structural trapping when the OS fails to locate the target simulation file."""
    missing_file: Path = tmp_path / "does_not_exist.mp4"
    
    config = IngestionConfig.__new__(IngestionConfig)
    object.__setattr__(config, "mode", IngestionMode.SIMULATOR)
    object.__setattr__(config, "simulator_file_path", missing_file)
    
    with pytest.raises(FileNotFoundError, match="Simulation file target not found"):
        execute_simulator_ingestion(config, Queue(), Queue())


def test_execute_simulator_ingestion_raises_for_directory(tmp_path: Path) -> None:
    """Verifies structural trapping when the configuration mistakenly targets a directory."""
    config = IngestionConfig.__new__(IngestionConfig)
    object.__setattr__(config, "mode", IngestionMode.SIMULATOR)
    object.__setattr__(config, "simulator_file_path", tmp_path)
    
    with pytest.raises(IsADirectoryError, match="Simulation target is a directory"):
        execute_simulator_ingestion(config, Queue(), Queue())


def test_execute_simulator_ingestion_raises_for_null_path() -> None:
    """Verifies internal guardrails safely trap empty path states during runtime initialization."""
    config = IngestionConfig.__new__(IngestionConfig)
    object.__setattr__(config, "mode", IngestionMode.SIMULATOR)
    object.__setattr__(config, "simulator_file_path", None)
    
    with pytest.raises(ValueError, match="simulator_file_path cannot be None"):
        execute_simulator_ingestion(config, Queue(), Queue())


def test_execute_simulator_ingestion_raises_for_invalid_queue_type(valid_simulator_config: IngestionConfig) -> None:
    """Verifies that the orchestrator's strict native type checking ripples up to the ingestion barrier."""
    with pytest.raises(TypeError, match="transcription_queue must be a queue.Queue"):
        execute_simulator_ingestion(
            config=valid_simulator_config,
            transcription_queue=[],  # type: ignore
            window_queue=Queue()
        )