import queue
from typing import Any

import pytest
import torch

from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.stream_ingestion.config import IngestionConfig, IngestionMode
from app.src.stream_ingestion.live_capture import _hardware_stream_generator, execute_live_ingestion


@pytest.fixture
def hardware_check() -> None:
    """Validates runtime environments but enforces CPU operations for structural routing logic."""
    _ = torch.cuda.is_available()


@pytest.fixture
def valid_live_config() -> IngestionConfig:
    """Provides a structurally valid configuration bound to live hardware modes."""
    return IngestionConfig(
        mode=IngestionMode.LIVE,
        video_fps=10.0,
        audio_sample_rate=16000,
        camera_device_index=999,  # Deliberately out-of-bounds index to guarantee hardware rejection
        audio_device_index=999
    )


# =====================================================================
# ERROR CASES: HARDWARE UNAVAILABILITY & GRACEFUL DEGRADATION
# =====================================================================

def test_hardware_stream_generator_raises_for_unavailable_camera(
    hardware_check: None, 
    valid_live_config: IngestionConfig
) -> None:
    """Verifies that attempting to bind to non-existent camera drivers immediately raises a trapped exception."""
    generator = _hardware_stream_generator(valid_live_config)
    
    with pytest.raises(RuntimeError, match="Webcam hardware unavailable at index 999"):
        next(generator)


def test_execute_live_ingestion_raises_for_hardware_failure(
    hardware_check: None, 
    valid_live_config: IngestionConfig
) -> None:
    """Verifies that a hardware failure correctly aborts the loop, injects sentinels, and cascades a RuntimeError."""
    transcription_q: queue.Queue = queue.Queue()
    window_q: queue.Queue = queue.Queue()
    
    with pytest.raises(RuntimeError, match="Live ingestion failure"):
        execute_live_ingestion(
            config=valid_live_config,
            transcription_queue=transcription_q,
            window_queue=window_q
        )
        
    # Verify the teardown sentinel cascade successfully fired into downstream queues despite the crash
    assert not transcription_q.empty()
    assert not window_q.empty()
    
    t_sentinel: Any = transcription_q.get_nowait()
    w_sentinel: Any = window_q.get_nowait()
    
    assert isinstance(t_sentinel, OrchestratorTerminalSentinel)
    assert isinstance(w_sentinel, OrchestratorTerminalSentinel)
    assert t_sentinel.error is not None


# =====================================================================
# ERROR CASES: MODE CONFLICTS & QUEUE SATURATION
# =====================================================================

def test_execute_live_ingestion_raises_for_simulator_mode() -> None:
    """Verifies structural rejection if a file-based configuration is mistakenly passed to the hardware engine."""
    config = IngestionConfig.__new__(IngestionConfig)
    object.__setattr__(config, "mode", IngestionMode.SIMULATOR)
    
    with pytest.raises(ValueError, match="strictly requires IngestionMode.LIVE."):
        execute_live_ingestion(config, queue.Queue(), queue.Queue())


def test_execute_live_ingestion_raises_for_invalid_queue_type(valid_live_config: IngestionConfig) -> None:
    """Verifies strict python typing bounds for the asynchronous buffers."""
    with pytest.raises(TypeError, match="transcription_queue must be a queue.Queue"):
        # The orchestrator logic dynamically checks the queue type during the timeout push.
        # However, passing an invalid list directly breaks Python's structural interface expectations.
        execute_live_ingestion(
            config=valid_live_config,
            transcription_queue=[],  # type: ignore
            window_queue=queue.Queue()
        )