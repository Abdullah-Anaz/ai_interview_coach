import queue
from pathlib import Path
from queue import Queue
from typing import Any, List

import numpy as np
import pytest
import torch

from app.src.orchestration.config import OrchestratorConfig
from app.src.orchestration.orchestrator import (
    OrchestratorTerminalSentinel,
    route_media_stream,
    run_orchestrator_node,
)
from app.src.orchestration.types import AudioChunk, SynchronizedWindowSlice


@pytest.fixture
def hardware_check() -> None:
    """Validates GPU/CPU tensor accessibility to mirror runtime orchestration constraints."""
    _ = torch.cuda.is_available()


@pytest.fixture
def blank_audio_chunk() -> AudioChunk:
    """Synthesizes a minimal acoustic vector payload."""
    return AudioChunk(
        sample_rate=16000,
        timestamp_sec=1.5,
        data=np.zeros((1600,), dtype=np.float32)
    )


@pytest.fixture
def blank_window_slice(blank_audio_chunk: AudioChunk) -> SynchronizedWindowSlice:
    """Synthesizes a minimal synchronized temporal extraction window."""
    return SynchronizedWindowSlice(
        window_id=1,
        start_time_sec=1.0,
        end_time_sec=3.0,
        audio=blank_audio_chunk,
        frames=tuple(),
        is_terminal=False
    )


# =====================================================================
# ROUTE MEDIA STREAM: EXPECTED BEHAVIOUR & EDGE CASES
# =====================================================================

def test_route_media_stream_expected_behaviour_for_standard_media() -> None:
    """Validates dual-queue routing for standard media using real test video."""
    media_path: Path = Path("app/src/tests/data/standard_interview.mp4")
    config: OrchestratorConfig = OrchestratorConfig(
        window_duration_sec=1.0,
        window_stride_sec=0.5
    )
    transcription_queue: Queue = Queue()
    window_queue: Queue = Queue()

    route_media_stream(
        source=media_path,
        config=config,
        transcription_queue=transcription_queue,
        window_queue=window_queue
    )

    asr_items: List[Any] = []
    while not transcription_queue.empty():
        asr_items.append(transcription_queue.get_nowait())

    assert len(asr_items) > 1
    asr_sentinel = asr_items.pop()
    assert isinstance(asr_sentinel, OrchestratorTerminalSentinel)
    assert asr_sentinel.error is None
    assert all(isinstance(item, AudioChunk) for item in asr_items)

    window_items: List[Any] = []
    while not window_queue.empty():
        window_items.append(window_queue.get_nowait())

    assert len(window_items) > 1
    window_sentinel = window_items.pop()
    assert isinstance(window_sentinel, OrchestratorTerminalSentinel)
    assert window_sentinel.error is None
    assert all(isinstance(item, SynchronizedWindowSlice) for item in window_items)


def test_route_media_stream_expected_behaviour_for_short_media() -> None:
    """Validates boundary slicing for inputs shorter than a single window duration."""
    media_path: Path = Path("app/src/tests/data/fractional_second_video.mp4")
    config: OrchestratorConfig = OrchestratorConfig(
        window_duration_sec=2.0,
        window_stride_sec=0.5
    )
    transcription_queue: Queue = Queue()
    window_queue: Queue = Queue()

    route_media_stream(
        source=media_path,
        config=config,
        transcription_queue=transcription_queue,
        window_queue=window_queue
    )

    window_items: List[Any] = []
    while not window_queue.empty():
        window_items.append(window_queue.get_nowait())

    assert len(window_items) == 2
    terminal_slice = window_items[0]
    assert isinstance(terminal_slice, SynchronizedWindowSlice)
    assert terminal_slice.is_terminal is True


def test_route_media_stream_expected_behaviour_for_audio_only_media() -> None:
    """Validates pipeline handles audio-only streams by emitting empty frame tuples."""
    media_path: Path = Path("app/src/tests/data/audio_only.wav")
    config: OrchestratorConfig = OrchestratorConfig(
        window_duration_sec=1.0,
        window_stride_sec=0.5
    )
    transcription_queue: Queue = Queue()
    window_queue: Queue = Queue()

    route_media_stream(
        source=media_path,
        config=config,
        transcription_queue=transcription_queue,
        window_queue=window_queue
    )

    window_items: List[Any] = []
    while not window_queue.empty():
        window_items.append(window_queue.get_nowait())

    assert len(window_items) > 1
    slice_packet = window_items[0]
    assert isinstance(slice_packet, SynchronizedWindowSlice)
    assert len(slice_packet.frames) == 0


def test_route_media_stream_raises_for_invalid_transcription_queue() -> None:
    """Validates type safety constraints on the transcription queue."""
    media_path: Path = Path("app/src/tests/data/standard_interview.mp4")
    config: OrchestratorConfig = OrchestratorConfig()

    with pytest.raises(TypeError) as exc_info:
        route_media_stream(
            source=media_path,
            config=config,
            transcription_queue=[],  # type: ignore
            window_queue=Queue()
        )
    assert "transcription_queue must be a queue.Queue" in str(exc_info.value)


def test_route_media_stream_raises_for_invalid_window_queue() -> None:
    """Validates type safety constraints on the window queue."""
    media_path: Path = Path("app/src/tests/data/standard_interview.mp4")
    config: OrchestratorConfig = OrchestratorConfig()

    with pytest.raises(TypeError) as exc_info:
        route_media_stream(
            source=media_path,
            config=config,
            transcription_queue=Queue(),
            window_queue=None  # type: ignore
        )
    assert "window_queue must be a queue.Queue" in str(exc_info.value)


def test_route_media_stream_raises_for_invalid_config() -> None:
    """Validates type safety constraints on the orchestrator configuration."""
    media_path: Path = Path("app/src/tests/data/standard_interview.mp4")

    with pytest.raises(TypeError) as exc_info:
        route_media_stream(
            source=media_path,
            config={"window_duration_sec": 2.0},  # type: ignore
            transcription_queue=Queue(),
            window_queue=Queue()
        )
    assert "config must be an OrchestratorConfig" in str(exc_info.value)


def test_route_media_stream_raises_for_transcription_queue_saturation() -> None:
    """Validates robust failure handling when the downstream ASR model blocks."""
    media_path: Path = Path("app/src/tests/data/standard_interview.mp4")
    config: OrchestratorConfig = OrchestratorConfig(queue_timeout_sec=0.01)
    
    transcription_queue: Queue = Queue(maxsize=1)
    window_queue: Queue = Queue()

    with pytest.raises(RuntimeError) as exc_info:
        route_media_stream(
            source=media_path,
            config=config,
            transcription_queue=transcription_queue,
            window_queue=window_queue
        )
    
    assert "Transcription queue saturated" in str(exc_info.value)


def test_route_media_stream_raises_for_window_queue_saturation() -> None:
    """Validates robust failure handling when downstream visual/acoustic models block."""
    media_path: Path = Path("app/src/tests/data/standard_interview.mp4")
    config: OrchestratorConfig = OrchestratorConfig(queue_timeout_sec=0.01)
    
    transcription_queue: Queue = Queue()
    window_queue: Queue = Queue(maxsize=1)

    with pytest.raises(RuntimeError) as exc_info:
        route_media_stream(
            source=media_path,
            config=config,
            transcription_queue=transcription_queue,
            window_queue=window_queue
        )
    
    assert "Downstream window queue saturated" in str(exc_info.value)


def test_route_media_stream_raises_for_nonexistent_file() -> None:
    """Validates failure propagation and error sentinel injection across both queues."""
    media_path: Path = Path("app/src/tests/data/does_not_exist.mp4")
    config: OrchestratorConfig = OrchestratorConfig()
    transcription_queue: Queue = Queue()
    window_queue: Queue = Queue()

    with pytest.raises(RuntimeError) as exc_info:
        route_media_stream(
            source=media_path,
            config=config,
            transcription_queue=transcription_queue,
            window_queue=window_queue
        )

    assert "Orchestration failure" in str(exc_info.value)

    sentinel_asr = transcription_queue.get_nowait()
    sentinel_win = window_queue.get_nowait()
    assert isinstance(sentinel_asr, OrchestratorTerminalSentinel)
    assert isinstance(sentinel_win, OrchestratorTerminalSentinel)
    assert sentinel_asr.error is not None
    assert sentinel_win.error is not None


def test_route_media_stream_raises_for_corrupt_media() -> None:
    """Validates graceful failure and sentinel dispatch when given non-media formats."""
    media_path: Path = Path("app/src/tests/data/invalid_format.txt")
    config: OrchestratorConfig = OrchestratorConfig()
    transcription_queue: Queue = Queue()
    window_queue: Queue = Queue()

    with pytest.raises(RuntimeError) as exc_info:
        route_media_stream(
            source=media_path,
            config=config,
            transcription_queue=transcription_queue,
            window_queue=window_queue
        )

    assert "Orchestration failure" in str(exc_info.value)
    
    assert not transcription_queue.empty()
    assert not window_queue.empty()


# =====================================================================
# MULTIPLEXER DAEMON: EXPECTED BEHAVIOUR & EDGE CASES
# =====================================================================

def test_run_orchestrator_node_expected_behaviour_for_multiplexing(
    hardware_check: None,
    blank_audio_chunk: AudioChunk,
    blank_window_slice: SynchronizedWindowSlice
) -> None:
    """Verifies that the multiplexer natively reads ingress packets and accurately duplicates them downstream."""
    q_t_in = Queue()
    q_w_in = Queue()
    q_ac_out = Queue()
    q_sem_out = Queue()
    q_vf_out = Queue()
    q_vs_out = Queue()

    # The continuous transcription chunk is silently drained, so q_sem_out receives the window slice instead
    q_t_in.put(blank_audio_chunk)
    q_t_in.put(OrchestratorTerminalSentinel())
    
    q_w_in.put(blank_window_slice)
    q_w_in.put(OrchestratorTerminalSentinel())

    run_orchestrator_node(
        q_transcription_in=q_t_in,
        q_window_in=q_w_in,
        q_acoustic_out=q_ac_out,
        q_semantic_out=q_sem_out,
        q_visual_facial_out=q_vf_out,
        q_visual_skeletal_out=q_vs_out,
        timeout_sec=0.1
    )

    # Verify window broadcasted to all four compute edges
    assert not q_sem_out.empty()
    assert not q_ac_out.empty()
    assert not q_vf_out.empty()
    assert not q_vs_out.empty()

    sem_payload: Any = q_sem_out.get_nowait()
    ac_payload: Any = q_ac_out.get_nowait()
    vf_payload: Any = q_vf_out.get_nowait()
    vs_payload: Any = q_vs_out.get_nowait()

    assert isinstance(sem_payload, SynchronizedWindowSlice)
    assert isinstance(ac_payload, SynchronizedWindowSlice)
    assert isinstance(vf_payload, SynchronizedWindowSlice)
    assert isinstance(vs_payload, SynchronizedWindowSlice)
    
    assert sem_payload.window_id == 1
    assert ac_payload.window_id == 1
    assert vf_payload.window_id == 1
    assert vs_payload.window_id == 1


def test_run_orchestrator_node_expected_behaviour_for_sentinel_cascade(hardware_check: None) -> None:
    """Verifies that the node immediately halts inner loops and broadcasts sentinels across the matrix."""
    q_t_in = Queue()
    q_w_in = Queue()
    q_ac_out = Queue()
    q_sem_out = Queue()
    q_vf_out = Queue()
    q_vs_out = Queue()

    err = RuntimeError("Simulated upstream crash")
    q_t_in.put(OrchestratorTerminalSentinel(error=err))
    q_w_in.put(OrchestratorTerminalSentinel(error=err))

    run_orchestrator_node(q_t_in, q_w_in, q_ac_out, q_sem_out, q_vf_out, q_vs_out)

    assert getattr(q_sem_out.get_nowait(), "error") == err
    assert getattr(q_ac_out.get_nowait(), "error") == err
    assert getattr(q_vf_out.get_nowait(), "error") == err
    assert getattr(q_vs_out.get_nowait(), "error") == err


def test_run_orchestrator_node_raises_for_invalid_queue_type() -> None:
    """Verifies structural rejection if raw lists are passed to the thread router."""
    with pytest.raises(TypeError, match="All parameters must be queue.Queue"):
        run_orchestrator_node(
            q_transcription_in=[],  # type: ignore
            q_window_in=Queue(),
            q_acoustic_out=Queue(),
            q_semantic_out=Queue(),
            q_visual_facial_out=Queue(),
            q_visual_skeletal_out=Queue()
        )