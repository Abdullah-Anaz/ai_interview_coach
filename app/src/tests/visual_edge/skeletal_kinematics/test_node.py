import json
import threading
from pathlib import Path
from queue import Queue
from typing import Any, Generator, List, Tuple

import numpy as np
import pytest
import torch

from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.orchestration.types import AudioChunk, SynchronizedWindowSlice, VideoFrame
from app.src.visual_edge.skeletal_kinematics.config import SkeletalConfig
from app.src.visual_edge.skeletal_kinematics.node import (
    _aggregate_payloads,
    _process_segment,
    run_skeletal_node,
)
from app.src.visual_edge.skeletal_kinematics.omni import HumanOmni
from app.src.visual_edge.skeletal_kinematics.types import SkeletalDescriptorPayload


@pytest.fixture(scope="module")
def active_config() -> Generator[SkeletalConfig, None, None]:
    """Provides a validated configuration matrix tied to the active hardware."""
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    calib_file: Path = Path("app/models/visual/test_node_calibration.json")
    weights_file: Path = Path("app/models/visual/test_human_omni_node.pth")
    
    calib_file.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_file, "w", encoding="utf-8") as file:
        json.dump({
            "perceptual_jitter": {"very_low_threshold": 0.1, "low_threshold": 0.2, "high_threshold": 0.3, "very_high_threshold": 0.4},
            "temporal_smoothness": {"very_low_threshold": 0.1, "low_threshold": 0.2, "high_threshold": 0.3, "very_high_threshold": 0.4}
        }, file)

    model = HumanOmni()
    torch.save(model.state_dict(), weights_file)

    config = SkeletalConfig(
        calibration_path=calib_file, 
        weights_path=weights_file, 
        device=device, 
        temporal_window_size=50
    )
    yield config

    if calib_file.exists(): 
        calib_file.unlink()
    if weights_file.exists(): 
        weights_file.unlink()


@pytest.fixture
def dummy_audio() -> AudioChunk:
    """Generates a neutral acoustic payload to satisfy multimodal window constraints."""
    return AudioChunk(
        sample_rate=16000, 
        timestamp_sec=0.0, 
        data=np.zeros(1600, dtype=np.float32)
    )


@pytest.fixture
def multi_frame_tuple() -> Tuple[VideoFrame, ...]:
    """Generates a mathematically valid sequence of spatial RGB numpy arrays for HumanOmni."""
    return (
        VideoFrame(frame_index=0, timestamp_sec=0.0, data=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
        VideoFrame(frame_index=1, timestamp_sec=0.2, data=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    )


@pytest.fixture
def single_frame_tuple() -> Tuple[VideoFrame, ...]:
    """Generates an isolated spatial RGB frame to test fallback boundary handling."""
    return (
        VideoFrame(frame_index=0, timestamp_sec=1.0, data=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
    )


@pytest.fixture
def anomalous_frame_tuple() -> Tuple[VideoFrame, ...]:
    """Generates an anomalous matrix shape to trigger upstream tensor coercion failures."""
    invalid_frame = VideoFrame.__new__(VideoFrame)
    object.__setattr__(invalid_frame, "frame_index", 0)
    object.__setattr__(invalid_frame, "timestamp_sec", 0.0)
    object.__setattr__(invalid_frame, "data", np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8))
    return (invalid_frame,)


def test_process_segment_expected_behaviour_for_multi_frame_window(
    active_config: SkeletalConfig,
    dummy_audio: AudioChunk,
    multi_frame_tuple: Tuple[VideoFrame, ...]
) -> None:
    """Tests the complete inference pass mapping raw Numpy arrays through the One Euro filtered text generator."""
    device = torch.device(active_config.device)
    model = HumanOmni().to(device)
    model.eval()
    
    window = SynchronizedWindowSlice(
        window_id=14,
        start_time_sec=10.0,
        end_time_sec=11.0,
        audio=dummy_audio,
        frames=multi_frame_tuple,
        is_terminal=False
    )

    payload: SkeletalDescriptorPayload = _process_segment(
        item=window, 
        model=model, 
        config=active_config, 
        device=device
    )

    assert payload.segment_id == 14
    assert payload.start_time_sec == 10.0
    assert payload.end_time_sec == 11.0
    assert isinstance(payload.behavioral_descriptor, str)
    assert len(payload.behavioral_descriptor) > 0
    assert isinstance(payload.perceptual_jitter, float)
    assert isinstance(payload.temporal_smoothness, float)


def test_process_segment_expected_behaviour_for_isolated_frame(
    active_config: SkeletalConfig,
    dummy_audio: AudioChunk,
    single_frame_tuple: Tuple[VideoFrame, ...]
) -> None:
    """Verifies the pipeline processes isolated edge case frames without sequence-length mathematical errors."""
    device = torch.device(active_config.device)
    model = HumanOmni().to(device)
    model.eval()
    
    window = SynchronizedWindowSlice(
        window_id=15,
        start_time_sec=1.0,
        end_time_sec=1.5,
        audio=dummy_audio,
        frames=single_frame_tuple,
        is_terminal=True
    )

    payload: SkeletalDescriptorPayload = _process_segment(
        item=window, 
        model=model, 
        config=active_config, 
        device=device
    )

    assert payload.segment_id == 15
    assert isinstance(payload.perceptual_jitter, float)


def test_process_segment_raises_for_empty_frames_tuple(
    active_config: SkeletalConfig,
    dummy_audio: AudioChunk
) -> None:
    """Ensures downstream pipeline intercepts and safely bubbles up empty matrix extraction failures."""
    device = torch.device(active_config.device)
    model = HumanOmni().to(device)
    
    window = SynchronizedWindowSlice(
        window_id=16,
        start_time_sec=0.0,
        end_time_sec=1.0,
        audio=dummy_audio,
        frames=(),
        is_terminal=False
    )

    with pytest.raises(RuntimeError, match="Failed to process skeletal segment 16"):
        _process_segment(item=window, model=model, config=active_config, device=device)


def test_process_segment_raises_for_invalid_tensor_shape(
    active_config: SkeletalConfig,
    dummy_audio: AudioChunk,
    anomalous_frame_tuple: Tuple[VideoFrame, ...]
) -> None:
    """Ensures architectural constraints successfully block spatial shape mismatches from crashing C++ convolutions."""
    device = torch.device(active_config.device)
    model = HumanOmni().to(device)
    
    window = SynchronizedWindowSlice(
        window_id=17,
        start_time_sec=0.0,
        end_time_sec=1.0,
        audio=dummy_audio,
        frames=anomalous_frame_tuple,
        is_terminal=False
    )

    with pytest.raises(RuntimeError, match="Failed to process skeletal segment 17"):
        _process_segment(item=window, model=model, config=active_config, device=device)


def test_aggregate_payloads_expected_behaviour_for_multiple_segments(active_config: SkeletalConfig) -> None:
    """Tests functional reduction of a chronological payload buffer into a single statistical average matrix."""
    payloads: List[SkeletalDescriptorPayload] = [
        SkeletalDescriptorPayload(1, 0.0, 2.0, "calm", 0.1, 0.4),
        SkeletalDescriptorPayload(2, 2.0, 4.0, "rigid", 0.5, 0.2)
    ]
    
    aggregated = _aggregate_payloads(payloads, active_config)
    
    assert isinstance(aggregated, SkeletalDescriptorPayload)
    assert aggregated.segment_id == 0
    assert aggregated.start_time_sec == 0.0
    assert aggregated.end_time_sec == 4.0
    assert isinstance(aggregated.behavioral_descriptor, str)
    assert aggregated.perceptual_jitter == pytest.approx(0.3)
    assert aggregated.temporal_smoothness == pytest.approx(0.3)


def test_aggregate_payloads_raises_for_empty_sequence(active_config: SkeletalConfig) -> None:
    """Tests that mathematical averaging blocks empty sequence inputs."""
    with pytest.raises(ValueError, match="Aggregation requires a strictly populated sequence of payloads."):
        _aggregate_payloads([], active_config)


def test_run_skeletal_node_expected_behaviour_for_clean_termination(active_config: SkeletalConfig) -> None:
    """Tests deterministic queue shutdown upon sentinel payload ingestion without blocking."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    in_q.put(OrchestratorTerminalSentinel())
    
    worker = threading.Thread(
        target=run_skeletal_node,
        args=(in_q, out_q, active_config),
        kwargs={"timeout_sec": 0.1},
        daemon=True
    )
    worker.start()

    assert isinstance(out_q.get(timeout=2.0), OrchestratorTerminalSentinel)


def test_run_skeletal_node_expected_behaviour_for_processing_valid_stream(
    active_config: SkeletalConfig,
    dummy_audio: AudioChunk,
    multi_frame_tuple: Tuple[VideoFrame, ...]
) -> None:
    """Tests holistic stream-reduce lifecycle computing sequential mathematical reductions across multiple segments."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    window_1 = SynchronizedWindowSlice(
        window_id=99,
        start_time_sec=0.0,
        end_time_sec=1.0,
        audio=dummy_audio,
        frames=multi_frame_tuple,
        is_terminal=False
    )
    window_2 = SynchronizedWindowSlice(
        window_id=100,
        start_time_sec=1.0,
        end_time_sec=2.0,
        audio=dummy_audio,
        frames=multi_frame_tuple,
        is_terminal=False
    )

    in_q.put(window_1)
    in_q.put(window_2)
    in_q.put(OrchestratorTerminalSentinel())

    worker = threading.Thread(
        target=run_skeletal_node,
        args=(in_q, out_q, active_config),
        kwargs={"timeout_sec": 0.2},
        daemon=True
    )
    worker.start()

    payload: Any = out_q.get(timeout=3.0)
    assert payload.segment_id == 0
    assert payload.end_time_sec == 2.0
    assert isinstance(payload, SkeletalDescriptorPayload)
    
    sentinel: Any = out_q.get(timeout=3.0)
    assert isinstance(sentinel, OrchestratorTerminalSentinel)


def test_run_skeletal_node_raises_for_invalid_input_queue_type(active_config: SkeletalConfig) -> None:
    """Tests structural validation on cross-process communication bounds."""
    with pytest.raises(TypeError, match="Queues must be instances of queue.Queue"):
        run_skeletal_node(input_queue=[], output_queue=Queue(), config=active_config)  # type: ignore


def test_run_skeletal_node_raises_for_invalid_output_queue_type(active_config: SkeletalConfig) -> None:
    """Tests structural validation on cross-process communication bounds."""
    with pytest.raises(TypeError, match="Queues must be instances of queue.Queue"):
        run_skeletal_node(input_queue=Queue(), output_queue=None, config=active_config)  # type: ignore


def test_run_skeletal_node_raises_for_initialization_failure() -> None:
    """Tests immediate sentinel transmission if the neural network fails hardware binding or weight deserialization."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    calib_file: Path = Path("app/models/visual/test_invalid_init.json")
    calib_file.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_file, "w", encoding="utf-8") as file:
        json.dump({
            "perceptual_jitter": {"very_low_threshold": 0.1, "low_threshold": 0.2, "high_threshold": 0.3, "very_high_threshold": 0.4},
            "temporal_smoothness": {"very_low_threshold": 0.1, "low_threshold": 0.2, "high_threshold": 0.3, "very_high_threshold": 0.4}
        }, file)

    weights_file: Path = Path("app/models/visual/test_human_omni_init.pth")
    weights_file.parent.mkdir(parents=True, exist_ok=True)
    weights_file.touch() 

    invalid_config = SkeletalConfig(
        calibration_path=calib_file,
        weights_path=weights_file,
        device="cuda:99"
    )

    try:
        with pytest.raises(RuntimeError, match="Skeletal node initialization failed"):
            run_skeletal_node(
                input_queue=in_q,
                output_queue=out_q,
                config=invalid_config
            )

        sentinel: Any = out_q.get_nowait()
        assert isinstance(sentinel, OrchestratorTerminalSentinel)
    finally:
        if calib_file.exists():
            calib_file.unlink()
        if weights_file.exists():
            weights_file.unlink()


def test_run_skeletal_node_raises_for_saturated_downstream_queue(
    active_config: SkeletalConfig,
    dummy_audio: AudioChunk,
    single_frame_tuple: Tuple[VideoFrame, ...]
) -> None:
    """Tests process crash isolation if the Late Fusion Nexus bottlenecks terminal markers."""
    in_q: Queue = Queue()
    out_q: Queue = Queue(maxsize=1)

    window = SynchronizedWindowSlice(
        window_id=100,
        start_time_sec=0.0,
        end_time_sec=1.0,
        audio=dummy_audio,
        frames=single_frame_tuple,
        is_terminal=False
    )

    out_q.put("block_the_queue")
    in_q.put(window)
    in_q.put(OrchestratorTerminalSentinel())

    with pytest.raises(RuntimeError, match="Skeletal aggregation failure"):
        run_skeletal_node(
            input_queue=in_q,
            output_queue=out_q,
            config=active_config,
            timeout_sec=0.1
        )


def test_run_skeletal_node_raises_for_processing_failure(
    active_config: SkeletalConfig,
    dummy_audio: AudioChunk
) -> None:
    """Tests failure propagation and downstream sentinel transmission during matrix collapse."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    corrupted_window = SynchronizedWindowSlice(
        window_id=101,
        start_time_sec=0.0,
        end_time_sec=1.0,
        audio=dummy_audio,
        frames=(),
        is_terminal=False
    )

    in_q.put(corrupted_window)

    with pytest.raises(RuntimeError, match="Skeletal processing failure"):
        run_skeletal_node(
            input_queue=in_q,
            output_queue=out_q,
            config=active_config,
            timeout_sec=2.0
        )

    sentinel: Any = out_q.get_nowait()
    assert isinstance(sentinel, OrchestratorTerminalSentinel)