import threading
from pathlib import Path
from queue import Queue
from typing import Any, List, Tuple, Dict

import numpy as np
import pytest
import torch

from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.orchestration.types import AudioChunk, SynchronizedWindowSlice, VideoFrame
from app.src.visual_edge.facial_gaze.config import FacialConfig
from app.src.visual_edge.facial_gaze.node import (
    _aggregate_payloads,
    _initialize_facial_models,
    _process_segment,
    run_facial_node,
)
from app.src.visual_edge.facial_gaze.poster import POSTERPlus
from app.src.visual_edge.facial_gaze.types import FacialDescriptorPayload


@pytest.fixture
def active_config() -> FacialConfig:
    """Provides a standard facial configuration matrix bound to available hardware."""
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    return FacialConfig(device=device)


@pytest.fixture(scope="module")
def loaded_production_models() -> Tuple[POSTERPlus, Any, torch.device]:
    """Loads the POSTER++ model and regression matrix strictly once per session."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA architecture is unavailable. Skipping structural tests.")

    weights_path: Path = Path("app/models/visual/poster_plus_latent.pth")
    if not weights_path.exists():
        pytest.skip(f"Actual model matrix missing at {weights_path}.")

    config = FacialConfig(device="cuda:0")
    return _initialize_facial_models(config)


@pytest.fixture
def dummy_audio() -> AudioChunk:
    """Generates an empty audio chunk to satisfy multimodal payload constraints."""
    return AudioChunk(sample_rate=16000, timestamp_sec=0.0, data=np.zeros(1600, dtype=np.float32))


@pytest.fixture
def multi_frame_tuple() -> Tuple[VideoFrame, ...]:
    """Generates a sequence of random RGB frames simulating continuous video."""
    return (
        VideoFrame(frame_index=0, timestamp_sec=0.0, data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)),
        VideoFrame(frame_index=1, timestamp_sec=0.2, data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    )


@pytest.fixture
def single_frame_tuple() -> Tuple[VideoFrame, ...]:
    """Generates a solitary RGB frame to simulate terminal edge cases."""
    return (
        VideoFrame(frame_index=0, timestamp_sec=2.5, data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)),
    )


def test_process_segment_expected_behaviour_for_multi_frame_window(
    active_config: FacialConfig,
    loaded_production_models: Tuple[POSTERPlus, Any, torch.device],
    dummy_audio: AudioChunk,
    multi_frame_tuple: Tuple[VideoFrame, ...]
) -> None:
    """Verifies end-to-end extraction of facial latents and behavioral scores across multiple frames."""
    visual_encoder, regressor, device = loaded_production_models
    
    window = SynchronizedWindowSlice(
        window_id=1,
        start_time_sec=0.0,
        end_time_sec=2.5,
        audio=dummy_audio,
        frames=multi_frame_tuple,
        is_terminal=False
    )
    
    payload: FacialDescriptorPayload = _process_segment(
        item=window,
        visual_encoder=visual_encoder,
        regressor=regressor,
        device=device,
        config=active_config,
        latent_cache={}
    )
    
    assert isinstance(payload, FacialDescriptorPayload)
    assert payload.segment_id == "1"
    assert payload.start_time_sec == 0.0
    assert payload.end_time_sec == 2.5
    assert isinstance(payload.behavioral_descriptor, str)
    assert len(payload.behavioral_descriptor) > 0


def test_process_segment_expected_behaviour_for_single_frame_window(
    active_config: FacialConfig,
    loaded_production_models: Tuple[POSTERPlus, Any, torch.device],
    dummy_audio: AudioChunk,
    single_frame_tuple: Tuple[VideoFrame, ...]
) -> None:
    """Verifies extraction stability when processing a sparsely populated, solitary frame segment."""
    visual_encoder, regressor, device = loaded_production_models
    
    window = SynchronizedWindowSlice(
        window_id=2,
        start_time_sec=2.5,
        end_time_sec=3.0,
        audio=dummy_audio,
        frames=single_frame_tuple,
        is_terminal=True
    )
    
    payload: FacialDescriptorPayload = _process_segment(
        item=window,
        visual_encoder=visual_encoder,
        regressor=regressor,
        device=device,
        config=active_config,
        latent_cache={}
    )
    
    assert isinstance(payload, FacialDescriptorPayload)
    assert payload.segment_id == "2"


def test_process_segment_expected_behaviour_for_memoized_cache_hit(
    active_config: FacialConfig,
    loaded_production_models: Tuple[POSTERPlus, Any, torch.device],
    dummy_audio: AudioChunk,
    multi_frame_tuple: Tuple[VideoFrame, ...]
) -> None:
    """Ensures overlapping frames utilize cached latents to bypass redundant GPU inference."""
    visual_encoder, regressor, device = loaded_production_models
    
    window = SynchronizedWindowSlice(
        window_id=3,
        start_time_sec=0.0,
        end_time_sec=2.5,
        audio=dummy_audio,
        frames=multi_frame_tuple,
        is_terminal=False
    )
    
    latent_cache: Dict[int, torch.Tensor] = {}
    
    payload_one: FacialDescriptorPayload = _process_segment(
        item=window,
        visual_encoder=visual_encoder,
        regressor=regressor,
        device=device,
        config=active_config,
        latent_cache=latent_cache
    )
    
    payload_two: FacialDescriptorPayload = _process_segment(
        item=window,
        visual_encoder=visual_encoder,
        regressor=regressor,
        device=device,
        config=active_config,
        latent_cache=latent_cache
    )
    
    assert payload_one.extraversion_score == pytest.approx(payload_two.extraversion_score)
    assert payload_one.neuroticism_score == pytest.approx(payload_two.neuroticism_score)


def test_process_segment_raises_for_empty_frames_tuple(
    active_config: FacialConfig,
    loaded_production_models: Tuple[POSTERPlus, Any, torch.device],
    dummy_audio: AudioChunk
) -> None:
    """Ensures the pipeline safely aborts if an empty frame sequence is passed."""
    visual_encoder, regressor, device = loaded_production_models
    
    empty_window = SynchronizedWindowSlice(
        window_id=6, start_time_sec=4.0, end_time_sec=5.0, audio=dummy_audio, frames=(), is_terminal=False
    )
    
    with pytest.raises(RuntimeError, match="Pipeline execution failed for segment 6"):
        _process_segment(empty_window, visual_encoder, regressor, device, active_config, {})


def test_process_segment_raises_for_invalid_input_type(
    active_config: FacialConfig,
    loaded_production_models: Tuple[POSTERPlus, Any, torch.device]
) -> None:
    """Ensures structural type assertions reject spoofed payload dictionaries or lists."""
    visual_encoder, regressor, device = loaded_production_models
    
    with pytest.raises(TypeError, match="Input must be a strictly typed SynchronizedWindowSlice"):
        _process_segment([1, 2, 3], visual_encoder, regressor, device, active_config, {})  # type: ignore


def test_initialize_facial_models_raises_for_missing_weights(active_config: FacialConfig) -> None:
    """Verifies the initialization sequence crashes gracefully if model weights are missing or deleted."""
    invalid_path = Path("app/models/visual/deleted_weights.pth")
    object.__setattr__(active_config, "poster_weights_path", invalid_path)
    
    with pytest.raises(RuntimeError, match="Failed to initialize POSTER\\+\\+ visual encoder"):
        _initialize_facial_models(config=active_config)


def test_aggregate_payloads_expected_behaviour_for_multiple_segments(active_config: FacialConfig) -> None:
    """Tests functional reduction of a chronological payload buffer into global statistical averages."""
    payloads: List[FacialDescriptorPayload] = [
        FacialDescriptorPayload("1", 0.0, 2.0, "neutral", 0.8, 0.2, 0.5, 0.9, 0.6),
        FacialDescriptorPayload("2", 2.0, 4.0, "smiling", 0.6, 0.4, 0.3, 0.7, 0.8)
    ]
    
    aggregated = _aggregate_payloads(payloads, active_config)
    
    assert isinstance(aggregated, FacialDescriptorPayload)
    assert aggregated.segment_id == "0"
    assert aggregated.extraversion_score == pytest.approx(0.7)
    assert aggregated.neuroticism_score == pytest.approx(0.3)


def test_aggregate_payloads_raises_for_empty_sequence(active_config: FacialConfig) -> None:
    """Tests that global mathematical averaging blocks empty sequence inputs."""
    with pytest.raises(ValueError, match="Aggregation requires a strictly populated sequence of payloads."):
        _aggregate_payloads([], active_config)


def test_run_facial_node_expected_behaviour_for_clean_termination(active_config: FacialConfig) -> None:
    """Tests structural egress interception of a terminal sentinel sequence without blocking."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    in_q.put(OrchestratorTerminalSentinel())
    
    worker = threading.Thread(
        target=run_facial_node,
        args=(in_q, out_q, active_config),
        kwargs={"timeout_sec": 0.1},
        daemon=True
    )
    worker.start()
    
    assert isinstance(out_q.get(timeout=2.0), OrchestratorTerminalSentinel)


def test_run_facial_node_expected_behaviour_for_ignoring_invalid_items(active_config: FacialConfig) -> None:
    """Tests the node loop gracefully skipping unmapped data structural types."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    in_q.put("invalid_string_payload")
    in_q.put(OrchestratorTerminalSentinel())

    worker = threading.Thread(
        target=run_facial_node,
        args=(in_q, out_q, active_config),
        kwargs={"timeout_sec": 0.1},
        daemon=True
    )
    worker.start()
    
    assert isinstance(out_q.get(timeout=2.0), OrchestratorTerminalSentinel)
    assert out_q.empty()


def test_run_facial_node_raises_for_invalid_queue_types(active_config: FacialConfig) -> None:
    """Tests strict type assertion blocking on the input queue signature."""
    with pytest.raises(TypeError, match="input_queue must be an instance of queue.Queue"):
        run_facial_node(input_queue=[], output_queue=Queue(), config=active_config)  # type: ignore


def test_run_facial_node_raises_for_invalid_config_type() -> None:
    """Tests strict instance validation blocking invalid configuration parameters."""
    with pytest.raises(TypeError, match="config must be an instance of FacialConfig"):
        run_facial_node(input_queue=Queue(), output_queue=Queue(), config={})  # type: ignore


def test_run_facial_node_raises_for_saturated_downstream_queue(active_config: FacialConfig) -> None:
    """Tests node termination crashing sequentially if the downstream buffer triggers saturation."""
    in_q: Queue = Queue()
    out_q: Queue = Queue(maxsize=1)

    out_q.put("block_the_queue")
    in_q.put(OrchestratorTerminalSentinel())

    with pytest.raises(RuntimeError, match="Output queue saturated during termination sequence."):
        run_facial_node(input_queue=in_q, output_queue=out_q, config=active_config, timeout_sec=0.1)