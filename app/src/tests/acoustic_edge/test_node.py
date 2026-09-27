import json
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, List, Tuple

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge
from transformers import AutoFeatureExtractor, WavLMModel

from app.src.acoustic_edge.config import AcousticConfig
from app.src.acoustic_edge.encoder import load_wavlm_components
from app.src.acoustic_edge.generator import load_ridge_regressor
from app.src.acoustic_edge.node import (
    _aggregate_payloads,
    _process_segment,
    run_acoustic_node,
)
from app.src.acoustic_edge.types import AcousticDescriptorPayload
from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.orchestration.types import AudioChunk, SynchronizedWindowSlice


def create_mock_free_segment(
    window_id: int, start_sec: float, end_sec: float, chunk: AudioChunk
) -> SynchronizedWindowSlice:
    """Generates a synthetic synchronized window slice for unmocked acoustic testing."""
    return SynchronizedWindowSlice(
        window_id=window_id,
        start_time_sec=start_sec,
        end_time_sec=end_sec,
        audio=chunk,
        frames=(),
        is_terminal=False
    )


@pytest.fixture(scope="module")
def production_infrastructure() -> AcousticConfig:
    """Provides a hardware-validated AcousticConfig utilizing authentic model paths and CUDA if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA architecture is unavailable. Skipping structural tests.")

    model_path: Path = Path("app/models/acoustic/ridge_ocean_regressor.joblib")
    if not model_path.exists():
        pytest.skip(f"Actual model matrix missing at {model_path}.")

    calib_file: Path = Path("app/models/acoustic/calibration.json")
    if not calib_file.exists():
        pytest.skip(f"Genuine calibration file missing at {calib_file}. Run generation script first.")

    return AcousticConfig(
        wavlm_repo_id="microsoft/wavlm-large",
        wavlm_revision="refs/pr/7",
        ridge_model_path=model_path,
        calibration_path=calib_file,
        device="cuda:0",
        use_float16=True
    )


@pytest.fixture(scope="module")
def loaded_production_models(
    production_infrastructure: AcousticConfig,
) -> Tuple[AutoFeatureExtractor, WavLMModel, Ridge]:
    """Loads the WavLM and Ridge models strictly once per session to prevent redundant VRAM initialization."""
    extractor, wavlm = load_wavlm_components(production_infrastructure)
    ridge = load_ridge_regressor(production_infrastructure)
    return extractor, wavlm, ridge


def test_process_segment_expected_behaviour_for_valid_segment(
    production_infrastructure: AcousticConfig,
    loaded_production_models: Tuple[AutoFeatureExtractor, WavLMModel, Ridge]
) -> None:
    """Verifies end-to-end extraction of acoustic latents and behavioral scores from a valid raw audio chunk."""
    extractor, wavlm, ridge = loaded_production_models

    sample_rate: int = extractor.sampling_rate
    audio_data: np.ndarray = np.random.randn(sample_rate).astype(np.float32)
    chunk = AudioChunk(data=audio_data, sample_rate=sample_rate, timestamp_sec=10.0)

    segment = create_mock_free_segment(77, 10.0, 11.0, chunk)

    payload: AcousticDescriptorPayload = _process_segment(
        item=segment,
        wavlm_extractor=extractor,
        wavlm_model=wavlm,
        ridge_model=ridge,
        config=production_infrastructure
    )

    assert isinstance(payload, AcousticDescriptorPayload)
    assert payload.segment_id == 77
    assert payload.start_time_sec == 10.0
    assert isinstance(payload.extraversion_score, float)
    assert isinstance(payload.openness_score, float)


def test_process_segment_raises_for_invalid_audio_data(
    production_infrastructure: AcousticConfig,
    loaded_production_models: Tuple[AutoFeatureExtractor, WavLMModel, Ridge]
) -> None:
    """Verifies the acoustic pipeline gracefully intercepts and raises errors on corrupted or empty arrays."""
    extractor, wavlm, ridge = loaded_production_models

    empty_data: np.ndarray = np.array([], dtype=np.float32)
    empty_chunk = AudioChunk(data=empty_data, sample_rate=16000, timestamp_sec=0.0)
    segment = create_mock_free_segment(1, 0.0, 1.0, empty_chunk)

    with pytest.raises(RuntimeError, match="Failed to process acoustic segment 1"):
        _process_segment(
            item=segment,
            wavlm_extractor=extractor,
            wavlm_model=wavlm,
            ridge_model=ridge,
            config=production_infrastructure
        )


def test_aggregate_payloads_expected_behaviour_for_multiple_segments() -> None:
    """Verifies functional reduction of a chronological payload buffer into global statistical averages."""
    payloads: List[AcousticDescriptorPayload] = [
        AcousticDescriptorPayload(1, 0.0, 2.0, ("confident",), 0.8, 0.2, 0.5, 0.9, 0.6),
        AcousticDescriptorPayload(2, 2.0, 4.0, ("clear",), 0.6, 0.4, 0.3, 0.7, 0.8)
    ]
    
    aggregated = _aggregate_payloads(payloads)
    
    assert isinstance(aggregated, AcousticDescriptorPayload)
    assert aggregated.segment_id == 0
    assert aggregated.start_time_sec == 0.0
    assert aggregated.end_time_sec == 4.0
    assert "confident" in aggregated.vocal_cues
    assert "clear" in aggregated.vocal_cues
    assert aggregated.extraversion_score == pytest.approx(0.7)
    assert aggregated.neuroticism_score == pytest.approx(0.3)


def test_aggregate_payloads_raises_for_empty_sequence() -> None:
    """Tests that mathematical averaging blocks empty sequence inputs from crashing the reducer."""
    with pytest.raises(ValueError, match="Aggregation requires a strictly populated sequence of payloads."):
        _aggregate_payloads([])


def test_run_acoustic_node_expected_behaviour_for_clean_termination(
    production_infrastructure: AcousticConfig
) -> None:
    """Verifies deterministic egress queue shutdown upon sentinel payload ingestion without blocking."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    in_q.put(OrchestratorTerminalSentinel())
    
    worker = threading.Thread(
        target=run_acoustic_node,
        args=(in_q, out_q, production_infrastructure),
        kwargs={"timeout_sec": 0.1},
        daemon=True
    )
    worker.start()
    
    # Extended timeout to 15 seconds to allow the unmocked WavLM model to load from disk
    assert isinstance(out_q.get(timeout=15.0), OrchestratorTerminalSentinel)


def test_run_acoustic_node_expected_behaviour_for_processing_valid_stream(
    production_infrastructure: AcousticConfig
) -> None:
    """Verifies sequential queue iteration and acoustic evaluation over multiple chronological windows."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    sample_rate: int = 16000
    audio_data_1: np.ndarray = np.random.randn(sample_rate).astype(np.float32)
    chunk_1 = AudioChunk(data=audio_data_1, sample_rate=sample_rate, timestamp_sec=0.0)
    segment_1 = create_mock_free_segment(1, 0.0, 1.0, chunk_1)

    audio_data_2: np.ndarray = np.random.randn(sample_rate).astype(np.float32)
    chunk_2 = AudioChunk(data=audio_data_2, sample_rate=sample_rate, timestamp_sec=1.0)
    segment_2 = create_mock_free_segment(2, 1.0, 2.0, chunk_2)

    in_q.put(segment_1)
    in_q.put(segment_2)
    in_q.put(OrchestratorTerminalSentinel())

    worker = threading.Thread(
        target=run_acoustic_node,
        args=(in_q, out_q, production_infrastructure),
        kwargs={"timeout_sec": 0.5},
        daemon=True
    )
    worker.start()

    payload: Any = out_q.get(timeout=15.0)
    assert isinstance(payload, AcousticDescriptorPayload)
    assert payload.segment_id == 0
    assert hasattr(payload, "conscientiousness_score")
    
    sentinel: Any = out_q.get(timeout=3.0)
    assert isinstance(sentinel, OrchestratorTerminalSentinel)


def test_run_acoustic_node_expected_behaviour_for_ignoring_invalid_items(
    production_infrastructure: AcousticConfig
) -> None:
    """Verifies the internal router gracefully drops structurally invalid types silently."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    in_q.put("invalid_string_payload")
    in_q.put(404)
    in_q.put(OrchestratorTerminalSentinel())

    worker = threading.Thread(
        target=run_acoustic_node,
        args=(in_q, out_q, production_infrastructure),
        kwargs={"timeout_sec": 0.1},
        daemon=True
    )
    worker.start()
    
    assert isinstance(out_q.get(timeout=15.0), OrchestratorTerminalSentinel)
    assert out_q.empty()


def test_run_acoustic_node_expected_behaviour_for_queue_timeout_recovery(
    production_infrastructure: AcousticConfig
) -> None:
    """Verifies the consumer loop survives empty queue polling and cleanly exits upon subsequent sentinel arrival."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    worker = threading.Thread(
        target=run_acoustic_node,
        args=(in_q, out_q, production_infrastructure),
        kwargs={"timeout_sec": 0.05},
        daemon=True
    )
    worker.start()

    time.sleep(0.1)
    in_q.put(OrchestratorTerminalSentinel())
    
    assert isinstance(out_q.get(timeout=15.0), OrchestratorTerminalSentinel)


def test_run_acoustic_node_raises_for_invalid_input_queue_type(
    production_infrastructure: AcousticConfig
) -> None:
    """Verifies strict type assertion blocking on the cross-process ingress interface."""
    with pytest.raises(TypeError, match="input_queue must be an instance of queue.Queue"):
        run_acoustic_node(
            input_queue=[],  # type: ignore
            output_queue=Queue(), 
            config=production_infrastructure
        )


def test_run_acoustic_node_raises_for_invalid_output_queue_type(
    production_infrastructure: AcousticConfig
) -> None:
    """Verifies strict type assertion blocking on the cross-process egress interface."""
    with pytest.raises(TypeError, match="output_queue must be an instance of queue.Queue"):
        run_acoustic_node(
            input_queue=Queue(), 
            output_queue=None,  # type: ignore
            config=production_infrastructure
        )


def test_run_acoustic_node_raises_for_invalid_config_type() -> None:
    """Verifies strict instance validation blocking corrupted initialization parameters."""
    with pytest.raises(TypeError, match="config must be an instance of AcousticConfig"):
        run_acoustic_node(
            input_queue=Queue(), 
            output_queue=Queue(), 
            config=dict()  # type: ignore
        )


def test_run_acoustic_node_raises_for_initialization_failure() -> None:
    """Verifies that an unbootable model routes an error sentinel downstream immediately before raising."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    calib_file: Path = Path("app/models/acoustic/test_invalid_init.json")
    calib_file.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_file, "w", encoding="utf-8") as f:
        json.dump({"very_low_threshold": 0.2, "low_threshold": 0.4, "high_threshold": 0.6, "very_high_threshold": 0.8}, f)

    invalid_config = AcousticConfig(
        wavlm_repo_id="invalid_user/non_existent_model_123",
        wavlm_revision="refs/pr/7",
        calibration_path=calib_file,
        device="cuda:0"
    )

    try:
        with pytest.raises(RuntimeError, match="Acoustic node initialization failed"):
            run_acoustic_node(
                input_queue=in_q, 
                output_queue=out_q, 
                config=invalid_config
            )

        sentinel: Any = out_q.get_nowait()
        assert isinstance(sentinel, OrchestratorTerminalSentinel)
    finally:
        if calib_file.exists():
            calib_file.unlink()


def test_run_acoustic_node_raises_for_saturated_downstream_queue(
    production_infrastructure: AcousticConfig
) -> None:
    """Verifies encapsulation stability if the downstream nexus bottlenecks during shutdown."""
    in_q: Queue = Queue()
    out_q: Queue = Queue(maxsize=1)

    out_q.put("block_the_queue")
    in_q.put(OrchestratorTerminalSentinel())

    with pytest.raises(RuntimeError, match="Output queue saturated during termination sequence"):
        run_acoustic_node(
            input_queue=in_q, 
            output_queue=out_q, 
            config=production_infrastructure, 
            timeout_sec=0.1
        )


def test_run_acoustic_node_raises_for_processing_failure(
    production_infrastructure: AcousticConfig
) -> None:
    """Verifies downstream propagation intercepts raw Numpy buffer corruption during active inference."""
    in_q: Queue = Queue()
    out_q: Queue = Queue()

    empty_data: np.ndarray = np.array([], dtype=np.float32)
    empty_chunk = AudioChunk(data=empty_data, sample_rate=16000, timestamp_sec=0.0)
    corrupted_segment = create_mock_free_segment(99, 0.0, 1.0, empty_chunk)

    in_q.put(corrupted_segment)
    in_q.put(OrchestratorTerminalSentinel())

    with pytest.raises(RuntimeError, match="Acoustic processing failure"):
        run_acoustic_node(
            input_queue=in_q, 
            output_queue=out_q, 
            config=production_infrastructure, 
            timeout_sec=2.0
        )

    sentinel: Any = out_q.get_nowait()
    assert isinstance(sentinel, OrchestratorTerminalSentinel)