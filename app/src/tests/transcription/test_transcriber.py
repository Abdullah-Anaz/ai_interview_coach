import queue
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest
import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.orchestration.types import AudioChunk, SynchronizedSegment, SynchronizedWindowSlice
from app.src.transcription.config import TranscriptionConfig
from app.src.transcription.transcriber import (
    _ensure_local_model_weights,
    _execute_inference,
    _process_segment,
    initialize_transducer_model,
    run_transcription_node,
)


# =====================================================================
# FIXTURES & HARDWARE CONFIGURATION
# =====================================================================

@pytest.fixture(scope="module")
def active_device() -> str:
    """Dynamically binds to CUDA if available, falling back to CPU for CI accessibility."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def active_config(active_device: str) -> TranscriptionConfig:
    """Provides a strict configuration matrix dynamically matching the available hardware."""
    return TranscriptionConfig(
        device=active_device,
        use_float16=True if "cuda" in active_device else False
    )


@pytest.fixture(scope="module")
def gpu_config() -> TranscriptionConfig:
    """Provides a strict GPU-bound configuration matrix leveraging float16 half-precision."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA architecture is unavailable. Skipping GPU-bound structural tests.")
    return TranscriptionConfig(device="cuda:0", use_float16=True)


@pytest.fixture(scope="module")
def cpu_config() -> TranscriptionConfig:
    """Provides a baseline CPU configuration matrix leveraging standard float32 precision."""
    return TranscriptionConfig(device="cpu", use_float16=False)


@pytest.fixture(scope="module")
def loaded_transducer_model(active_config: TranscriptionConfig) -> EncDecRNNTBPEModel:
    """
    Atomically loads the NeMo Parakeet-TDT weights from disk strictly once per testing session,
    binding it to the active hardware accelerator to prevent redundant I/O bottlenecks.
    """
    return initialize_transducer_model(active_config)


@pytest.fixture
def dummy_audio_chunk() -> AudioChunk:
    """Generates a mathematically valid 1-second acoustic numpy array at 16kHz."""
    return AudioChunk(
        timestamp_sec=0.0,
        data=np.random.randn(16000).astype(np.float32),
        sample_rate=16000
    )


@pytest.fixture
def standard_window_slice(dummy_audio_chunk: AudioChunk) -> SynchronizedWindowSlice:
    """Constructs a fully populated, standard temporal window for unmocked processing."""
    return SynchronizedWindowSlice(
        window_id=1,
        start_time_sec=0.0,
        end_time_sec=1.0,
        audio=dummy_audio_chunk,
        frames=tuple(),
        is_terminal=False
    )


# =====================================================================
# _ENSURE_LOCAL_MODEL_WEIGHTS TESTS
# =====================================================================

def test_ensure_local_model_weights_expected_behaviour_for_valid_repo(cpu_config: TranscriptionConfig) -> None:
    """Verifies the module successfully locates (or downloads) and returns the absolute POSIX path to the weights."""
    path = _ensure_local_model_weights(cpu_config)
    assert isinstance(path, Path)
    assert path.exists()
    assert path.name == cpu_config.model_filename


def test_ensure_local_model_weights_raises_for_invalid_repo() -> None:
    """Verifies structural error boundaries when the HuggingFace API rejects a non-existent repository."""
    broken_config = TranscriptionConfig(
        model_repo_id="invalid-org/does-not-exist",
        model_filename="fake.nemo"
    )
    with pytest.raises(RuntimeError, match="Failed to download Parakeet-TDT weights"):
        _ensure_local_model_weights(broken_config)


# =====================================================================
# INITIALIZE_TRANSDUCER_MODEL TESTS
# =====================================================================

def test_initialize_transducer_model_expected_behaviour_for_gpu_float16(gpu_config: TranscriptionConfig) -> None:
    """Verifies true hardware instantiation on CUDA utilizing float16 VRAM compression constraints."""
    model = initialize_transducer_model(gpu_config)
    assert model is not None
    param = next(model.parameters())
    assert param.dtype == torch.float16
    assert "cuda" in str(param.device)


def test_initialize_transducer_model_expected_behaviour_for_cpu_float32(cpu_config: TranscriptionConfig) -> None:
    """Verifies functional fallback architecture on CPU utilizing standard float32 precision."""
    model = initialize_transducer_model(cpu_config)
    assert model is not None
    param = next(model.parameters())
    assert param.dtype == torch.float32
    assert "cpu" in str(param.device)


def test_initialize_transducer_model_raises_for_missing_weights() -> None:
    """Verifies catastrophic initialization failure sequentially propagates if weights are corrupted/missing."""
    broken_config = TranscriptionConfig(
        model_filename="corrupted_file.nemo"
    )
    with pytest.raises(RuntimeError, match="Failed to download Parakeet-TDT weights"):
        initialize_transducer_model(broken_config)


# =====================================================================
# _EXECUTE_INFERENCE TESTS
# =====================================================================

def test_execute_inference_expected_behaviour_for_standard_tensor(
    loaded_transducer_model: EncDecRNNTBPEModel,
    active_device: str
) -> None:
    """Verifies the raw PyTorch forward pass successfully calculates log-probabilities on standard acoustic vectors."""
    audio_signal = torch.randn(16000, dtype=torch.float32)
    
    output = _execute_inference(loaded_transducer_model, audio_signal, active_device)
    assert isinstance(output, tuple)
    assert len(output) >= 2 
    assert isinstance(output[0], torch.Tensor)


def test_execute_inference_expected_behaviour_for_short_tensor_padding(
    loaded_transducer_model: EncDecRNNTBPEModel,
    active_device: str
) -> None:
    """Edge Case: Verifies micro-chunks (<1600 samples) dynamically apply zero-padding to prevent STFT collapse."""
    tiny_signal = torch.randn(500, dtype=torch.float32)
    
    output = _execute_inference(loaded_transducer_model, tiny_signal, active_device)
    assert isinstance(output, tuple)
    assert isinstance(output[0], torch.Tensor)


def test_execute_inference_raises_for_invalid_tensor_shape(
    loaded_transducer_model: EncDecRNNTBPEModel,
    active_device: str
) -> None:
    """Verifies hardware failure boundaries intercept multi-dimensional matrix mismatches."""
    invalid_signal = torch.randn(2, 16000, dtype=torch.float32)  # Expected 1D before batching
    
    # The error occurs prior to the try/except block during the zero-padding torch.cat() operation
    with pytest.raises(RuntimeError, match="Tensors must have same number of dimensions"):
        _execute_inference(loaded_transducer_model, invalid_signal, active_device)


# =====================================================================
# _PROCESS_SEGMENT TESTS
# =====================================================================

def test_process_segment_expected_behaviour_for_valid_window(
    standard_window_slice: SynchronizedWindowSlice,
    loaded_transducer_model: EncDecRNNTBPEModel,
    active_config: TranscriptionConfig
) -> None:
    """Verifies end-to-end holistic mapping from Numpy payload to temporally aligned text datastructures."""
    segment = _process_segment(standard_window_slice, loaded_transducer_model, active_config)
    
    assert isinstance(segment, SynchronizedSegment)
    assert segment.segment_id == 1
    assert segment.audio_slice_start_sec == 0.0
    assert segment.audio_slice_end_sec == 1.0
    assert isinstance(segment.words, tuple)
    assert segment.is_final is False


def test_process_segment_expected_behaviour_for_terminal_window_without_frames(
    dummy_audio_chunk: AudioChunk,
    loaded_transducer_model: EncDecRNNTBPEModel,
    active_config: TranscriptionConfig
) -> None:
    """Edge Case: Verifies semantic extraction remains stable on pure-acoustic segments lacking spatial frames."""
    terminal_slice = SynchronizedWindowSlice(
        window_id=2, start_time_sec=1.0, end_time_sec=2.0, audio=dummy_audio_chunk, frames=tuple(), is_terminal=True
    )
    
    segment = _process_segment(terminal_slice, loaded_transducer_model, active_config)
    assert segment.is_final is True
    assert segment.visual_frame_end_idx == 0


def test_process_segment_raises_for_invalid_input_type(
    loaded_transducer_model: EncDecRNNTBPEModel,
    active_config: TranscriptionConfig
) -> None:
    """Verifies structural boundary defenses reject spoofed mapping interfaces."""
    with pytest.raises(TypeError, match="Input payload must be a strictly typed SynchronizedWindowSlice"):
        _process_segment({"spoofed_data": True}, loaded_transducer_model, active_config)  # type: ignore


def test_process_segment_raises_for_inference_failure(
    loaded_transducer_model: EncDecRNNTBPEModel,
    active_config: TranscriptionConfig
) -> None:
    """Verifies downstream propagation intercepts raw Numpy buffer corruption."""
    
    # Bypassing the strict __post_init__ 1D matrix validation to intentionally crash the PyTorch Cuda kernel
    corrupted_chunk = AudioChunk.__new__(AudioChunk)
    object.__setattr__(corrupted_chunk, "sample_rate", 16000)
    object.__setattr__(corrupted_chunk, "timestamp_sec", 0.0)
    object.__setattr__(corrupted_chunk, "data", np.array([[1, 2], [3, 4]], dtype=np.float32))
    
    corrupted_slice = SynchronizedWindowSlice(
        window_id=3, start_time_sec=0.0, end_time_sec=1.0, audio=corrupted_chunk, frames=tuple(), is_terminal=False
    )
    
    with pytest.raises(RuntimeError, match="ASR inference failure on segment 3"):
        _process_segment(corrupted_slice, loaded_transducer_model, active_config)


# =====================================================================
# RUN_TRANSCRIPTION_NODE TESTS
# =====================================================================

def test_run_transcription_node_expected_behaviour_for_clean_termination(cpu_config: TranscriptionConfig) -> None:
    """Verifies deterministic egress queue shutdown upon sentinel payload ingestion without hanging."""
    in_q, out_q = queue.Queue(), queue.Queue()
    in_q.put(OrchestratorTerminalSentinel())

    run_transcription_node(in_q, out_q, cpu_config, timeout_sec=1.0)

    result = out_q.get(timeout=1.0)
    assert isinstance(result, OrchestratorTerminalSentinel)


def test_run_transcription_node_expected_behaviour_for_processing_valid_stream(
    standard_window_slice: SynchronizedWindowSlice,
    cpu_config: TranscriptionConfig
) -> None:
    """Verifies sequential queue iteration and semantic text transcription over multiple chronological windows."""
    in_q, out_q = queue.Queue(), queue.Queue()
    
    in_q.put(standard_window_slice)
    in_q.put(OrchestratorTerminalSentinel())

    run_transcription_node(in_q, out_q, cpu_config, timeout_sec=2.0)

    segment = out_q.get(timeout=1.0)
    assert isinstance(segment, SynchronizedSegment)
    assert segment.segment_id == 1
    
    sentinel = out_q.get(timeout=1.0)
    assert isinstance(sentinel, OrchestratorTerminalSentinel)


def test_run_transcription_node_expected_behaviour_for_ignoring_invalid_items(cpu_config: TranscriptionConfig) -> None:
    """Verifies the internal router gracefully drops structurally invalid types silently."""
    in_q, out_q = queue.Queue(), queue.Queue()
    
    in_q.put("corrupted_string")
    in_q.put(OrchestratorTerminalSentinel())

    run_transcription_node(in_q, out_q, cpu_config, timeout_sec=1.0)

    sentinel = out_q.get(timeout=1.0)
    assert isinstance(sentinel, OrchestratorTerminalSentinel)
    assert out_q.empty()


def test_run_transcription_node_raises_for_invalid_input_queue(cpu_config: TranscriptionConfig) -> None:
    """Verifies strict type assertion blocking on the cross-process ingress interface."""
    with pytest.raises(TypeError, match="Communication endpoints must strictly be queue.Queue instances."):
        run_transcription_node(input_queue=[], output_queue=queue.Queue(), config=cpu_config)  # type: ignore


def test_run_transcription_node_raises_for_invalid_output_queue(cpu_config: TranscriptionConfig) -> None:
    """Verifies strict type assertion blocking on the cross-process egress interface."""
    with pytest.raises(TypeError, match="Communication endpoints must strictly be queue.Queue instances."):
        run_transcription_node(input_queue=queue.Queue(), output_queue=None, config=cpu_config)  # type: ignore


def test_run_transcription_node_raises_for_invalid_config_type() -> None:
    """Verifies strict instance validation blocking corrupted initialization parameters."""
    with pytest.raises(TypeError, match="Configuration must be a TranscriptionConfig instance."):
        run_transcription_node(queue.Queue(), queue.Queue(), config={})  # type: ignore


def test_run_transcription_node_raises_for_initialization_failure() -> None:
    """Verifies that an unbootable model routes an error sentinel downstream immediately before raising."""
    in_q, out_q = queue.Queue(), queue.Queue()
    broken_config = TranscriptionConfig(model_filename="corrupted_file.nemo")
    
    with pytest.raises(RuntimeError, match="Transcription initialization aborted"):
        run_transcription_node(in_q, out_q, broken_config)
        
    sentinel = out_q.get_nowait()
    assert isinstance(sentinel, OrchestratorTerminalSentinel)
    assert isinstance(sentinel.error, Exception)


def test_run_transcription_node_raises_for_output_queue_saturation_on_termination(cpu_config: TranscriptionConfig) -> None:
    """Error Case: Verifies encapsulation stability if the downstream nexus bottlenecks during shutdown."""
    in_q = queue.Queue()
    out_q = queue.Queue(maxsize=1)
    
    out_q.put("blocking_element")
    in_q.put(OrchestratorTerminalSentinel())

    with pytest.raises(RuntimeError, match="Egress saturated during terminal cascade."):
        run_transcription_node(in_q, out_q, cpu_config, timeout_sec=0.1)


def test_run_transcription_node_raises_for_output_queue_saturation_on_processing(
    standard_window_slice: SynchronizedWindowSlice,
    cpu_config: TranscriptionConfig
) -> None:
    """Error Case: Verifies exact error message formatting if the downstream nexus bottlenecks during transcription."""
    in_q = queue.Queue()
    out_q = queue.Queue(maxsize=1)
    
    out_q.put("blocking_element")
    in_q.put(standard_window_slice)

    with pytest.raises(RuntimeError, match="Transcription egress queue saturated."):
        run_transcription_node(in_q, out_q, cpu_config, timeout_sec=0.1)