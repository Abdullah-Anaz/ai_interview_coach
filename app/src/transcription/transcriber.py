import logging
import queue
from pathlib import Path
from typing import Any, Tuple

import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import EncDecRNNTBPEModel

from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.orchestration.types import SynchronizedSegment, SynchronizedWindowSlice
from app.src.transcription.config import TranscriptionConfig
from app.src.transcription.decoder import decode_transducer_output

logger: logging.Logger = logging.getLogger(__name__)


def _ensure_local_model_weights(config: TranscriptionConfig) -> Path:
    """
    Verifies local existence of the model weights, downloading them if missing.

    Args:
        config (TranscriptionConfig): Configuration containing repo and path targets.

    Returns:
        Path: The absolute path to the local .nemo model file.

    Raises:
        RuntimeError: If the download fails due to network or permission issues.
    """
    config.local_weights_dir.mkdir(parents=True, exist_ok=True)
    target_path: Path = config.local_weights_dir / config.model_filename

    if target_path.exists():
        return target_path

    try:
        downloaded_path: str = hf_hub_download(
            repo_id=config.model_repo_id,
            filename=config.model_filename,
            local_dir=str(config.local_weights_dir)
        )
        return Path(downloaded_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to download Parakeet-TDT weights: {exc}") from exc


def initialize_transducer_model(config: TranscriptionConfig) -> EncDecRNNTBPEModel:
    """
    Loads the Parakeet-TDT model strictly from local disk and maps it to VRAM.

    Applies half-precision (float16) casting if configured, ensuring the model
    operates within the VRAM constraints of the target edge hardware.

    Args:
        config (TranscriptionConfig): Model and hardware configuration parameters.

    Returns:
        EncDecRNNTBPEModel: The loaded, device-mapped NeMo ASR model.

    Raises:
        RuntimeError: If model restoration or device mapping fails.
    """
    local_path: Path = _ensure_local_model_weights(config)

    try:
        model: EncDecRNNTBPEModel = EncDecRNNTBPEModel.restore_from(
            restore_path=str(local_path),
            map_location=config.device
        )

        if config.use_float16:
            model = model.half()

        model = model.to(config.device)
        model.eval()
        
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Transducer model: {exc}") from exc


def _execute_inference(
    model: EncDecRNNTBPEModel,
    audio_signal: torch.Tensor,
    device: str
) -> Tuple[torch.Tensor, ...]:
    """
    Executes the PyTorch forward pass through the Transducer network.
    Pads micro-chunks with silence to prevent STFT normalization crashes.

    Args:
        model (EncDecRNNTBPEModel): Initialized NeMo ASR model.
        audio_signal (torch.Tensor): 1D continuous waveform tensor.
        device (str): Compute device identifier.

    Returns:
        Tuple[torch.Tensor, ...]: Raw log-probabilities and duration tensors.

    Raises:
        RuntimeError: If the forward pass fails on the GPU.
    """
    min_safe_samples: int = 1600
    if audio_signal.shape[0] < min_safe_samples:
        padding = torch.zeros(
            min_safe_samples - audio_signal.shape[0], 
            dtype=audio_signal.dtype
        )
        audio_signal = torch.cat([audio_signal, padding], dim=0)

    batched_signal: torch.Tensor = audio_signal.unsqueeze(0).to(device)
    signal_length: torch.Tensor = torch.tensor([batched_signal.shape[1]], dtype=torch.long, device=device)

    try:
        with torch.no_grad():
            output: Tuple[torch.Tensor, ...] = model.forward(
                input_signal=batched_signal,
                input_signal_length=signal_length
            )
        return output
    except Exception as exc:
        raise RuntimeError(f"GPU inference failed during forward pass: {exc}") from exc


def _process_segment(
    item: SynchronizedWindowSlice,
    model: EncDecRNNTBPEModel,
    config: TranscriptionConfig
) -> SynchronizedSegment:
    """
    Executes raw acoustic inference to extract temporally aligned semantic datastructures.

    Args:
        item (SynchronizedWindowSlice): Temporally bound multimodal window slice.
        model (EncDecRNNTBPEModel): Pre-warmed NeMo Transducer model.
        config (TranscriptionConfig): Execution constraints and parameter matrix.

    Returns:
        SynchronizedSegment: The semantic payload enriched with transcribed tokens.

    Raises:
        TypeError: If the ingestion structure violates window constraints.
        RuntimeError: If acoustic inference or tensor processing fails.
    """
    if not isinstance(item, SynchronizedWindowSlice):
        raise TypeError("Input payload must be a strictly typed SynchronizedWindowSlice.")

    try:
        audio_tensor: torch.Tensor = torch.from_numpy(item.audio.data).float()

        raw_output: Tuple[torch.Tensor, ...] = _execute_inference(
            model=model,
            audio_signal=audio_tensor,
            device=config.device
        )

        decoded_segment = decode_transducer_output(
            raw_tensors=raw_output,
            segment_id=item.window_id,
            start_offset_sec=item.start_time_sec,
            is_final=item.is_terminal,
            tokenizer=model.tokenizer 
        )

        return SynchronizedSegment(
            segment_id=item.window_id,
            audio_slice_start_sec=item.start_time_sec,
            audio_slice_end_sec=item.end_time_sec,
            words=decoded_segment.words,
            visual_frame_start_idx=0,
            visual_frame_end_idx=len(item.frames) if hasattr(item, 'frames') and item.frames else 0,
            is_final=item.is_terminal
        )
    
    except Exception as exc:
        raise RuntimeError(f"ASR inference failure on segment {item.window_id}: {exc}") from exc


def run_transcription_node(
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    config: TranscriptionConfig,
    timeout_sec: float = 1.0
) -> None:
    """
    Executes the continuous asynchronous multiprocessing loop for the ASR Edge.

    Args:
        input_queue (queue.Queue): Upstream buffer supplying multimodal windows.
        output_queue (queue.Queue): Downstream buffer routing to the Semantic Node.
        config (TranscriptionConfig): Hardware and initialization matrix.
        timeout_sec (float): Polling duration for the queue lock. Defaults to 1.0.

    Raises:
        TypeError: If queue interfaces or configuration instances are malformed.
        RuntimeError: If model ignition or queue propagation collapses.
    """
    if not isinstance(input_queue, queue.Queue) or not isinstance(output_queue, queue.Queue):
        raise TypeError("Communication endpoints must strictly be queue.Queue instances.")
    if not isinstance(config, TranscriptionConfig):
        raise TypeError("Configuration must be a TranscriptionConfig instance.")

    try:
        model: EncDecRNNTBPEModel = initialize_transducer_model(config)
    except Exception as exc:
        output_queue.put(OrchestratorTerminalSentinel(error=exc))
        raise RuntimeError(f"Transcription initialization aborted: {exc}") from exc

    while True:
        try:
            item: Any = input_queue.get(timeout=timeout_sec)
        except queue.Empty:
            continue

        if isinstance(item, OrchestratorTerminalSentinel):
            try:
                output_queue.put(item, timeout=timeout_sec)
                break
            except queue.Full as exc:
                raise RuntimeError("Egress saturated during terminal cascade.") from exc

        if not isinstance(item, SynchronizedWindowSlice):
            continue

        try:
            segment: SynchronizedSegment = _process_segment(item, model, config)
            output_queue.put(segment, timeout=timeout_sec)
        except queue.Full as exc:
            raise RuntimeError("Transcription egress queue saturated.") from exc
        except Exception as exc:
            logger.error("Transcription execution failure on segment %s: %s", getattr(item, 'window_id', 'unknown'), exc)
            output_queue.put(OrchestratorTerminalSentinel(error=exc))
            raise RuntimeError(f"Transcription pipeline aborted: {exc}") from exc