import logging
from queue import Full, Queue
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch

from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.orchestration.types import SynchronizedWindowSlice
from app.src.visual_edge.facial_gaze.config import FacialConfig
from app.src.visual_edge.facial_gaze.generator import generate_facial_descriptor
from app.src.visual_edge.facial_gaze.poster import POSTERPlus
from app.src.visual_edge.facial_gaze.types import FacialDescriptorPayload
from app.src.visual_edge.preprocessor import process_facial_frames

logger: logging.Logger = logging.getLogger(__name__)


def _initialize_facial_models(config: FacialConfig) -> Tuple[POSTERPlus, Any, torch.device]:
    """
    Allocates and validates the neural inference matrices for spatial processing.

    Args:
        config (FacialConfig): The fully validated configuration matrix.

    Returns:
        Tuple[POSTERPlus, Any, torch.device]: The visual encoder, scikit-learn regressor, and hardware binding.

    Raises:
        RuntimeError: If neural initialization, hardware binding, or regressor deserialization fails.
    """
    try:
        device: torch.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        visual_encoder = POSTERPlus(feature_dim=config.feature_dim)
        state_dict: Dict[str, Any] = torch.load(
            config.poster_weights_path, 
            map_location=device, 
            weights_only=True
        )
        visual_encoder.load_state_dict(state_dict)
        visual_encoder = visual_encoder.to(device)
        visual_encoder.eval()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize POSTER++ visual encoder on device {config.device}: {exc}") from exc

    try:
        regressor: Any = joblib.load(config.regressor_weights_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to deserialize regression artifact from {config.regressor_weights_path}: {exc}") from exc

    return visual_encoder, regressor, device


def _process_segment(
    item: SynchronizedWindowSlice,
    visual_encoder: POSTERPlus,
    regressor: Any,
    device: torch.device,
    config: FacialConfig,
    latent_cache: Dict[int, torch.Tensor]
) -> FacialDescriptorPayload:
    """
    Executes inference compression utilizing a frame-level cache to eliminate redundant spatial calculations.

    Args:
        item (SynchronizedWindowSlice): The aligned multimodal window from the orchestrator.
        visual_encoder (POSTERPlus): The pre-loaded spatial feature extractor.
        regressor (Any): The pre-loaded sklearn behavioral mapper.
        device (torch.device): The bound hardware accelerator.
        config (FacialConfig): The active threshold configuration.
        latent_cache (Dict[int, torch.Tensor]): Mutable dictionary storing previously computed frame tensors.

    Returns:
        FacialDescriptorPayload: The immutable transport object containing raw segment telemetry.

    Raises:
        TypeError: If the input is not a valid SynchronizedWindowSlice.
        ValueError: If the segment contains an empty frame matrix or latent bounds mismatch.
        RuntimeError: If the multi-stage compression fails mathematically.
    """
    if not isinstance(item, SynchronizedWindowSlice):
        raise TypeError("Input must be a strictly typed SynchronizedWindowSlice.")

    try:
        if not item.frames:
            raise ValueError("Segment contains an empty frame sequence.")

        unseen_frames = [frame for frame in item.frames if frame.frame_index not in latent_cache]

        if unseen_frames:
            raw_unseen: np.ndarray = np.stack([frame.data for frame in unseen_frames])
            tensor_gpu: torch.Tensor = process_facial_frames(frames=raw_unseen, device=device)

            with torch.inference_mode():
                new_latents: torch.Tensor = visual_encoder(tensor_gpu)
                if new_latents.dim() == 1:
                    new_latents = new_latents.unsqueeze(0)
                    
            for i, frame in enumerate(unseen_frames):
                latent_cache[frame.frame_index] = new_latents[i]

        window_latents: torch.Tensor = torch.stack([latent_cache[frame.frame_index] for frame in item.frames])

        if window_latents.dim() > 1 and window_latents.shape[0] > 1:
            pooled_latent: torch.Tensor = window_latents.mean(dim=0, keepdim=True)
        else:
            pooled_latent = window_latents

        if pooled_latent.shape[-1] != config.feature_dim:
            raise ValueError(f"Latent dimension mismatch. Expected {config.feature_dim}, received {pooled_latent.shape[-1]}.")

        latent_np: np.ndarray = pooled_latent.cpu().numpy()
        
        if latent_np.ndim == 1:
            latent_np = latent_np.reshape(1, -1)

        predictions: np.ndarray = regressor.predict(latent_np)
        
        ocean_scores: Dict[str, float] = {
            "extraversion": float(predictions[0, 0]),
            "neuroticism": float(predictions[0, 1]),
            "agreeableness": float(predictions[0, 2]),
            "conscientiousness": float(predictions[0, 3]),
            "openness": float(predictions[0, 4])
        }

        descriptor_text: str = generate_facial_descriptor(
            scores=ocean_scores,
            calibration_thresholds=config.calibration_thresholds
        )

        return FacialDescriptorPayload(
            segment_id=str(item.window_id),
            start_time_sec=item.start_time_sec,
            end_time_sec=item.end_time_sec,
            behavioral_descriptor=descriptor_text,
            extraversion_score=ocean_scores["extraversion"],
            neuroticism_score=ocean_scores["neuroticism"],
            agreeableness_score=ocean_scores["agreeableness"],
            conscientiousness_score=ocean_scores["conscientiousness"],
            openness_score=ocean_scores["openness"]
        )

    except Exception as exc:
        segment_id: str = str(getattr(item, 'window_id', 'unknown'))
        raise RuntimeError(f"Pipeline execution failed for segment {segment_id}: {exc}") from exc


def _aggregate_payloads(
    payloads: List[FacialDescriptorPayload],
    config: FacialConfig
) -> FacialDescriptorPayload:
    """
    Reduces a sequence of facial segment payloads into a singular statistical average matrix.

    Args:
        payloads (List[FacialDescriptorPayload]): Accumulated spatial inference payloads.
        config (FacialConfig): Configuration block dictating calibration descriptor thresholds.

    Returns:
        FacialDescriptorPayload: Unified terminal payload containing global statistical averages.

    Raises:
        ValueError: If the payload list is explicitly empty.
    """
    if not payloads:
        raise ValueError("Aggregation requires a strictly populated sequence of payloads.")

    ext_mean: float = float(np.mean([p.extraversion_score for p in payloads]))
    neu_mean: float = float(np.mean([p.neuroticism_score for p in payloads]))
    agr_mean: float = float(np.mean([p.agreeableness_score for p in payloads]))
    con_mean: float = float(np.mean([p.conscientiousness_score for p in payloads]))
    ope_mean: float = float(np.mean([p.openness_score for p in payloads]))

    ocean_scores: Dict[str, float] = {
        "extraversion": ext_mean,
        "neuroticism": neu_mean,
        "agreeableness": agr_mean,
        "conscientiousness": con_mean,
        "openness": ope_mean
    }

    descriptor_text: str = generate_facial_descriptor(
        scores=ocean_scores,
        calibration_thresholds=config.calibration_thresholds
    )

    return FacialDescriptorPayload(
        segment_id="0",
        start_time_sec=payloads[0].start_time_sec,
        end_time_sec=payloads[-1].end_time_sec,
        behavioral_descriptor=descriptor_text,
        extraversion_score=ext_mean,
        neuroticism_score=neu_mean,
        agreeableness_score=agr_mean,
        conscientiousness_score=con_mean,
        openness_score=ope_mean
    )


def run_facial_node(
    input_queue: Queue,
    output_queue: Queue,
    config: FacialConfig,
    timeout_sec: float = 5.0
) -> None:
    """
    Executes the continuous Facial Edge consumer loop using a Stream-Reduce aggregation pattern.

    Args:
        input_queue (Queue): Ingress queue supplying SynchronizedWindowSlice objects.
        output_queue (Queue): Egress queue receiving the unified FacialDescriptorPayload.
        config (FacialConfig): Hardware, model, and threshold configuration matrix.
        timeout_sec (float): Block timeout in seconds for queue retrieval and insertion.

    Raises:
        TypeError: If queue or configuration parameters violate expected types.
        RuntimeError: If model initialization fails or downstream queue saturates.
    """
    if not isinstance(input_queue, Queue):
        raise TypeError("input_queue must be an instance of queue.Queue.")
    if not isinstance(output_queue, Queue):
        raise TypeError("output_queue must be an instance of queue.Queue.")
    if not isinstance(config, FacialConfig):
        raise TypeError("config must be an instance of FacialConfig.")

    try:
        visual_encoder, regressor, device = _initialize_facial_models(config)
    except Exception as exc:
        output_queue.put(OrchestratorTerminalSentinel(error=exc))
        raise RuntimeError(f"Facial node initialization failed: {exc}") from exc

    history_buffer: List[FacialDescriptorPayload] = []
    latent_cache: Dict[int, torch.Tensor] = {}

    while True:
        try:
            item: Any = input_queue.get(timeout=timeout_sec)
        except Exception:
            continue

        if isinstance(item, OrchestratorTerminalSentinel):
            if history_buffer:
                try:
                    final_payload: FacialDescriptorPayload = _aggregate_payloads(history_buffer, config)
                    output_queue.put(final_payload, timeout=timeout_sec)
                    logger.info("[FACIAL SUCCESS] Emitted global aggregated payload to Fusion Nexus.")
                except Full as exc:
                    raise RuntimeError(f"Facial aggregation failure: queue saturated. {exc}") from exc
                except Exception as exc:
                    try:
                        output_queue.put(OrchestratorTerminalSentinel(error=exc), timeout=timeout_sec)
                    except Full:
                        pass 
                    raise RuntimeError(f"Facial aggregation failure: {exc}") from exc

            try:
                output_queue.put(item, timeout=timeout_sec)
                history_buffer.clear()
                latent_cache.clear()
                continue
            except Full as exc:
                raise RuntimeError("Output queue saturated during termination sequence.") from exc

        if not isinstance(item, SynchronizedWindowSlice):
            continue

        try:
            payload: FacialDescriptorPayload = _process_segment(
                item=item,
                visual_encoder=visual_encoder,
                regressor=regressor,
                device=device,
                config=config,
                latent_cache=latent_cache
            )
            history_buffer.append(payload)
            logger.info("[FACIAL STREAM] Processed segment %s into aggregation buffer.", payload.segment_id)
        except Exception as exc:
            try:
                output_queue.put(OrchestratorTerminalSentinel(error=exc), timeout=timeout_sec)
            except Full:
                pass
            logger.error("[FACIAL ERROR] Processing failed: %s", exc)
            raise RuntimeError(f"Facial processing failure: {exc}") from exc