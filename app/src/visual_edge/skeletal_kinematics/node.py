import logging
import queue
from typing import Any, Dict, List

import numpy as np
import torch

from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.orchestration.types import SynchronizedWindowSlice
from app.src.visual_edge.preprocessor import process_skeletal_frames
from app.src.visual_edge.skeletal_kinematics.config import SkeletalConfig
from app.src.visual_edge.skeletal_kinematics.generator import generate_behavioral_descriptor
from app.src.visual_edge.skeletal_kinematics.omni import HumanOmni
from app.src.visual_edge.skeletal_kinematics.one_euro_filter import calculate_kinematic_metrics
from app.src.visual_edge.skeletal_kinematics.types import SkeletalDescriptorPayload

logger: logging.Logger = logging.getLogger(__name__)


def _process_segment(
    item: SynchronizedWindowSlice,
    model: HumanOmni,
    config: SkeletalConfig,
    device: torch.device
) -> SkeletalDescriptorPayload:
    """
    Extracts tri-branch kinematics and generates a BeMERC behavioral descriptor for a single window.

    Args:
        item (SynchronizedWindowSlice): The synchronized temporal video window payload.
        model (HumanOmni): The internally initialized neural architecture.
        config (SkeletalConfig): The active configuration matrix governing translations.
        device (torch.device): The target compute device.

    Returns:
        SkeletalDescriptorPayload: The immutable transport object containing the generated text.

    Raises:
        ValueError: If the segment contains an empty frame matrix.
        RuntimeError: If data conversion, inference, or translation fails.
    """
    try:
        if not item.frames:
            raise ValueError("Segment contains an empty frame sequence.")

        raw_frames: np.ndarray = np.stack([frame.data for frame in item.frames])
        video_tensor: torch.Tensor = process_skeletal_frames(frames=raw_frames, device=device)
        
        with torch.no_grad():
            trajectory_tensor: torch.Tensor = model(video_tensor)
            
        trajectory: np.ndarray = trajectory_tensor.cpu().numpy()
        
        duration: float = item.end_time_sec - item.start_time_sec
        fps_calculation: float = len(item.frames) / duration if duration > 0.0 else 50.0
            
        metrics: Dict[str, float] = calculate_kinematic_metrics(
            raw_trajectory=trajectory,
            fps=fps_calculation
        )
        
        jitter: float = metrics["perceptual_jitter"]
        smoothness: float = metrics["temporal_smoothness"]
        
        descriptor: str = generate_behavioral_descriptor(
            perceptual_jitter=jitter,
            temporal_smoothness=smoothness,
            config=config
        )

        return SkeletalDescriptorPayload(
            segment_id=item.window_id,
            start_time_sec=item.start_time_sec,
            end_time_sec=item.end_time_sec,
            behavioral_descriptor=descriptor,
            perceptual_jitter=jitter,
            temporal_smoothness=smoothness
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to process skeletal segment {item.window_id}: {exc}") from exc


def _aggregate_payloads(
    payloads: List[SkeletalDescriptorPayload],
    config: SkeletalConfig
) -> SkeletalDescriptorPayload:
    """
    Reduces a sequence of skeletal segment payloads into a singular statistical average matrix.

    Args:
        payloads (List[SkeletalDescriptorPayload]): Accumulated spatial inference payloads.
        config (SkeletalConfig): Configuration block dictating calibration descriptor thresholds.

    Returns:
        SkeletalDescriptorPayload: Unified terminal payload containing global statistical averages.

    Raises:
        ValueError: If the payload list is explicitly empty.
    """
    if not payloads:
        raise ValueError("Aggregation requires a strictly populated sequence of payloads.")

    jitter_mean: float = float(np.mean([p.perceptual_jitter for p in payloads]))
    smoothness_mean: float = float(np.mean([p.temporal_smoothness for p in payloads]))

    descriptor_text: str = generate_behavioral_descriptor(
        perceptual_jitter=jitter_mean,
        temporal_smoothness=smoothness_mean,
        config=config
    )

    return SkeletalDescriptorPayload(
        segment_id=0,
        start_time_sec=payloads[0].start_time_sec,
        end_time_sec=payloads[-1].end_time_sec,
        behavioral_descriptor=descriptor_text,
        perceptual_jitter=jitter_mean,
        temporal_smoothness=smoothness_mean
    )


def run_skeletal_node(
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    config: SkeletalConfig,
    timeout_sec: float = 1.0
) -> None:
    """
    Executes the isolated multiprocessing loop for the Skeletal Kinematics pipeline using Stream-Reduce.

    Args:
        input_queue (queue.Queue): The upstream buffer supplying synchronized windows.
        output_queue (queue.Queue): The downstream buffer traversing to the Late Fusion Nexus.
        config (SkeletalConfig): The matrix governing pipeline execution and translation.
        timeout_sec (float): Polling duration for the queue lock. Defaults to 1.0.

    Raises:
        TypeError: If queue interfaces or configuration matrices are structurally invalid.
        RuntimeError: If node initialization fails or downstream buffers saturate.
    """
    if not isinstance(input_queue, queue.Queue) or not isinstance(output_queue, queue.Queue):
        raise TypeError("Queues must be instances of queue.Queue")

    try:
        device: torch.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model = HumanOmni(feature_dim=config.feature_dim, num_joints=config.num_joints)
        
        state_dict: Dict[str, Any] = torch.load(
            config.weights_path, 
            map_location=device, 
            weights_only=True
        )
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
    except Exception as exc:
        output_queue.put(OrchestratorTerminalSentinel())
        raise RuntimeError(f"Skeletal node initialization failed: {exc}") from exc

    history_buffer: List[SkeletalDescriptorPayload] = []

    while True:
        try:
            item: Any = input_queue.get(timeout=timeout_sec)
        except queue.Empty:
            continue

        if isinstance(item, OrchestratorTerminalSentinel):
            if history_buffer:
                try:
                    final_payload: SkeletalDescriptorPayload = _aggregate_payloads(history_buffer, config)
                    output_queue.put(final_payload, timeout=timeout_sec)
                    logger.info("[SKELETAL SUCCESS] Emitted global aggregated payload to Fusion Nexus.")
                except queue.Full as exc:
                    raise RuntimeError(f"Skeletal aggregation failure: queue saturated. {exc}") from exc
                except Exception as exc:
                    try:
                        output_queue.put(OrchestratorTerminalSentinel(error=exc), timeout=timeout_sec)
                    except queue.Full:
                        pass # Ignore secondary saturation crash
                    raise RuntimeError(f"Skeletal aggregation failure: {exc}") from exc
                    
            try:
                output_queue.put(item, timeout=timeout_sec)
                history_buffer.clear()
                continue
            except queue.Full as exc:
                raise RuntimeError("Output queue saturated during termination.") from exc

        if not isinstance(item, SynchronizedWindowSlice):
            continue

        try:
            payload: SkeletalDescriptorPayload = _process_segment(item, model, config, device)
            history_buffer.append(payload)
            logger.info("[SKELETAL STREAM] Processed segment %s into aggregation buffer.", payload.segment_id)
        except Exception as exc:
            logger.error("Processing failure on segment %s: %s", getattr(item, 'window_id', 'unknown'), exc)
            output_queue.put(OrchestratorTerminalSentinel(error=exc), timeout=timeout_sec)
            raise RuntimeError(f"Skeletal processing failure: {exc}") from exc