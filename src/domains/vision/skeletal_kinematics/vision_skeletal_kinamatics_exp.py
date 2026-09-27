import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.core.registry import register_domain
from src.domains.vision.adapters.base_adapter import PredictTrajectoryFn
from src.domains.vision.video_preprocessor import (
    create_face_detector,
    create_vision_transform,
    process_video_pipeline,
)
from src.domains.vision.skeletal_kinematics.one_euro_filter import calculate_kinematic_metrics
from src.domains.vision.skeletal_kinematics.adapters.mediapipe_gcn import create_mediapipe_gcn_adapter
from src.domains.vision.skeletal_kinematics.adapters.human_omni import create_human_omni_adapter

logger: logging.Logger = logging.getLogger(__name__)


def resolve_device(config_device: str, global_device: torch.device) -> torch.device:
    """
    Determines the execution hardware device based on configuration or fallbacks.

    Args:
        config_device (str): Device identifier string from configuration.
        global_device (torch.device): Fallback device from CLI.

    Returns:
        torch.device: Resolved PyTorch hardware device.
    """
    try:
        config_lower: str = config_device.lower()
        if config_lower in ["cuda", "cpu"]:
            resolved: torch.device = torch.device(config_lower)
            if config_lower == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but unavailable. Falling back to CPU.")
                return torch.device("cpu")
            return resolved
        return global_device
    except Exception as exc:
        logger.error("Failed to resolve target device: %s", exc)
        return global_device


def process_single_video(
    video_path: str,
    detector: Any,
    transform: Any,
    predict_traj_fn: PredictTrajectoryFn,
) -> Dict[str, Any]:
    """
    Executes the skeletal transformation and spatial inference pipeline on a single video.

    Args:
        video_path (str): Target video path.
        detector (Any): Face detection model.
        transform (Any): Tensor transformation pipeline.
        predict_traj_fn (PredictTrajectoryFn): Closure for predicting kinematic trajectories.

    Returns:
        Dict[str, Any]: Mapping of video path to kinematic metrics and execution status.
    """
    tensor_sequence: Optional[torch.Tensor]
    fps: float
    tensor_sequence, fps = process_video_pipeline(video_path, detector, transform)

    if tensor_sequence is None:
        return {
            "video_path": video_path,
            "perceptual_jitter": 0.0,
            "temporal_smoothness": 0.0,
            "status": "FAILED_PREPROCESSING"
        }

    trajectory: Optional[np.ndarray] = predict_traj_fn(tensor_sequence, fps)
    
    if trajectory is None:
        return {
            "video_path": video_path,
            "perceptual_jitter": 0.0,
            "temporal_smoothness": 0.0,
            "status": "FAILED_INFERENCE"
        }
        
    metrics: Dict[str, float] = calculate_kinematic_metrics(
        raw_trajectory=trajectory, 
        fps=fps
    )
    
    return {
        "video_path": video_path,
        "perceptual_jitter": metrics.get("perceptual_jitter", 0.0),
        "temporal_smoothness": metrics.get("temporal_smoothness", 0.0),
        "status": "SUCCESS"
    }


def _append_to_checkpoint(results_batch: List[Dict[str, Any]], checkpoint_path: Path, write_header: bool) -> None:
    """
    Appends intermediate metrics to disk to prevent data loss and memory bloat.

    Args:
        results_batch (List[Dict[str, Any]]): Batch of processed video metrics.
        checkpoint_path (Path): Path for checkpoint CSV.
        write_header (bool): Whether to write CSV column headers.
    """
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode ('a')
        pd.DataFrame(results_batch).to_csv(checkpoint_path, mode='a', header=write_header, index=False)
        logger.info("Appended %d records to checkpoint %s", len(results_batch), checkpoint_path)
    except Exception as exc:
        logger.error("Failed to append checkpoint to %s: %s", checkpoint_path, exc)


@register_domain("skeletal_kinematics")
def run_skeletal_kinematics_experiment(config: Dict[str, Any], global_device: torch.device) -> None:
    """
    Orchestrates the skeletal kinematics evaluation pipeline.

    Args:
        config (Dict[str, Any]): Experiment configuration parameters.
        global_device (torch.device): Default hardware target.

    Raises:
        FileNotFoundError: Triggered when benchmark manifest is missing.
        ValueError: Triggered when manifest lacks 'video_path' or active model is undefined.
    """
    device: torch.device = resolve_device(config.get("device", ""), global_device)
    logger.info("Initializing skeletal kinematics experiment on device: %s", device)

    model_config: Dict[str, Any] = config.get("model_config", {})
    data_config: Dict[str, Any] = config.get("data_config", {})
    eval_config: Dict[str, Any] = config.get("evaluation", {})

    manifest_path: Path = Path(data_config.get("manifest_path", "src/data/vision/manifest.csv"))
    if not manifest_path.exists():
        logger.error("Benchmark manifest not found: %s", manifest_path)
        raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")

    manifest_df: pd.DataFrame = pd.read_csv(manifest_path)
    if "video_path" not in manifest_df.columns:
        raise ValueError("Manifest must contain a 'video_path' column.")

    video_paths: List[str] = manifest_df["video_path"].tolist()
    
    detector: Any = create_face_detector(device=device)
    transform: Any = create_vision_transform()
    
    active_model: str = model_config.get("active_model", "mediapipe_gcn")
    predict_traj_fn: PredictTrajectoryFn
    
    if active_model == "human_omni":
        predict_traj_fn = create_human_omni_adapter(model_config, device)
    else:
        predict_traj_fn = create_mediapipe_gcn_adapter(model_config, device)

    output_path: Path = Path(eval_config.get("output_results_path", f"src/results/vision/skeletal_{active_model}_results.csv"))
    checkpoint_path: Path = output_path.with_name(f"{output_path.stem}_checkpoint.csv")
    checkpoint_interval: int = eval_config.get("checkpoint_interval", 100)

    # --- 1. RESUME LOGIC ---
    processed_videos = set()
    write_header = True
    
    if checkpoint_path.exists():
        try:
            existing_df = pd.read_csv(checkpoint_path)
            if "video_path" in existing_df.columns:
                processed_videos = set(existing_df["video_path"].tolist())
                write_header = False
                logger.info("Resuming from checkpoint. %d videos already processed.", len(processed_videos))
        except Exception as e:
            logger.warning("Could not read existing checkpoint. Starting fresh. Error: %s", e)

    # Filter out videos that are already in the checkpoint
    video_paths_to_process = [vp for vp in video_paths if vp not in processed_videos]
    logger.info("Videos remaining to process: %d", len(video_paths_to_process))

    # --- 2. BATCH PROCESSING (MEMORY OPTIMIZATION) ---
    batch_results: List[Dict[str, Any]] = []
    
    idx: int
    video_path: str
    for idx, video_path in enumerate(video_paths_to_process):
        result_dict: Dict[str, Any] = process_single_video(
            video_path=video_path,
            detector=detector,
            transform=transform,
            predict_traj_fn=predict_traj_fn,
        )
        batch_results.append(result_dict)
        logger.info("Processed [%d/%d] %s | Status: %s", idx + 1, len(video_paths_to_process), video_path, result_dict["status"])

        # Flush to disk and clear from RAM
        if (idx + 1) % checkpoint_interval == 0:
            _append_to_checkpoint(batch_results, checkpoint_path, write_header)
            write_header = False  # Ensure header is only written once
            batch_results.clear()

    # Flush any remaining items in the batch
    if batch_results:
        _append_to_checkpoint(batch_results, checkpoint_path, write_header)
        batch_results.clear()

    # --- 3. FINALIZE ---
    if not checkpoint_path.exists():
        logger.warning("No processing was completed.")
        return

    # Load the complete dataset from disk once at the very end to calculate metrics
    final_results_df = pd.read_csv(checkpoint_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_results_df.to_csv(output_path, index=False)
    
    # if checkpoint_path.exists():
    #     checkpoint_path.unlink()
    
    successful_results: pd.DataFrame = final_results_df[final_results_df["status"] == "SUCCESS"]
    mean_jitter: float = float(successful_results["perceptual_jitter"].mean()) if not successful_results.empty else 0.0
    mean_smoothness: float = float(successful_results["temporal_smoothness"].mean()) if not successful_results.empty else 0.0

    metrics_path: Path = output_path.with_name(f"{output_path.stem}_metrics.json")
    metrics_summary: Dict[str, Any] = {
        "model_architecture": active_model,
        "total_samples": len(video_paths),
        "successful_samples": len(successful_results),
        "mean_perceptual_jitter": mean_jitter,
        "mean_temporal_smoothness": mean_smoothness
    }
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)