import ast
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch

from src.core.registry import register_domain

from src.domains.vision.adapters.base_adapter import PredictTrajectoryFn
from src.domains.vision.facial_and_gaze.adapters.poster_plus import create_poster_adapter
from src.domains.vision.facial_and_gaze.adapters.gaze_360 import create_gaze360_adapter
from src.domains.vision.facial_and_gaze.adapters.gazetr import create_gazetr_adapter
from src.domains.vision.video_preprocessor import (
    create_face_detector,
    create_vision_transform,
    process_video_pipeline
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Adapter Registry
# ---------------------------------------------------------
# Maps the config architecture name to the respective factory function
ADAPTER_REGISTRY: Dict[str, Callable[[Dict[str, Any], torch.device], PredictTrajectoryFn]] = {
    "POSTER++": create_poster_adapter,
    "Gaze360": create_gaze360_adapter,
    "GazeTR-Hybrid": create_gazetr_adapter, # Uncomment once implemented
}


def _append_to_checkpoint(results: List[Dict[str, Any]], checkpoint_path: Path) -> None:
    """
    Appends a batch of intermediate metrics to disk to prevent data loss.

    Args:
        results (List[Dict[str, Any]]): Batch of processed video metrics.
        checkpoint_path (Path): Path for the checkpoint CSV.
    """
    if not results:
        return

    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        df: pd.DataFrame = pd.DataFrame(results)
        
        if checkpoint_path.exists():
            df.to_csv(checkpoint_path, mode="a", header=False, index=False)
        else:
            df.to_csv(checkpoint_path, mode="w", header=True, index=False)
            
        logger.info("Appended %d records to checkpoint %s", len(results), checkpoint_path)
    except Exception as exc:
        logger.error("Failed to append checkpoint to %s: %s", checkpoint_path, exc)


@register_domain("facial_and_gaze")
def run_facial_and_gaze_experiment(config: Dict[str, Any], global_device: torch.device) -> None:
    """
    Orchestrates the facial affective and gaze evaluation pipeline dynamically.

    Args:
        config (Dict[str, Any]): Experiment configuration parameters.
        global_device (torch.device): Default hardware target.

    Raises:
        FileNotFoundError: Triggered when benchmark manifest is missing.
        ValueError: Triggered when manifest lacks 'video_path' column or architecture is unsupported.
        RuntimeError: Triggered upon critical failure during compilation.
    """
    device_str: str = config.get("device", "")
    device: torch.device = torch.device(device_str) if device_str else global_device
    logger.info("Initializing facial and gaze experiment on device: %s", device)

    model_config: Dict[str, Any] = config.get("model_config", {})
    data_config: Dict[str, Any] = config.get("data_config", {})
    eval_config: Dict[str, Any] = config.get("evaluation", {})

    architecture_name: str = model_config.get("architecture_name", "UNKNOWN_MODEL")
    if architecture_name not in ADAPTER_REGISTRY:
        raise ValueError(f"Unsupported architecture_name '{architecture_name}'. Available options: {list(ADAPTER_REGISTRY.keys())}")

    manifest_path: Path = Path(data_config.get("manifest_path", "src/data/vision/manifest.csv"))
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")

    manifest_df: pd.DataFrame = pd.read_csv(manifest_path)
    if "video_path" not in manifest_df.columns:
        raise ValueError("Manifest must contain a 'video_path' column.")

    video_paths: List[str] = manifest_df["video_path"].tolist()
    total_videos: int = len(video_paths)
    
    detector: Any = create_face_detector(device=device, margin=20)
    transform: Any = create_vision_transform(target_size=(224, 224))
    
    # Dynamically select and initialize the adapter from the registry
    adapter_factory = ADAPTER_REGISTRY[architecture_name]
    predict_fn: PredictTrajectoryFn = adapter_factory(model_config, device)
    logger.info("Successfully initialized %s adapter.", architecture_name)

    output_path: Path = Path(eval_config.get("output_results_path", f"src/results/vision/{architecture_name.lower()}_results.csv"))
    checkpoint_path: Path = output_path.with_name(f"{output_path.stem}_checkpoint.csv")
    checkpoint_interval: int = eval_config.get("checkpoint_interval", 100)

    processed_videos: Set[str] = set()

    if checkpoint_path.exists():
        try:
            checkpoint_df: pd.DataFrame = pd.read_csv(checkpoint_path, usecols=["video_path"])
            if not checkpoint_df.empty:
                processed_videos = set(checkpoint_df["video_path"].tolist())
                logger.info("Resuming experiment. %d/%d videos already processed.", len(processed_videos), total_videos)
        except Exception as exc:
            logger.warning("Checkpoint parsing issue, starting fresh: %s", exc)

    pending_videos: List[str] = [p for p in video_paths if p not in processed_videos]
    batch_results: List[Dict[str, Any]] = []
    
    for idx, video_path in enumerate(pending_videos):
        result_dict: Dict[str, Any] = {
            "video_path": video_path,
            "status": "FAILED",
            "latent_embeddings": None,
            "fps": 0.0,
            "error_message": ""
        }
        
        try:
            tensor_sequence: Optional[torch.Tensor]
            fps: float
            tensor_sequence, fps = process_video_pipeline(
                video_path=video_path,
                detector=detector,
                transform=transform
            )
            
            if tensor_sequence is not None:
                latents: Optional[np.ndarray] = predict_fn(tensor_sequence, fps)
                
                if latents is not None:
                    result_dict["latent_embeddings"] = latents.tolist()
                    result_dict["fps"] = fps
                    result_dict["status"] = "SUCCESS"
                else:
                    result_dict["error_message"] = "Adapter returned None."
            else:
                result_dict["error_message"] = "Preprocessor failed to extract facial sequence."
                
        except Exception as exc:
            result_dict["error_message"] = str(exc)
            
        batch_results.append(result_dict)
        
        absolute_idx: int = len(processed_videos) + idx + 1
        logger.info(
            "Processed [%d/%d] | Status: %s | Video: %s", 
            absolute_idx, 
            total_videos, 
            result_dict["status"], 
            Path(video_path).name
        )
        
        if absolute_idx % checkpoint_interval == 0:
            _append_to_checkpoint(batch_results, checkpoint_path)
            batch_results.clear()

    if batch_results:
        _append_to_checkpoint(batch_results, checkpoint_path)
        batch_results.clear()

    try:
        final_df: pd.DataFrame = pd.read_csv(checkpoint_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            
        successful_results: pd.DataFrame = final_df[final_df["status"] == "SUCCESS"]

        success_rate: float = (len(successful_results) / total_videos) if total_videos else 0.0
        domain_shift_score: float = 0.0
        is_feature_stable: bool = False
        
        if not successful_results.empty and "latent_embeddings" in successful_results.columns:
            embeddings_list: List[Any] = successful_results["latent_embeddings"].apply(ast.literal_eval).tolist()
            embeddings_arr: np.ndarray = np.array(embeddings_list)
            
            if len(embeddings_arr) > 1:
                frame_variance: np.ndarray = np.var(embeddings_arr, axis=0)
                domain_shift_score = float(np.mean(frame_variance))
                is_feature_stable = domain_shift_score > 1e-5

        suitability_status: str
        if success_rate == 1.0 and is_feature_stable:
            suitability_status = "VALIDATED_FOR_LATE_FUSION"
        elif success_rate > 0.0:
            suitability_status = "PARTIAL_SUCCESS_WARNING"
        else:
            suitability_status = "FAILED_VALIDATION"

        param_count: float = float(model_config.get("parameter_count_millions", 0.0))
        gflops: float = float(model_config.get("computational_load_gflops", 0.0))

        metrics_path: Path = output_path.with_name(f"{output_path.stem}_metrics.json")
        metrics_summary: Dict[str, Any] = {
            "model_architecture": architecture_name,
            "parameter_count_millions": param_count,
            "computational_load_gflops": gflops,
            "total_samples": total_videos,
            "successful_samples": len(successful_results),
            "success_rate": success_rate,
            "latent_feature_stability_score": domain_shift_score,
            "model_suitability_status": suitability_status
        }
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=4)
            
        logger.info("Experiment finalized successfully. Results saved to %s", output_path)
            
    except Exception as exc:
        logger.error("Failed to compile final dataset: %s", exc)
        raise RuntimeError(f"Failed to finalize experiment results: {exc}")