import json
import logging
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from src.domains.acoustic.feature_extraction import extract_latents_batch, initialize_acoustic_model, resolve_device
from src.domains.acoustic.metrics_evaluation import compute_real_time_factor, compute_ridge_cv_correlations
from src.core.registry import register_domain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _chunk_iterable(iterable: List[Any], size: int) -> Iterator[List[Any]]:
    """
    Yields bounded sequential subsets from an iterable.

    Args:
        iterable (List[Any]): The source sequence to partition.
        size (int): The maximum length of each generated subset.

    Yields:
        Iterator[List[Any]]: A generator producing partitioned subsets.
    """
    try:
        iterator: Iterator[Any] = iter(iterable)
        for first in iterator:
            yield [first] + list(islice(iterator, size - 1))
    except Exception as e:
        logger.error("Failed to safely chunk iterable: %s", e)
        yield []


def load_checkpoint_map(checkpoint_path: Path) -> Dict[str, List[float]]:
    """
    Reads existing latent feature vectors from persistent storage to enable process resumption.

    Args:
        checkpoint_path (Path): The designated file path for the state map.

    Returns:
        Dict[str, List[float]]: A mapping of file paths to their corresponding extracted latent vectors.
    """
    try:
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data: Dict[str, List[float]] = json.load(f)
                logger.info("Resuming from checkpoint. Found %d existing records.", len(data))
                return data
        return {}
    except Exception as e:
        logger.error("Checkpoint file is corrupted or unreadable. Starting fresh. Error: %s", e)
        return {}


def save_checkpoint_map(checkpoint_map: Dict[str, List[float]], checkpoint_path: Path) -> None:
    """
    Executes an atomic write operation to persist the pipeline state map.

    Args:
        checkpoint_map (Dict[str, List[float]]): The current state mapping of paths to latent vectors.
        checkpoint_path (Path): The target destination file path.
    """
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path = checkpoint_path.with_suffix('.tmp')
        
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_map, f, indent=2, ensure_ascii=False)
            
        temp_path.replace(checkpoint_path)
    except Exception as e:
        logger.error("Failed to persist checkpoint map atomically: %s", e)


def process_acoustic_batches(
    pending_tasks: List[Tuple[str, float]],
    checkpoint_map: Dict[str, List[float]],
    checkpoint_path: Path,
    model: Any,
    extractor: Any,
    framework: str,
    target_sr: int,
    device: torch.device,
    batch_size: int
) -> Tuple[float, float]:
    """
    Processes audio files in batches, extracts full latent vectors, and aggressively checkpoints progress.

    Args:
        pending_tasks (List[Tuple[str, float]]): Audio paths and durations that have not yet been processed.
        checkpoint_map (Dict[str, List[float]]): The current state of processed files.
        checkpoint_path (Path): File system path to save the JSON checkpoint.
        model (Any): The instantiated acoustic model.
        extractor (Any): The feature extractor (if applicable).
        framework (str): The framework identifier.
        target_sr (int): Target sampling rate.
        device (torch.device): Compute device.
        batch_size (int): Number of files to process simultaneously.

    Returns:
        Tuple[float, float]: Total inference time and total valid duration processed.

    Raises:
        RuntimeError: If batch processing or checkpoint saving fails critically.
    """
    total_inference_time: float = 0.0
    valid_duration_processed: float = 0.0
    total_batches: int = (len(pending_tasks) + batch_size - 1) // batch_size

    try:
        chunk: List[Tuple[str, float]]
        # Wrapped the iterable in tqdm for a CLI progress bar
        for chunk in tqdm(_chunk_iterable(pending_tasks, batch_size), total=total_batches, desc="Extracting Audio Latents"):
            paths_chunk: Tuple[str, ...]
            dur_chunk: Tuple[float, ...]
            paths_chunk, dur_chunk = zip(*chunk)
            
            latent_vectors: List[List[float]]
            inf_time: float
            latent_vectors, inf_time = extract_latents_batch(
                audio_paths=list(paths_chunk),
                model=model,
                extractor=extractor,
                framework=framework,
                target_sr=target_sr,
                device=device
            )
            
            total_inference_time += inf_time
            
            p: str
            vec: List[float]
            d: float
            for p, vec, d in zip(paths_chunk, latent_vectors, dur_chunk):
                checkpoint_map[p] = vec
                valid_duration_processed += float(d)
                
            save_checkpoint_map(checkpoint_map, checkpoint_path)
            
        return total_inference_time, valid_duration_processed

    except Exception as e:
        logger.error("Critical failure during batch processing loop: %s", e)
        save_checkpoint_map(checkpoint_map, checkpoint_path)
        raise RuntimeError(f"Batch processing failed: {e}") from e


def finalize_evaluation_metrics(
    df: pd.DataFrame,
    checkpoint_map: Dict[str, List[float]],
    audio_paths: List[str],
    architecture: str,
    framework: str,
    total_inference_time: float,
    valid_duration_processed: float,
    rtf: float,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compiles final correlation metrics via Ridge CV and persists lightweight evaluation data to disk.

    Args:
        df (pd.DataFrame): The primary dataset DataFrame.
        checkpoint_map (Dict[str, List[float]]): The mapped latent features by audio path.
        audio_paths (List[str]): Ordered list of audio paths corresponding to the DataFrame.
        architecture (str): The string identifier of the acoustic model.
        framework (str): The underlying framework execution string.
        total_inference_time (float): Sum of inference durations in seconds.
        valid_duration_processed (float): Sum of valid processed audio lengths in seconds.
        rtf (float): The calculated Real Time Factor.
        config (Dict[str, Any]): The loaded configuration dictionary.

    Returns:
        Dict[str, Any]: The fully constructed evaluation metrics payload.

    Raises:
        RuntimeError: If data compilation, grouping, or file output fails.
    """
    try:
        final_latents: List[List[float]] = [checkpoint_map.get(p, []) for p in audio_paths]
        df["extracted_features"] = final_latents
        
        chalearn_traits: List[str] = [
            "extraversion", "neuroticism", "agreeableness", 
            "conscientiousness", "openness", "interview_score"
        ]
        
        global_correlations: Dict[str, Dict[str, float]] = compute_ridge_cv_correlations(
            df=df, feature_col="extracted_features", traits=chalearn_traits
        )
        
        stratified_correlations: Dict[str, Any] = {}
        if "applied_snr_db" in df.columns:
            for snr_val, group_df in df.groupby("applied_snr_db"):
                snr_key: str = f"snr_{int(snr_val)}db" if pd.notna(snr_val) else "snr_unknown"
                stratified_correlations[snr_key] = compute_ridge_cv_correlations(
                    df=group_df, feature_col="extracted_features", traits=chalearn_traits
                )
                
        df_out: pd.DataFrame = df.drop(columns=["extracted_features"])
        output_csv: Path = Path(config.get("output_csv", f"{architecture}_evaluation_results.csv"))
        
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_csv, index=False)
        
        metrics: Dict[str, Any] = {
            "architecture": architecture,
            "framework": framework,
            "computational_latency": {
                "total_new_inference_sec": total_inference_time,
                "total_new_audio_sec": valid_duration_processed,
                "real_time_factor_rtf": rtf
            },
            "paralinguistic_quality_correlation_global": global_correlations,
            "paralinguistic_quality_correlation_stratified": stratified_correlations
        }
        
        metrics_json: Path = output_csv.with_suffix('.metrics.json')
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
            
        return metrics

    except Exception as e:
        logger.error("Failed to compile final evaluation metrics: %s", e)
        raise RuntimeError(f"Metrics compilation failed: {e}") from e


@register_domain("acoustic_edge_evaluation")
def evaluate_acoustic_edge(config: Dict[str, Any], global_device: torch.device) -> None:
    """
    Executes the Step-by-Step Acoustic Evaluation pipeline targeting latency, cross-validated correlation, and SPDD.
    Registers the execution domain within the central architecture registry and implements atomic resuming.

    Args:
        config (Dict[str, Any]): Dictionary containing architecture parameters and file paths.
        global_device (torch.device): Primary hardware device for inference execution.

    Raises:
        FileNotFoundError: If the benchmark dataset cannot be located.
    """
    device: torch.device = resolve_device(config.get("device", "cpu"), global_device)
    
    dataset_path: Path = Path(config.get("dataset_path", "chalearn_multimodal_benchmark_noisy.csv"))
    if not dataset_path.exists():
        logger.error("Dataset missing at path: %s", dataset_path)
        raise FileNotFoundError(f"Prepared ChaLearn dataset missing at: {dataset_path}")
        
    try:
        df: pd.DataFrame = pd.read_csv(dataset_path)
        logger.info("Loaded %d records from the benchmark", len(df))
    except Exception as e:
        logger.error("Failed to read dataset CSV: %s", e)
        raise

    architecture: str = config.get("architecture", "wavlm")
    model_id: str = config.get("model_id", "microsoft/wavlm-base-plus")
    target_sr: int = config.get("target_sample_rate", 16000)
    batch_size: int = config.get("batch_size", 4)
    
    checkpoint_path: Path = Path(config.get("checkpoint_path", f"src/results/acoustic/checkpoints/{architecture}_checkpoint.json"))
    checkpoint_map: Dict[str, List[float]] = load_checkpoint_map(checkpoint_path)
    
    model: Any
    extractor: Any
    framework: str
    model, extractor, framework = initialize_acoustic_model(architecture, model_id, device)
    
    audio_paths: List[str] = df["audio_path"].astype(str).tolist()
    durations: List[float] = pd.to_numeric(df["duration_sec"], errors='coerce').fillna(0.0).tolist()
    
    pending_tasks: List[Tuple[str, float]] = [
        (p, d) for p, d in zip(audio_paths, durations) if p not in checkpoint_map
    ]
    
    logger.info("Total samples: %d | Cached: %d | Pending: %d", len(audio_paths), len(checkpoint_map), len(pending_tasks))
    
    total_inference_time: float = 0.0
    valid_duration_processed: float = 0.0
    
    if pending_tasks:
        total_inference_time, valid_duration_processed = process_acoustic_batches(
            pending_tasks=pending_tasks,
            checkpoint_map=checkpoint_map,
            checkpoint_path=checkpoint_path,
            model=model,
            extractor=extractor,
            framework=framework,
            target_sr=target_sr,
            device=device,
            batch_size=batch_size
        )

    rtf: float = compute_real_time_factor(total_inference_time, valid_duration_processed) if valid_duration_processed > 0 else 0.0
    
    finalize_evaluation_metrics(
        df=df,
        checkpoint_map=checkpoint_map,
        audio_paths=audio_paths,
        architecture=architecture,
        framework=framework,
        total_inference_time=total_inference_time,
        valid_duration_processed=valid_duration_processed,
        rtf=rtf,
        config=config
    )
    
    logger.info("Pipeline Execution Complete. RTF for recent run: %.4f", rtf)