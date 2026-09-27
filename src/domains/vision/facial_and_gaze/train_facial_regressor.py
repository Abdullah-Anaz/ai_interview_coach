import argparse
import ast
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger: logging.Logger = logging.getLogger(__name__)

CHALEARN_TRAITS: List[str] = [
    "extraversion", 
    "neuroticism", 
    "agreeableness", 
    "conscientiousness", 
    "openness", 
    "interview"
]


def _parse_dictionary_payload(data: Dict[Any, Any]) -> Dict[str, Dict[str, float]]:
    """
    Extracts psychometric trait scores from a deserialized nested dictionary payload.

    Args:
        data (Dict[Any, Any]): The raw dictionary loaded from the ChaLearn pickle file.

    Returns:
        Dict[str, Dict[str, float]]: A mapping of video stems to their respective trait scores.
    """
    parsed_annotations: Dict[str, Dict[str, float]] = {}
    keys: List[Any] = list(data.keys())
    
    if any(str(k).lower() in CHALEARN_TRAITS for k in keys):
        for trait, videos in data.items():
            trait_clean: str = str(trait).lower()
            if isinstance(videos, dict):
                for vid_name, score in videos.items():
                    vid_clean: str = Path(str(vid_name)).stem
                    if vid_clean not in parsed_annotations:
                        parsed_annotations[vid_clean] = {}
                    parsed_annotations[vid_clean][trait_clean] = float(score)
        return parsed_annotations

    for vid_name, traits in data.items():
        vid_clean: str = Path(str(vid_name)).stem
        if isinstance(traits, dict):
            parsed_annotations[vid_clean] = {str(k).lower(): float(v) for k, v in traits.items()}
            
    return parsed_annotations


def _parse_dataframe_payload(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Extracts psychometric trait scores from a deserialized DataFrame payload.

    Args:
        data (pd.DataFrame): The raw DataFrame loaded from the ChaLearn pickle file.

    Returns:
        Dict[str, Dict[str, float]]: A mapping of video stems to their respective trait scores.
    """
    parsed_annotations: Dict[str, Dict[str, float]] = {}
    
    for _, row in data.iterrows():
        vid_clean: str = Path(str(row.get("video", row.name))).stem
        parsed_annotations[vid_clean] = {
            str(col).lower(): float(row[col]) for col in data.columns if str(col).lower() != "video"
        }
        
    return parsed_annotations


def load_chalearn_annotations(annotation_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Recursively discovers and parses ground-truth ChaLearn annotations from the filesystem.

    Args:
        annotation_dir (Path): The root directory containing the .pkl files.

    Returns:
        Dict[str, Dict[str, float]]: A consolidated mapping of all video stems to their trait scores.

    Raises:
        FileNotFoundError: If no .pkl annotation files exist within the target directory.
        RuntimeError: If binary deserialization fails on any target file.
    """
    merged_annotations: Dict[str, Dict[str, float]] = {}
    pkl_files: List[Path] = list(annotation_dir.rglob("*.pkl"))
    
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl annotation files found in {annotation_dir}")

    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as file:
                data: Any = pickle.load(file, encoding="latin1")
            
            parsed_data: Dict[str, Dict[str, float]] = {}
            if isinstance(data, dict):
                parsed_data = _parse_dictionary_payload(data)
            elif isinstance(data, pd.DataFrame):
                parsed_data = _parse_dataframe_payload(data)
                
            merged_annotations.update(parsed_data)
            logger.info("Parsed %d unique videos from %s", len(parsed_data), pkl_path.name)
            
        except Exception as exc:
            logger.error("Failed to parse annotation file %s: %s", pkl_path, exc)
            raise RuntimeError(f"Annotation deserialization failed: {exc}") from exc

    return merged_annotations


def extract_target_keys(annotations: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Determines the applicable psychological target vectors present in the dataset.

    Args:
        annotations (Dict[str, Dict[str, float]]): The consolidated ground-truth annotations.

    Returns:
        List[str]: The strictly ordered target trait names for mathematical regression.

    Raises:
        ValueError: If the annotation dictionary is completely empty.
    """
    if not annotations:
        raise ValueError("Cannot extract target keys from an empty annotation mapping.")
        
    sample_traits: Dict[str, float] = next(iter(annotations.values()))
    target_keys: List[str] = [t for t in CHALEARN_TRAITS if t in sample_traits or f"{t}_score" in sample_traits]
    
    return target_keys if target_keys else list(sample_traits.keys())


def prepare_dataset(
    adapter_csv_path: Path, 
    annotations: Dict[str, Dict[str, float]]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Cross-references latent embeddings with ground-truth labels to build mathematical matrices.
    Dynamically infers the embedding dimensionality to support varied model architectures.

    Args:
        adapter_csv_path (Path): Filesystem path to the adapter output CSV.
        annotations (Dict[str, Dict[str, float]]): The parsed ChaLearn ground truth.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: The independent variables (X), dependent variables (Y), and trait order.

    Raises:
        ValueError: If the input CSV is malformed or if zero intersecting records exist.
    """
    try:
        df: pd.DataFrame = pd.read_csv(adapter_csv_path)
    except Exception as exc:
        raise ValueError(f"Failed to read results from {adapter_csv_path}: {exc}") from exc
    
    if "video_path" not in df.columns or "latent_embeddings" not in df.columns:
        raise ValueError("CSV is missing required 'video_path' or 'latent_embeddings' columns.")

    X_list: List[List[float]] = []
    Y_list: List[List[float]] = []
    target_keys: List[str] = extract_target_keys(annotations)
    expected_dim: int = 0

    for _, row in df.iterrows():
        if row.get("status") != "SUCCESS":
            continue
            
        video_filename: str = Path(str(row["video_path"])).stem
        
        if video_filename in annotations:
            try:
                embedding: List[float] = ast.literal_eval(row["latent_embeddings"])
                
                if expected_dim == 0:
                    expected_dim = len(embedding)
                    logger.info("Inferred latent embedding dimensionality: %d", expected_dim)
                    
                if len(embedding) != expected_dim:
                    continue
                    
                trait_scores: List[float] = [annotations[video_filename].get(tk, 0.0) for tk in target_keys]
                
                X_list.append(embedding)
                Y_list.append(trait_scores)
            except Exception as exc:
                logger.debug("Failed to evaluate structural embedding for %s: %s", video_filename, exc)

    X: np.ndarray = np.array(X_list, dtype=np.float32)
    Y: np.ndarray = np.array(Y_list, dtype=np.float32)

    if len(X) == 0:
        raise ValueError("Zero intersecting samples found between adapter results and annotations.")

    return X, Y, target_keys


def train_and_evaluate(
    X: np.ndarray, 
    Y: np.ndarray, 
    target_keys: List[str]
) -> Tuple[Pipeline, Dict[str, Dict[str, float]]]:
    """
    Fits a mathematical regression pipeline and computes linear and monotonic validity coefficients.

    Args:
        X (np.ndarray): The latent embedding independent matrix.
        Y (np.ndarray): The continuous multi-target dependent matrix.
        target_keys (List[str]): The names of the psychological traits being evaluated.

    Returns:
        Tuple[Pipeline, Dict[str, Dict[str, float]]]: The converged regression model and its validation metrics.
    """
    pipeline: Pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MultiOutputRegressor(Ridge(alpha=1.0, random_state=42)))
    ])
    
    logger.info("Training Multi-Output Ridge Regressor on %d samples.", len(X))
    pipeline.fit(X, Y)
    
    Y_pred: np.ndarray = pipeline.predict(X)
    metrics: Dict[str, Dict[str, float]] = {}
    
    logger.info("Psychometric Validation:")
    for i, trait_name in enumerate(target_keys):
        truth_col: np.ndarray = Y[:, i]
        pred_col: np.ndarray = Y_pred[:, i]
        
        if np.std(truth_col) == 0 or np.std(pred_col) == 0:
            logger.info("%-18s : Undefined (Zero Variance)", trait_name.capitalize())
            metrics[trait_name] = {
                "pearson_r": 0.0, "pearson_p": 1.0,
                "spearman_rho": 0.0, "spearman_p": 1.0
            }
            continue
            
        r_val: float
        p_val_r: float
        rho_val: float
        p_val_rho: float
        
        r_val, p_val_r = pearsonr(truth_col, pred_col)
        rho_val, p_val_rho = spearmanr(truth_col, pred_col)
        
        metrics[trait_name] = {
            "pearson_r": float(r_val), 
            "pearson_p": float(p_val_r),
            "spearman_rho": float(rho_val),
            "spearman_p": float(p_val_rho)
        }
        
        logger.info(
            "%-18s : Pearson r = %+.4f (p=%.4f) | Spearman rho = %+.4f (p=%.4f)", 
            trait_name.capitalize(), r_val, p_val_r, rho_val, p_val_rho
        )
        
    return pipeline, metrics


def main() -> None:
    """
    Coordinates the holistic data ingestion, mathematical fitting, and pipeline export lifecycle.
    """
    parser = argparse.ArgumentParser(description="Trains the Latent Regressor for Visual Encoders.")
    parser.add_argument("--csv", type=Path, default=Path("results/vision/facial_poster_results.csv"), help="Adapter results CSV")
    parser.add_argument("--annotations", type=Path, default=Path("src/data/chalearn/annotations"), help="ChaLearn PKL dir")
    parser.add_argument("--output", type=Path, default=Path("app/models/visual/facial_poster_regressor.joblib"), help="Output model path")
    parser.add_argument("--metrics", type=Path, default=Path("results/vision/poster_plus_correlation_metrics.json"), help="Output JSON metrics path")
    args: argparse.Namespace = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        annotations: Dict[str, Dict[str, float]] = load_chalearn_annotations(args.annotations)
        
        X: np.ndarray
        Y: np.ndarray
        target_keys: List[str]
        X, Y, target_keys = prepare_dataset(args.csv, annotations)
        
        trained_pipeline: Pipeline
        validation_metrics: Dict[str, Dict[str, float]]
        trained_pipeline, validation_metrics = train_and_evaluate(X, Y, target_keys)
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained_pipeline, args.output)
        logger.info("Successfully serialized regression artifact to %s", args.output)
        
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics, "w", encoding="utf-8") as f:
            json.dump(validation_metrics, f, indent=4)
        logger.info("Successfully serialized validation metrics to %s", args.metrics)
        
    except Exception as exc:
        logger.error("Regressor training aborted: %s", exc)


if __name__ == "__main__":
    main()