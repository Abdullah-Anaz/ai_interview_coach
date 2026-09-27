import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_source_data(csv_path: Path, json_path: Path) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """
    Loads the benchmark dataset and the extracted latent embeddings from disk.

    Args:
        csv_path (Path): Path to the ChaLearn benchmark CSV file.
        json_path (Path): Path to the JSON checkpoint containing WavLM embeddings.

    Returns:
        Tuple[pd.DataFrame, Dict[str, List[float]]]: A tuple containing the dataset 
        DataFrame and a dictionary mapping audio paths to their latent vectors.

    Raises:
        RuntimeError: If either file cannot be located or parsed.
    """
    try:
        df: pd.DataFrame = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {csv_path}: {e}") from e

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            checkpoint_map: Dict[str, List[float]] = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings from {json_path}: {e}") from e

    return df, checkpoint_map


def prepare_training_matrices(
    df: pd.DataFrame, 
    checkpoint_map: Dict[str, List[float]], 
    traits: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns the tabular dataset with the extracted embeddings and constructs 
    the feature (X) and target (Y) matrices for regression training.

    Args:
        df (pd.DataFrame): The base dataset containing target labels and audio paths.
        checkpoint_map (Dict[str, List[float]]): The mapped latent features.
        traits (List[str]): The column names corresponding to the target regression variables.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The aligned feature matrix (X) and target matrix (Y).

    Raises:
        ValueError: If essential columns are missing or if the resulting valid dataset is empty.
    """
    if "audio_path" not in df.columns:
        raise ValueError("The required column 'audio_path' is missing from the dataset.")

    missing_traits: List[str] = [trait for trait in traits if trait not in df.columns]
    if missing_traits:
        raise ValueError(f"The following target traits are missing from the dataset: {missing_traits}")

    audio_paths: List[str] = df["audio_path"].astype(str).tolist()
    embeddings: List[List[float] | None] = [checkpoint_map.get(p) for p in audio_paths]
    
    df_copy: pd.DataFrame = df.copy()
    df_copy["extracted_features"] = embeddings

    valid_mask: pd.Series = df_copy["extracted_features"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    df_valid: pd.DataFrame = df_copy[valid_mask]

    if df_valid.empty:
        raise ValueError("No valid embeddings aligned with the dataset. Matrices cannot be constructed.")

    x_matrix: np.ndarray = np.vstack(df_valid["extracted_features"].values)
    y_matrix: np.ndarray = df_valid[traits].values

    return x_matrix, y_matrix


def train_ridge_regressor(x_matrix: np.ndarray, y_matrix: np.ndarray, alpha: float = 1.0) -> Ridge:
    """
    Instantiates and fits a Ridge regression model using the provided matrices.

    Args:
        x_matrix (np.ndarray): The input feature matrix of shape (n_samples, n_features).
        y_matrix (np.ndarray): The target label matrix of shape (n_samples, n_targets).
        alpha (float): Regularization strength.

    Returns:
        Ridge: The trained scikit-learn Ridge regression model.

    Raises:
        ValueError: If input matrices have mismatched sample counts.
    """
    if x_matrix.shape[0] != y_matrix.shape[0]:
        raise ValueError(
            f"Sample mismatch: X has {x_matrix.shape[0]} samples, Y has {y_matrix.shape[0]} samples."
        )

    model: Ridge = Ridge(alpha=alpha)
    model.fit(x_matrix, y_matrix)
    
    return model


def persist_model(model: Ridge, output_path: Path) -> None:
    """
    Serializes the trained regression model to the filesystem.

    Args:
        model (Ridge): The fitted scikit-learn model.
        output_path (Path): The designated file path for the exported joblib file.

    Raises:
        RuntimeError: If directory creation or file serialization fails.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to serialize and save the model to {output_path}: {e}") from e


def run_export_pipeline() -> None:
    """
    Orchestrates the data loading, matrix preparation, training, and exportation workflow.
    """
    csv_path: Path = Path("src/data/chalearn/chalearn_multimodal_benchmark_noisy.csv")
    json_path: Path = Path("src/results/acoustic/checkpoints/wavlm_checkpoint.json")
    output_path: Path = Path("src/models/acoustic/ridge_ocean_regressor.joblib")
    
    target_traits: List[str] = [
        "extraversion", 
        "neuroticism", 
        "agreeableness", 
        "conscientiousness", 
        "openness", 
        "interview_score"
    ]

    try:
        logger.info("Initializing data load...")
        df, checkpoint_map = load_source_data(csv_path, json_path)
        
        logger.info("Constructing feature matrices...")
        x_matrix, y_matrix = prepare_training_matrices(df, checkpoint_map, target_traits)
        
        logger.info("Fitting Ridge regressor on %d valid samples...", x_matrix.shape[0])
        trained_model: Ridge = train_ridge_regressor(x_matrix, y_matrix)
        
        logger.info("Persisting model to disk...")
        persist_model(trained_model, output_path)
        
        logger.info("Pipeline completed successfully. Model exported to %s", output_path)

    except (RuntimeError, ValueError) as e:
        logger.error("Pipeline aborted due to error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.critical("Pipeline aborted due to unexpected critical failure: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    run_export_pipeline()