import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger: logging.Logger = logging.getLogger(__name__)

CHALEARN_TRAITS: List[str] = [
    "extraversion", 
    "neuroticism", 
    "agreeableness", 
    "conscientiousness", 
    "openness", 
    "interview"
]


def load_embeddings_matrix(csv_path: Path) -> np.ndarray:
    """
    Extracts and stacks the 1024-dimensional latent embeddings from the model output.

    Args:
        csv_path (Path): Filesystem path to the adapter results CSV.

    Returns:
        np.ndarray: A dense matrix of valid embeddings shaped (samples, dimensions).

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV is malformed, missing columns, or contains zero valid embeddings.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found at {csv_path}")

    try:
        df: pd.DataFrame = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV at {csv_path}: {exc}") from exc

    if "latent_embeddings" not in df.columns or "status" not in df.columns:
        raise ValueError("CSV is missing required 'latent_embeddings' or 'status' columns.")

    valid_embeddings: List[List[float]] = []
    expected_dim: int = 0

    for _, row in df.iterrows():
        if row.get("status") != "SUCCESS":
            continue

        try:
            embedding: List[float] = ast.literal_eval(row["latent_embeddings"])
            
            if expected_dim == 0:
                expected_dim = len(embedding)
                
            if len(embedding) == expected_dim:
                valid_embeddings.append(embedding)
        except Exception as exc:
            logger.debug("Skipping malformed embedding array: %s", exc)

    if not valid_embeddings:
        raise ValueError("Zero valid embeddings extracted from the CSV.")

    return np.array(valid_embeddings, dtype=np.float32)


def generate_calibration_thresholds(
    predictions: np.ndarray, 
    target_traits: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculates statistical quintiles across the predicted continuous score distributions.

    Args:
        predictions (np.ndarray): The regressed multi-target matrix shaped (samples, traits).
        target_traits (List[str]): The ordered psychological trait labels.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary mapping each trait to its four threshold boundaries.
    """
    calibration_matrix: Dict[str, Dict[str, float]] = {}
    
    for idx, trait in enumerate(target_traits):
        trait_scores: np.ndarray = predictions[:, idx]
        
        p20: float = float(np.percentile(trait_scores, 20))
        p40: float = float(np.percentile(trait_scores, 40))
        p60: float = float(np.percentile(trait_scores, 60))
        p80: float = float(np.percentile(trait_scores, 80))
        
        calibration_matrix[trait] = {
            "very_low_threshold": p20,
            "low_threshold": p40,
            "high_threshold": p60,
            "very_high_threshold": p80
        }
        
    return calibration_matrix


def main() -> None:
    """
    Coordinates the embedding ingestion, pipeline inference, and artifact serialization workflow.
    """
    parser = argparse.ArgumentParser(description="Generates statistical BeMERC thresholds.")
    parser.add_argument("--csv", type=Path, default=Path("results/vision/facial_poster_results.csv"), help="POSTER++ results CSV")
    parser.add_argument("--model", type=Path, default=Path("app/models/visual/facial_poster_regressor.joblib"), help="Regression model path")
    parser.add_argument("--output", type=Path, default=Path("app/models/visual/facial_calibration.json"), help="Output JSON artifact path")
    args: argparse.Namespace = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        if not args.model.exists():
            raise FileNotFoundError(f"Regression model not found at {args.model}. Run the training script first.")
            
        logger.info("Loading regression pipeline from %s", args.model)
        pipeline: Pipeline = joblib.load(args.model)
        
        logger.info("Extracting raw latent embeddings from %s", args.csv)
        X: np.ndarray = load_embeddings_matrix(args.csv)
        
        logger.info("Predicting continuous scores for %d samples.", len(X))
        Y_pred: np.ndarray = pipeline.predict(X)
        
        if Y_pred.shape[1] != len(CHALEARN_TRAITS):
            logger.warning("Pipeline output dimensions (%d) mismatch expected trait count (%d).", Y_pred.shape[1], len(CHALEARN_TRAITS))
            
        calibration_data: Dict[str, Dict[str, float]] = generate_calibration_thresholds(
            predictions=Y_pred,
            target_traits=CHALEARN_TRAITS[:Y_pred.shape[1]]
        )
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as json_file:
            json.dump(calibration_data, json_file, indent=4)
            
        logger.info("Successfully serialized BeMERC calibration boundaries to %s", args.output)
        
    except Exception as exc:
        logger.error("Calibration generation aborted: %s", exc)


if __name__ == "__main__":
    main()