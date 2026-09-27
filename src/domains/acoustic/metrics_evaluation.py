import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

logger = logging.getLogger(__name__)


def compute_real_time_factor(inference_time_sec: float, total_audio_duration_sec: float) -> float:
    """
    Computes the Real-Time Factor (RTF) for edge latency profiling.

    Args:
        inference_time_sec (float): Cumulative time spent executing the forward passes.
        total_audio_duration_sec (float): Cumulative duration of the processed audio.

    Returns:
        float: The RTF scalar (values < 1.0 indicate faster-than-real-time performance).
    """
    try:
        if total_audio_duration_sec <= 0:
            return float('inf')
        return float(inference_time_sec / total_audio_duration_sec)
    except Exception as e:
        logger.error("Failed to compute RTF: %s", e)
        return float('nan')


def compute_ridge_cv_correlations(
    df: pd.DataFrame, 
    feature_col: str, 
    traits: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates high-dimensional embeddings using dynamically scaled CV Ridge Regression.

    Args:
        df (pd.DataFrame): The dataset containing features and ground truth traits.
        feature_col (str): The column name containing the latent feature lists.
        traits (List[str]): A list of ground truth column names to predict.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary mapping each trait to its
        Pearson and Spearman correlation coefficients.

    Raises:
        KeyError: If the specified feature_col does not exist in the DataFrame.
        RuntimeError: If matrix stacking or Ridge Regression execution fails.
    """
    try:
        if feature_col not in df.columns:
            raise KeyError(f"Feature column '{feature_col}' missing from DataFrame.")

        valid_mask: pd.Series = df[feature_col].apply(lambda x: isinstance(x, list) and len(x) > 0)
        df_valid: pd.DataFrame = df[valid_mask].copy()

        n_samples: int = len(df_valid)
        
        if n_samples < 2:
            logger.warning("Insufficient samples (%d) for cross-validation.", n_samples)
            return {trait: {"pearson": float('nan'), "spearman": float('nan')} for trait in traits}

        x_matrix: np.ndarray = np.vstack(df_valid[feature_col].values)
        
        # Dynamically scale splits for tiny test batches to prevent ValueError
        actual_splits: int = min(5, n_samples)
        cv: KFold = KFold(n_splits=actual_splits, shuffle=True, random_state=42)
        model: Ridge = Ridge(alpha=1.0)
        
        results: Dict[str, Dict[str, float]] = {}

        for trait in traits:
            if trait not in df_valid.columns:
                continue
                
            y_vector: np.ndarray = df_valid[trait].values
            y_pred: np.ndarray = cross_val_predict(model, x_matrix, y_vector, cv=cv, n_jobs=-1)
            
            p_corr, _ = pearsonr(y_vector, y_pred)
            s_corr, _ = spearmanr(y_vector, y_pred)
            
            results[trait] = {
                "pearson": float(p_corr),
                "spearman": float(s_corr)
            }
            
        return results

    except Exception as e:
        logger.error("Ridge CV correlation computation failed: %s", e)
        raise RuntimeError(f"Correlation computation failed: {e}") from e