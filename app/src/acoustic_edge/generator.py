from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge

from app.src.acoustic_edge.config import AcousticConfig


def load_ridge_regressor(config: AcousticConfig) -> Ridge:
    """
    Initializes the scikit-learn Ridge regression model from persistent storage.

    Args:
        config (AcousticConfig): The active configuration matrix containing the model path.

    Returns:
        Ridge: The instantiated scikit-learn model ready for inference.

    Raises:
        FileNotFoundError: If the designated .joblib file does not exist.
        RuntimeError: If deserialization of the model file fails.
    """
    if not config.ridge_model_path.is_file():
        raise FileNotFoundError(f"Ridge model missing at {config.ridge_model_path}")

    try:
        model: Ridge = joblib.load(config.ridge_model_path)
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load Ridge components: {exc}") from exc


def _evaluate_trait(
    score: float, 
    thresholds: Tuple[float, float, float, float], 
    trait_name: str
) -> str:
    """
    Evaluates a continuous trait score against dynamic boundaries to return a quintile descriptor.

    Args:
        score (float): The predicted continuous numerical score.
        thresholds (Tuple[float, float, float, float]): Ascending percentile thresholds.
        trait_name (str): The specific personality trait being evaluated.

    Returns:
        str: A formatted natural language label representing the specific quintile bucket.
    """
    very_low, low, high, very_high = thresholds

    if score > very_high:
        return f"very high {trait_name}"
    if score > high:
        return f"high {trait_name}"
    if score < very_low:
        return f"very low {trait_name}"
    if score < low:
        return f"low {trait_name}"

    return f"medium {trait_name}"


def _format_cues_to_string(cues: List[str]) -> str:
    """
    Formats a list of discrete natural language descriptors into a grammatically correct string.

    Args:
        cues (List[str]): A list of string descriptors representing vocal behaviors.

    Returns:
        str: A concatenated natural language string suitable for LLM injection.
    """
    if not cues:
        return "moderate levels across all OCEAN traits"

    if len(cues) == 1:
        return cues[0]

    return f"{', '.join(cues[:-1])}, and {cues[-1]}"


def generate_competency_descriptors(
    embedding: torch.Tensor,
    ridge_model: Ridge,
    config: AcousticConfig
) -> Tuple[Tuple[str, ...], Dict[str, float]]:
    """
    Maps continuous structural features to quintile-based traits and returns raw scores.

    Args:
        embedding (torch.Tensor): A 1D tensor containing the pooled WavLM latent features.
        ridge_model (Ridge): The trained regression model for latent trait prediction.
        config (AcousticConfig): The active configuration matrix holding statistical thresholds.

    Returns:
        Tuple[Tuple[str, ...], Dict[str, float]]: A tuple containing the formatted string tuple and raw score dictionary.

    Raises:
        ValueError: If the embeddings tensor is empty or invalid.
        RuntimeError: If the prediction or text generation process encounters an error.
    """
    if embedding.numel() == 0:
        raise ValueError("Cannot generate descriptors from empty embeddings.")

    try:
        emb_numpy: np.ndarray = embedding.cpu().numpy().reshape(1, -1)
        trait_scores: np.ndarray = ridge_model.predict(emb_numpy)[0]

        thresholds: Tuple[float, float, float, float] = (
            config.very_low_threshold,
            config.low_threshold,
            config.high_threshold,
            config.very_high_threshold
        )

        trait_names: List[str] = [
            "extraversion", 
            "neuroticism", 
            "agreeableness", 
            "conscientiousness", 
            "openness"
        ]

        cues: List[str] = [
            _evaluate_trait(float(trait_scores[i]), thresholds, trait_names[i])
            for i in range(5)
        ]

        cues_str: str = _format_cues_to_string(cues)
        descriptor: str = f"The candidate's profile indicates {cues_str}."

        scores_dict: Dict[str, float] = {
            "extraversion_score": float(trait_scores[0]),
            "neuroticism_score": float(trait_scores[1]),
            "agreeableness_score": float(trait_scores[2]),
            "conscientiousness_score": float(trait_scores[3]),
            "openness_score": float(trait_scores[4])
        }

        return (descriptor,), scores_dict

    except Exception as exc:
        raise RuntimeError(f"Descriptor generation failed: {exc}") from exc