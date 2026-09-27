from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch
from sklearn.linear_model import Ridge

from app.src.acoustic_edge.config import AcousticConfig
from app.src.acoustic_edge.generator import (
    _evaluate_trait,
    _format_cues_to_string,
    generate_competency_descriptors,
    load_ridge_regressor,
)


@pytest.fixture(scope="module")
def production_config() -> AcousticConfig:
    """Provides a structural configuration targeting actual serialised artifacts."""
    model_path: Path = Path("app/models/acoustic/ridge_ocean_regressor.joblib")
    calib_path: Path = Path("app/models/acoustic/calibration.json")
    
    if not model_path.exists() or not calib_path.exists():
        pytest.skip("Production artifacts missing. Please run export and calibration scripts first.")
        
    return AcousticConfig(
        wavlm_repo_id="microsoft/wavlm-large",
        wavlm_revision="refs/pr/7",
        ridge_model_path=model_path,
        calibration_path=calib_path,
        device="cpu",
        use_float16=False
    )


def test_load_ridge_regressor_expected_behaviour_for_valid_model(production_config: AcousticConfig) -> None:
    """Tests if the loader successfully deserializes the actual .joblib file from disk."""
    model: Ridge = load_ridge_regressor(production_config)
    assert isinstance(model, Ridge)
    assert hasattr(model, "coef_")


def test_load_ridge_regressor_raises_for_missing_file() -> None:
    """Tests if the loader safely catches missing model targets."""
    missing_path: Path = Path("app/models/acoustic/does_not_exist.joblib")
    config = AcousticConfig.__new__(AcousticConfig)
    object.__setattr__(config, "ridge_model_path", missing_path)
    
    with pytest.raises(FileNotFoundError, match="Ridge model missing"):
        load_ridge_regressor(config)


def test_load_ridge_regressor_raises_for_corrupted_file() -> None:
    """Tests if the loader wraps deserialization failures into a RuntimeError."""
    corrupt_path: Path = Path("app/models/acoustic/test_corrupt_model.joblib")
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("corrupted binary object")
    
    config = AcousticConfig.__new__(AcousticConfig)
    object.__setattr__(config, "ridge_model_path", corrupt_path)
    
    try:
        with pytest.raises(RuntimeError, match="Failed to load Ridge components"):
            load_ridge_regressor(config)
    finally:
        if corrupt_path.exists():
            corrupt_path.unlink()


def test_evaluate_trait_expected_behaviour_for_quintile_boundaries() -> None:
    """Tests the five-tier mapping logic against specific numerical thresholds."""
    thresholds: Tuple[float, float, float, float] = (0.2, 0.4, 0.6, 0.8)
    
    assert _evaluate_trait(0.9, thresholds, "trait") == "very high trait"
    assert _evaluate_trait(0.7, thresholds, "trait") == "high trait"
    assert _evaluate_trait(0.5, thresholds, "trait") == "medium trait"
    assert _evaluate_trait(0.3, thresholds, "trait") == "low trait"
    assert _evaluate_trait(0.1, thresholds, "trait") == "very low trait"


def test_format_cues_to_string_expected_behaviour_for_various_lengths() -> None:
    """Tests grammatical formatting rules for descriptor sequences."""
    assert _format_cues_to_string([]) == "moderate levels across all OCEAN traits"
    assert _format_cues_to_string(["high extraversion"]) == "high extraversion"
    assert _format_cues_to_string(["high extraversion", "low openness"]) == "high extraversion, and low openness"
    assert _format_cues_to_string(["A", "B", "C"]) == "A, B, and C"


def test_generate_competency_descriptors_expected_behaviour_for_valid_embedding(production_config: AcousticConfig) -> None:
    """Tests the end-to-end translation mapping to text and extracting raw float dictionaries."""
    ridge_model: Ridge = load_ridge_regressor(production_config)
    expected_features: int = ridge_model.n_features_in_
    valid_embedding: torch.Tensor = torch.randn(expected_features)
    
    result: Tuple[Tuple[str, ...], Dict[str, float]] = generate_competency_descriptors(
        embedding=valid_embedding, 
        ridge_model=ridge_model,
        config=production_config
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    
    cues, scores = result
    assert isinstance(cues, tuple)
    assert isinstance(cues[0], str)
    assert cues[0].startswith("The candidate's profile indicates")
    
    assert isinstance(scores, dict)
    assert "extraversion_score" in scores
    assert isinstance(scores["openness_score"], float)


def test_generate_competency_descriptors_raises_for_empty_embedding(production_config: AcousticConfig) -> None:
    """Tests structural rejection of unpopulated tensors."""
    ridge_model: Ridge = load_ridge_regressor(production_config)
    empty_embedding: torch.Tensor = torch.tensor([])
    
    with pytest.raises(ValueError, match="Cannot generate descriptors from empty embeddings"):
        generate_competency_descriptors(
            embedding=empty_embedding, 
            ridge_model=ridge_model,
            config=production_config
        )


def test_generate_competency_descriptors_raises_for_incompatible_tensor_shape(production_config: AcousticConfig) -> None:
    """Tests encapsulation of scikit-learn dimension errors."""
    ridge_model: Ridge = load_ridge_regressor(production_config)
    invalid_embedding: torch.Tensor = torch.randn(3)
    
    with pytest.raises(RuntimeError, match="Descriptor generation failed"):
        generate_competency_descriptors(
            embedding=invalid_embedding, 
            ridge_model=ridge_model,
            config=production_config
        )