import json
from pathlib import Path
from typing import Dict, Any

import pytest

from app.src.visual_edge.facial_gaze.generator import (
    _evaluate_psychometric_tier,
    generate_facial_descriptor
)

CALIBRATION_PATH: Path = Path("app/models/visual/facial_calibration.json")


def _get_actual_thresholds() -> Dict[str, Dict[str, float]]:
    """
    Fetches the actual production calibration artifact from the host disk.
    Ensures zero mocking is used during integration testing.
    """
    if not CALIBRATION_PATH.exists():
        pytest.fail(f"Actual threshold artifact missing at {CALIBRATION_PATH}. Cannot run integration tests.")
        
    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_evaluate_psychometric_tier_expected_behaviour_for_strict_boundaries() -> None:
    """Verifies that mathematical proxies correctly map to strict integer tiers using actual disk thresholds."""
    actual_data: Dict[str, Dict[str, float]] = _get_actual_thresholds()
    t: Dict[str, float] = actual_data["extraversion"]
    
    tier_0_val: float = t["very_low_threshold"] - 0.1
    tier_1_val: float = (t["very_low_threshold"] + t["low_threshold"]) / 2.0
    tier_2_val: float = (t["low_threshold"] + t["high_threshold"]) / 2.0
    tier_3_val: float = (t["high_threshold"] + t["very_high_threshold"]) / 2.0
    tier_4_val: float = t["very_high_threshold"] + 0.1
    
    assert _evaluate_psychometric_tier(tier_0_val, t) == 0
    assert _evaluate_psychometric_tier(tier_1_val, t) == 1
    assert _evaluate_psychometric_tier(tier_2_val, t) == 2
    assert _evaluate_psychometric_tier(tier_3_val, t) == 3
    assert _evaluate_psychometric_tier(tier_4_val, t) == 4


def test_generate_facial_descriptor_expected_behaviour_for_average_scores() -> None:
    """Verifies valid translation string generation for candidate behavior falling into median percentiles."""
    actual_data: Dict[str, Dict[str, float]] = _get_actual_thresholds()
    
    scores: Dict[str, float] = {}
    for trait in ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]:
        t: Dict[str, float] = actual_data[trait]
        scores[trait] = (t["low_threshold"] + t["high_threshold"]) / 2.0
    
    result: str = generate_facial_descriptor(scores, actual_data)
    
    expected: str = "The candidate maintains steady, natural eye contact, shows appropriate, balanced facial expressivity, and displays polite, approachable non-verbal cues. Furthermore, they displays standard baseline affective responses, and displays adaptable non-verbal communication."
    assert result == expected


def test_generate_facial_descriptor_expected_behaviour_for_extreme_low_scores() -> None:
    """Verifies valid translation string generation for severe negative behavioral deviations using actual thresholds."""
    actual_data: Dict[str, Dict[str, float]] = _get_actual_thresholds()
    
    scores: Dict[str, float] = {}
    for trait in ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]:
        scores[trait] = actual_data[trait]["very_low_threshold"] - 0.1
        
    result: str = generate_facial_descriptor(scores, actual_data)
    
    expected: str = "The candidate demonstrates highly erratic gaze patterns with minimal eye contact, exhibits a flat, unresponsive facial affect, and projects a detached, critically guarded demeanor. Furthermore, they maintains a calm, highly resilient composure, and exhibits rigid, highly conventional non-verbal patterns."
    assert result == expected


def test_generate_facial_descriptor_expected_behaviour_for_extreme_high_scores() -> None:
    """Verifies valid translation string generation for aggressive positive behavioral deviations."""
    actual_data: Dict[str, Dict[str, float]] = _get_actual_thresholds()
    
    scores: Dict[str, float] = {}
    for trait in ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]:
        scores[trait] = actual_data[trait]["very_high_threshold"] + 0.1
        
    result: str = generate_facial_descriptor(scores, actual_data)
    
    expected: str = "The candidate demonstrates intense, unbroken visual focus, exhibits highly animated, exuberant facial expressivity, and displays exceptional warmth and deeply empathetic affect. Furthermore, they displays highly reactive and anxious affect, and exhibits highly dynamic and expansive non-verbal behaviors."
    assert result == expected


def test_generate_facial_descriptor_raises_for_invalid_score_matrix_type() -> None:
    """Ensures translation aborts if the scores are not provided as a structured dictionary."""
    actual_data: Dict[str, Dict[str, float]] = _get_actual_thresholds()
    
    with pytest.raises(TypeError, match="scores must be a dictionary mapping traits to numeric values."):
        generate_facial_descriptor(["invalid", "list"], actual_data)  # type: ignore


def test_generate_facial_descriptor_raises_for_missing_trait_in_scores() -> None:
    """Ensures translation aborts if mandatory OCEAN traits are omitted from the regression output dictionary."""
    actual_data: Dict[str, Dict[str, float]] = _get_actual_thresholds()
    
    incomplete_scores: Dict[str, float] = {
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "conscientiousness": 0.5,
        "openness": 0.5
        # Missing neuroticism
    }
    
    with pytest.raises(ValueError, match="scores dictionary missing required trait: 'neuroticism'"):
        generate_facial_descriptor(incomplete_scores, actual_data)


def test_generate_facial_descriptor_raises_for_invalid_score_values() -> None:
    """Ensures translation aborts if non-numeric scores are passed in the regression dictionary."""
    actual_data: Dict[str, Dict[str, float]] = _get_actual_thresholds()
    
    corrupted_scores: Dict[str, Any] = {
        "extraversion": 0.5,
        "neuroticism": "High",  # type: ignore
        "agreeableness": 0.5,
        "conscientiousness": 0.5,
        "openness": 0.5
    }
    
    with pytest.raises(TypeError, match="Score for 'neuroticism' must be a numeric value."):
        generate_facial_descriptor(corrupted_scores, actual_data)  # type: ignore


def test_generate_facial_descriptor_raises_for_invalid_threshold_matrix_type() -> None:
    """Ensures translation aborts if the required configuration matrix is fundamentally malformed."""
    scores: Dict[str, float] = {t: 0.5 for t in ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]}
    
    with pytest.raises(TypeError, match="calibration_thresholds must be a dictionary."):
        generate_facial_descriptor(scores, ["not", "a", "dict"])  # type: ignore


def test_generate_facial_descriptor_raises_for_missing_trait_in_thresholds() -> None:
    """Ensures translation aborts if mandatory psychometric thresholds are missing from the configuration matrix."""
    actual_data: Dict[str, Dict[str, float]] = _get_actual_thresholds()
    scores: Dict[str, float] = {t: 0.5 for t in ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]}
    
    incomplete_thresholds: Dict[str, Dict[str, float]] = {
        "extraversion": actual_data["extraversion"],
        "neuroticism": actual_data["neuroticism"],
        "agreeableness": actual_data["agreeableness"],
        "conscientiousness": actual_data["conscientiousness"]
        # Missing openness definitions
    }
    
    with pytest.raises(ValueError, match="calibration_thresholds must contain 'openness' definitions."):
        generate_facial_descriptor(scores, incomplete_thresholds)


def test_generate_facial_descriptor_raises_for_malformed_threshold_keys() -> None:
    """Ensures translation aborts if the internal threshold boundaries deviate from strict schema definitions."""
    actual_data: Dict[str, Dict[str, float]] = _get_actual_thresholds()
    scores: Dict[str, float] = {t: 0.5 for t in ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]}
    
    corrupted_thresholds: Dict[str, Dict[str, float]] = dict(actual_data)
    corrupted_thresholds["agreeableness"] = {
        "very_low_threshold": 0.2,
        "low_threshold": 0.4
        # Missing high and very_high boundaries
    }
    
    with pytest.raises(ValueError, match="Missing required boundary key 'high_threshold' in agreeableness thresholds."):
        generate_facial_descriptor(scores, corrupted_thresholds)