from typing import Dict


def _evaluate_psychometric_tier(score: float, thresholds: Dict[str, float]) -> int:
    """
    Evaluates a continuous regression score against the statistical BeMERC quintiles.

    Args:
        score (float): The continuous psychometric score to evaluate.
        thresholds (Dict[str, float]): The validated threshold dictionary for the target trait.

    Returns:
        int: The resolved bin index (0 to 4) representing the categorical percentile rank.
    """
    if score < thresholds["very_low_threshold"]:
        return 0
    if score < thresholds["low_threshold"]:
        return 1
    if score < thresholds["high_threshold"]:
        return 2
    if score < thresholds["very_high_threshold"]:
        return 3
    
    return 4


def _get_trait_phrase(trait: str, tier: int) -> str:
    """
    Maps an evaluated psychological trait tier to its corresponding descriptive BeMERC phrase.

    Args:
        trait (str): The OCEAN trait identifier.
        tier (int): The calculated integer tier (0-4).

    Returns:
        str: The behavioral string describing the specific trait expression.

    Raises:
        ValueError: If an unmapped trait or tier integer is provided.
    """
    phrases: Dict[str, Dict[int, str]] = {
        "extraversion": {
            0: "exhibits a flat, unresponsive facial affect",
            1: "displays subdued and minimal facial expressivity",
            2: "shows appropriate, balanced facial expressivity",
            3: "demonstrates engaging and animated facial expressions",
            4: "exhibits highly animated, exuberant facial expressivity"
        },
        "conscientiousness": {
            0: "demonstrates highly erratic gaze patterns with minimal eye contact",
            1: "shows frequent gaze shifting and inconsistent eye contact",
            2: "maintains steady, natural eye contact",
            3: "exhibits highly focused, stable gaze patterns",
            4: "demonstrates intense, unbroken visual focus"
        },
        "agreeableness": {
            0: "projects a detached, critically guarded demeanor",
            1: "appears reserved and highly formal",
            2: "displays polite, approachable non-verbal cues",
            3: "shows warm, cooperative engagement",
            4: "displays exceptional warmth and deeply empathetic affect"
        },
        "neuroticism": {
            0: "maintains a calm, highly resilient composure",
            1: "shows steady emotional regulation",
            2: "displays standard baseline affective responses",
            3: "exhibits visible signs of psychological tension",
            4: "displays highly reactive and anxious affect"
        },
        "openness": {
            0: "exhibits rigid, highly conventional non-verbal patterns",
            1: "shows cautious, standard expressive range",
            2: "displays adaptable non-verbal communication",
            3: "demonstrates fluid, inventive expressive range",
            4: "exhibits highly dynamic and expansive non-verbal behaviors"
        }
    }
    
    if trait not in phrases:
        raise ValueError(f"Invalid trait requested: {trait}")
    if tier not in phrases[trait]:
        raise ValueError(f"Invalid tier {tier} mapped for trait {trait}")
        
    return phrases[trait][tier]


def generate_facial_descriptor(
    scores: Dict[str, float], 
    calibration_thresholds: Dict[str, Dict[str, float]]
) -> str:
    """
    Translates the continuous full-spectrum OCEAN matrix into a natural language behavioral profile.

    Args:
        scores (Dict[str, float]): The mapping of trait names to their regressed continuous scores.
        calibration_thresholds (Dict[str, Dict[str, float]]): The validated nested threshold matrix.

    Returns:
        str: A grammatically correct, immutable BeMERC translation paragraph.

    Raises:
        TypeError: If input types violate the structural schema.
        ValueError: If mandatory psychological traits are missing from the input or thresholds.
    """
    if not isinstance(scores, dict):
        raise TypeError("scores must be a dictionary mapping traits to numeric values.")
    if not isinstance(calibration_thresholds, dict):
        raise TypeError("calibration_thresholds must be a dictionary.")

    required_traits = ["extraversion", "conscientiousness", "agreeableness", "neuroticism", "openness"]
    required_keys = ["very_low_threshold", "low_threshold", "high_threshold", "very_high_threshold"]
    
    for trait in required_traits:
        if trait not in scores:
            raise ValueError(f"scores dictionary missing required trait: '{trait}'")
        if not isinstance(scores[trait], (int, float)):
            raise TypeError(f"Score for '{trait}' must be a numeric value.")
            
        if trait not in calibration_thresholds:
            raise ValueError(f"calibration_thresholds must contain '{trait}' definitions.")
            
        trait_data = calibration_thresholds[trait]
        if not isinstance(trait_data, dict):
            raise TypeError(f"Threshold data for {trait} must be a dictionary.")
            
        for key in required_keys:
            if key not in trait_data:
                raise ValueError(f"Missing required boundary key '{key}' in {trait} thresholds.")

    tiers: Dict[str, int] = {}
    for trait in required_traits:
        tiers[trait] = _evaluate_psychometric_tier(float(scores[trait]), calibration_thresholds[trait])

    c_phrase = _get_trait_phrase("conscientiousness", tiers["conscientiousness"])
    e_phrase = _get_trait_phrase("extraversion", tiers["extraversion"])
    a_phrase = _get_trait_phrase("agreeableness", tiers["agreeableness"])
    n_phrase = _get_trait_phrase("neuroticism", tiers["neuroticism"])
    o_phrase = _get_trait_phrase("openness", tiers["openness"])

    return f"The candidate {c_phrase}, {e_phrase}, and {a_phrase}. Furthermore, they {n_phrase}, and {o_phrase}."