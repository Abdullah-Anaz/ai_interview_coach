import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def compute_quintile_thresholds(scores: np.ndarray) -> Dict[str, float]:
    """
    Calculates the 20th, 40th, 60th, and 80th percentiles across the continuous distribution.

    Args:
        scores (np.ndarray): A 1D array of continuous numerical visual metrics.

    Returns:
        Dict[str, float]: A mapped dictionary containing the statistical boundaries.

    Raises:
        ValueError: If the array is empty or lacks sufficient variance.
    """
    if scores.size == 0:
        raise ValueError("Cannot compute thresholds from an empty array.")

    # Drop NaNs and enforce float64 for mathematical precision
    clean_scores: np.ndarray = scores[~np.isnan(scores)].astype(np.float64)

    if clean_scores.size < 5:
        raise ValueError("Insufficient data points to calculate valid quintiles.")

    p20: float = float(np.percentile(clean_scores, 20))
    p40: float = float(np.percentile(clean_scores, 40))
    p60: float = float(np.percentile(clean_scores, 60))
    p80: float = float(np.percentile(clean_scores, 80))

    if not (p20 <= p40 <= p60 <= p80):
        raise ValueError(f"Percentile collapse detected in distribution: {p20}, {p40}, {p60}, {p80}")

    return {
        "very_low_threshold": p20,
        "low_threshold": p40,
        "high_threshold": p60,
        "very_high_threshold": p80
    }


def main() -> None:
    """
    Coordinates the ingestion of Skeletal Kinematics data and artifact generation.
    """
    parser = argparse.ArgumentParser(description="Generates skeletal_calibration.json from Human-Omni matrices.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=False,
        default=Path("results/vision/human_omni.csv"),
        help="Path to the Human-Omni inference results CSV."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=False,
        default=Path("app/models/visual/skeletal_calibration.json"),
        help="Destination path for the generated BeMERC JSON file."
    )
    args: argparse.Namespace = parser.parse_args()

    if not args.csv_path.is_file():
        raise FileNotFoundError(f"Could not find Human-Omni dataset at {args.csv_path}")

    try:
        dataframe: pd.DataFrame = pd.read_csv(args.csv_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV dataset: {exc}") from exc

    required_columns = ["perceptual_jitter", "temporal_smoothness"]
    for col in required_columns:
        if col not in dataframe.columns:
            raise ValueError(f"Required metric column '{col}' missing from dataset.")

    jitter_scores: np.ndarray = dataframe["perceptual_jitter"].to_numpy()
    smoothness_scores: np.ndarray = dataframe["temporal_smoothness"].to_numpy()

    calibration_matrix: Dict[str, Dict[str, float]] = {
        "perceptual_jitter": compute_quintile_thresholds(jitter_scores),
        "temporal_smoothness": compute_quintile_thresholds(smoothness_scores)
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(calibration_matrix, file, indent=4)

    print(f"Skeletal calibration artifact successfully generated at: {args.output_path.resolve()}")
    print("\n--- Structural Calibration Matrix ---")
    print(json.dumps(calibration_matrix, indent=4))


if __name__ == "__main__":
    main()