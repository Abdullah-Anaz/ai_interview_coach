import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

OCEAN_COLUMNS: List[str] = [
    "extraversion",
    "neuroticism",
    "agreeableness",
    "conscientiousness",
    "openness",
]


def parse_arguments(cli_args: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parses command-line arguments for statistical quintile calibration generation.

    Args:
        cli_args (Sequence[str] | None): Command-line argument strings to parse.

    Returns:
        argparse.Namespace: Parsed CLI argument container with validated parameters.
    """
    parser = argparse.ArgumentParser(
        description="Generates calibration.json quintile boundaries from evaluation scores."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to evaluation CSV file containing Big Five personality trait scores.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=False,
        default=Path("app/models/acoustic/calibration.json"),
        help="Target filesystem path where the calibration JSON file will be written.",
    )
    return parser.parse_args(cli_args)


def extract_ocean_scores(csv_path: Path) -> np.ndarray:
    """
    Loads and extracts normalized Big Five trait scores from an evaluation CSV file.

    Args:
        csv_path (Path): Path to the target CSV file containing OCEAN personality traits.

    Returns:
        np.ndarray: A flattened one-dimensional NumPy array of numeric trait scores.

    Raises:
        FileNotFoundError: If the specified CSV file path does not exist on disk.
        ValueError: If required OCEAN columns are missing or if no valid numeric scores exist.
        RuntimeError: If reading or parsing the CSV file encounters an unrecoverable failure.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Target evaluation CSV not found: {csv_path}")

    try:
        dataframe: pd.DataFrame = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV dataset from {csv_path}: {exc}") from exc

    missing_columns: List[str] = [
        col for col in OCEAN_COLUMNS if col not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"CSV file at {csv_path} is missing required trait columns: {missing_columns}"
        )

    try:
        numeric_subset: pd.DataFrame = dataframe[OCEAN_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to parse OCEAN columns into numeric values: {exc}"
        ) from exc

    raw_scores: np.ndarray = numeric_subset.to_numpy().flatten()
    clean_scores: np.ndarray = raw_scores[~np.isnan(raw_scores)].astype(np.float64)

    if clean_scores.size == 0:
        raise ValueError(
            f"No valid numeric trait scores could be retrieved from {csv_path}."
        )

    return clean_scores


def compute_quintile_thresholds(scores: np.ndarray) -> Dict[str, float]:
    """
    Computes statistical quintile boundaries across a continuous score distribution.

    Args:
        scores (np.ndarray): 1D array of continuous numeric scores.

    Returns:
        Dict[str, float]: Dictionary mapping quintile threshold identifiers to rounded values.

    Raises:
        ValueError: If scores array is empty or percentile values do not strictly ascend.
    """
    if scores.size == 0:
        raise ValueError("Cannot compute statistical thresholds from an empty array.")

    p20: float = float(np.percentile(scores, 20))
    p40: float = float(np.percentile(scores, 40))
    p60: float = float(np.percentile(scores, 60))
    p80: float = float(np.percentile(scores, 80))

    if not (p20 < p40 < p60 < p80):
        raise ValueError(
            f"Percentile collapse detected. Boundaries must be strictly ascending: "
            f"p20={p20}, p40={p40}, p60={p60}, p80={p80}"
        )

    return {
        "very_low_threshold": round(p20, 4),
        "low_threshold": round(p40, 4),
        "high_threshold": round(p60, 4),
        "very_high_threshold": round(p80, 4),
    }


def export_calibration_file(
    calibration_data: Dict[str, float], output_path: Path
) -> None:
    """
    Serializes trait calibration boundaries to a persistent JSON file.

    Args:
        calibration_data (Dict[str, float]): Dictionary containing calibrated boundary thresholds.
        output_path (Path): File path destination for the output JSON artifact.

    Raises:
        RuntimeError: If filesystem operations or JSON serialization fails.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(calibration_data, json_file, indent=2)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to serialize calibration data to {output_path}: {exc}"
        ) from exc


def main() -> None:
    """
    Coordinates extraction, threshold computation, and file export for trait calibration.

    Raises:
        FileNotFoundError: If the specified CSV dataset file is missing.
        ValueError: If columns or values in the dataset are invalid.
        RuntimeError: If disk read or write operations encounter unhandled errors.
    """
    args: argparse.Namespace = parse_arguments()
    scores: np.ndarray = extract_ocean_scores(csv_path=args.csv_path)
    calibration_dict: Dict[str, float] = compute_quintile_thresholds(scores=scores)
    export_calibration_file(
        calibration_data=calibration_dict, output_path=args.output_path
    )


if __name__ == "__main__":
    main()