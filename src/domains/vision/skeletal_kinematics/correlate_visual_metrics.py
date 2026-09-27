import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def parse_arguments(cli_args: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parses command-line arguments for statistical metric correlation.

    Args:
        cli_args (Sequence[str] | None): Command-line argument strings to parse.

    Returns:
        argparse.Namespace: Parsed CLI argument container with validated parameters.
    """
    parser = argparse.ArgumentParser(
        description="Dynamically maps video paths to ground-truth sets and calculates correlations."
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=False,
        default=Path("src/data/chalearn/annotations"),
        help="Base directory containing the nested ChaLearn annotation splits.",
    )
    parser.add_argument(
        "--visual-metrics-path",
        type=Path,
        required=False,
        default=Path("results/vision/mediapipe_gcn_results.csv"),
        help="Path to the visual subsystem inference results.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=False,
        default=Path("src/results/vision/correlation_results_mediapipe.csv"),
        help="Destination path for the exported correlation matrix.",
    )
    return parser.parse_args(cli_args)


def extract_split_and_filename(video_path: str) -> Tuple[str, str]:
    """
    Parses a raw video path to extract its dataset split and canonical filename.

    Args:
        video_path (str): The raw string path (e.g., 'src/data/chalearn/val/video.005.mp4').

    Returns:
        Tuple[str, str]: A tuple containing the split directory name and the file name.
    """
    path_obj: Path = Path(video_path)
    return path_obj.parent.name, path_obj.name


def resolve_annotation_file_path(split_name: str, annotations_dir: Path) -> Path:
    """
    Maps a dataset split identifier to its corresponding ChaLearn nested annotation path.

    Args:
        split_name (str): The dataset split extracted from the video path (e.g., 'val').
        annotations_dir (Path): The base annotations directory.

    Returns:
        Path: The absolute path to the target .pkl file.

    Raises:
        ValueError: If the extracted split name does not match known ChaLearn partitions.
    """
    mapping: Dict[str, str] = {
        "val": "val-annotation-e/annotation_validation.pkl",
        "train": "train-annotation/annotation_training.pkl",
        "test": "test-annotation-e/annotation_test.pkl",
    }
    
    if split_name not in mapping:
        raise ValueError(f"Unsupported dataset split extracted from path: '{split_name}'")
        
    return annotations_dir / mapping[split_name]


def load_chalearn_pickle(file_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Deserializes a ChaLearn annotation pickle file into a structured dictionary.

    Args:
        file_path (Path): Filesystem path to the .pkl annotation file.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary mapping video filenames to trait dictionaries.

    Raises:
        FileNotFoundError: If the target file does not exist.
        RuntimeError: If deserialization fails.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found at {file_path}")

    try:
        with open(file_path, "rb") as file:
            raw_data: Dict[str, Dict[str, float]] = pickle.load(file, encoding="latin1")

        parsed_data: Dict[str, Dict[str, float]] = {}
        traits: List[str] = list(raw_data.keys())
        
        if not traits:
            return parsed_data

        for vid_id in raw_data[traits[0]].keys():
            parsed_data[vid_id] = {trait: float(raw_data[trait][vid_id]) for trait in traits}
            
        return parsed_data
        
    except Exception as exc:
        raise RuntimeError(f"Failed to load ground-truth data from {file_path}: {exc}") from exc


def build_unified_dataset(metrics_df: pd.DataFrame, annotations_dir: Path) -> pd.DataFrame:
    """
    Iterates through inference metrics, dynamically loading and appending ground truth.

    Args:
        metrics_df (pd.DataFrame): The visual metrics DataFrame containing 'video_path'.
        annotations_dir (Path): The base annotations directory.

    Returns:
        pd.DataFrame: A unified DataFrame containing both visual metrics and ground truth.

    Raises:
        ValueError: If the metrics DataFrame lacks the required 'video_path' column.
    """
    if "video_path" not in metrics_df.columns:
        raise ValueError("Visual metrics file must contain a 'video_path' column.")

    loaded_annotations: Dict[str, Dict[str, Dict[str, float]]] = {}
    merged_records: List[Dict[str, Any]] = []

    for _, row in metrics_df.iterrows():
        video_path: str = str(row["video_path"])
        split, filename = extract_split_and_filename(video_path)

        if split not in loaded_annotations:
            pkl_path: Path = resolve_annotation_file_path(split, annotations_dir)
            loaded_annotations[split] = load_chalearn_pickle(pkl_path)

        ground_truth: Dict[str, float] | None = loaded_annotations[split].get(filename)
        
        if ground_truth:
            record: Dict[str, Any] = row.to_dict()
            record.update(ground_truth)
            merged_records.append(record)

    return pd.DataFrame(merged_records)


def compute_correlations(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistical alignment between empirical visual markers and psychological traits.

    Args:
        merged_data (pd.DataFrame): The unified dataset containing aligned metrics.

    Returns:
        pd.DataFrame: A matrix of Pearson and Spearman coefficients and corresponding p-values.
    """
    visual_metrics: List[str] = [
        col for col in merged_data.columns 
        if "jitter" in col.lower() or "smoothness" in col.lower()
    ]
    psychological_traits: List[str] = [
        "Extraversion", "Neuroticism", "Agreeableness", 
        "Conscientiousness", "Openness", "Interview"
    ]
    
    # ChaLearn pickle keys are frequently capitalized. This normalizes the lookup.
    available_traits: List[str] = [
        col for col in merged_data.columns 
        if col.capitalize() in psychological_traits or col in psychological_traits
    ]

    results: List[Dict[str, Any]] = []

    for metric in visual_metrics:
        for trait in available_traits:
            valid_subset: pd.DataFrame = merged_data[[metric, trait]].dropna()
            
            if len(valid_subset) < 2:
                continue

            vector_x: np.ndarray = valid_subset[metric].to_numpy(dtype=float)
            vector_y: np.ndarray = valid_subset[trait].to_numpy(dtype=float)

            pearson_r, pearson_p = pearsonr(vector_x, vector_y)
            spearman_r, spearman_p = spearmanr(vector_x, vector_y)

            results.append({
                "Visual_Metric": metric,
                "Psychological_Trait": trait,
                "Pearson_r": round(float(pearson_r), 4),
                "Pearson_p_value": float(pearson_p),
                "Spearman_rho": round(float(spearman_r), 4),
                "Spearman_p_value": float(spearman_p)
            })

    return pd.DataFrame(results)


def main() -> None:
    """
    Coordinates dynamic ingestion, structural alignment, and correlation computation.
    """
    args: argparse.Namespace = parse_arguments()

    if not args.visual_metrics_path.is_file():
        raise FileNotFoundError(f"Visual metrics file not found at {args.visual_metrics_path}")

    try:
        visual_metrics: pd.DataFrame = pd.read_csv(args.visual_metrics_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read visual metrics: {exc}") from exc

    merged_data: pd.DataFrame = build_unified_dataset(
        metrics_df=visual_metrics, 
        annotations_dir=args.annotations_dir
    )

    if merged_data.empty:
        raise ValueError("Dataset build resulted in an empty matrix. Verify paths and identifiers.")

    correlation_matrix: pd.DataFrame = compute_correlations(merged_data=merged_data)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    correlation_matrix.to_csv(args.output_path, index=False)
    
    print(f"Correlation analysis complete. Exported to: {args.output_path.resolve()}")
    print("\n--- Correlation Summary ---")
    print(correlation_matrix.to_string(index=False))


if __name__ == "__main__":
    main()