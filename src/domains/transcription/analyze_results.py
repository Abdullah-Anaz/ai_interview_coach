import json
import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_wer_stability_metrics(
    df: pd.DataFrame, 
    wer_column: str
) -> Dict[str, Union[float, int]]:
    """
    Computes stability and hallucination metrics from a distribution of Word Error Rates.

    Args:
        df (pd.DataFrame): The dataset containing the transcription results.
        wer_column (str): The specific column name housing the numerical WER values.

    Returns:
        Dict[str, Union[float, int]]: A dictionary mapping metric names to their computed values.

    Raises:
        KeyError: If the specified WER column is missing from the DataFrame.
        RuntimeError: If statistical computation fails on the provided data.
    """
    try:
        if wer_column not in df.columns:
            raise KeyError(f"Target metric column '{wer_column}' not found in DataFrame.")

        wer_series: pd.Series = df[wer_column].dropna()

        return {
            "std_dev": float(wer_series.std()),
            "extreme_failures_over_1_0": int((wer_series > 1.0).sum()),
            "severe_hallucinations_over_2_0": int((wer_series > 2.0).sum()),
            "maximum_wer": float(wer_series.max())
        }

    except Exception as e:
        logger.error("Failed to compute stability metrics: %s", e)
        raise RuntimeError(f"Metric computation failed: {e}") from e


def analyze_transcription_directory(
    directory_path: Path, 
    wer_column: str
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Iterates over all CSV files in a directory to compute WER stability metrics for each.

    Args:
        directory_path (Path): The file system path to the directory containing normalized CSV results.
        wer_column (str): The column name housing the WER values to analyze.

    Returns:
        Dict[str, Dict[str, Union[float, int]]]: A nested dictionary mapping filenames to their respective metrics.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        RuntimeError: If file ingestion or processing fails.
    """
    try:
        if not directory_path.exists() or not directory_path.is_dir():
            raise FileNotFoundError(f"Result directory missing or invalid: {directory_path}")

        csv_files: List[Path] = list(directory_path.glob("*.csv"))
        results: Dict[str, Dict[str, Union[float, int]]] = {}

        for file_path in csv_files:
            df: pd.DataFrame = pd.read_csv(file_path)
            metrics: Dict[str, Union[float, int]] = compute_wer_stability_metrics(df, wer_column)
            results[file_path.name] = metrics

        return results

    except Exception as e:
        logger.error("Failed to process directory %s: %s", directory_path, e)
        raise RuntimeError(f"Directory analysis failed: {e}") from e


def save_metrics_to_json(
    metrics_data: Dict[str, Dict[str, Union[float, int]]], 
    output_path: Path
) -> None:
    """
    Persists the computed metrics dictionary to a JSON file.

    Args:
        metrics_data (Dict[str, Dict[str, Union[float, int]]]): The nested metrics payload.
        output_path (Path): The destination path for the JSON file.

    Raises:
        RuntimeError: If file creation or JSON serialization fails.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=4)
        logger.info("Successfully persisted metrics to %s", output_path)
    except Exception as e:
        logger.error("Failed to save metrics to JSON: %s", e)
        raise RuntimeError(f"JSON persistence failed: {e}") from e


if __name__ == "__main__":
    target_directory: Path = Path("src/results/transcription/normalized")
    target_metric_column: str = "wer_normalized"
    output_json_path: Path = Path("src/results/transcription/normalized/hallucination_metrics_summary.json")

    try:
        analysis_results: Dict[str, Dict[str, Union[float, int]]] = analyze_transcription_directory(
            directory_path=target_directory,
            wer_column=target_metric_column
        )

        for filename, model_metrics in analysis_results.items():
            print(f"\nResults for {filename}:")
            for metric_name, value in model_metrics.items():
                print(f"  - {metric_name}: {value}")

        save_metrics_to_json(
            metrics_data=analysis_results, 
            output_path=output_json_path
        )

    except Exception as fatal_error:
        logger.critical("Script execution terminated due to error: %s", fatal_error)