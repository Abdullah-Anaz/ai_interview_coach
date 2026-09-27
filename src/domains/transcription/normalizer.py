import sys
import json
import logging
from pathlib import Path
from typing import Union, List, Dict, Any

import pandas as pd
import jiwer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_normalization_pipeline() -> jiwer.Compose:
    """
    Creates and returns a jiwer text normalization pipeline.

    Returns:
        jiwer.Compose: The compiled sequence of jiwer text transformations.
    """
    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.SubstituteRegexes({
            r"\b(u+m+|h+m+|m+|u+h+)\b": "mmm",
            r"\b(a+h+)\b": "ah"
        }),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords()
    ])

def normalize_text(text: str, pipeline: jiwer.Compose) -> str:
    """
    Applies a jiwer transformation pipeline to a single text string.

    Args:
        text (str): The raw input text.
        pipeline (jiwer.Compose): The jiwer transformation pipeline.

    Returns:
        str: The normalized text string.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return ""
        
        transformed: List[List[str]] = pipeline(text)
        
        if not transformed or not transformed[0]:
            return ""
            
        return " ".join(transformed[0])
    except Exception as e:
        logging.error("Failed to normalize text '%s': %s", text, e)
        return ""

def compute_row_wer(ground_truth: str, prediction: str, pipeline: jiwer.Compose) -> float:
    """
    Computes the individual Word Error Rate (WER) between a ground truth and prediction.

    Args:
        ground_truth (str): The reference text.
        prediction (str): The hypothesis text.
        pipeline (jiwer.Compose): The normalization pipeline.

    Returns:
        float: The calculated WER score.
    """
    try:
        if not isinstance(ground_truth, str) or not isinstance(prediction, str):
            return 1.0
            
        if not ground_truth.strip():
            return 1.0 if prediction.strip() else 0.0
            
        return float(jiwer.wer(
            ground_truth,
            prediction,
            reference_transform=pipeline,
            hypothesis_transform=pipeline
        ))
    except Exception as e:
        logging.error("Failed to compute row WER: %s", e)
        return 1.0

def compute_aggregated_wer(references: List[str], predictions: List[str]) -> float:
    """
    Computes the aggregated WER across a dataset.

    Args:
        references (List[str]): List of normalized reference strings.
        predictions (List[str]): List of normalized hypothesis strings.

    Returns:
        float: The aggregated WER.
    """
    try:
        if not references:
            return 1.0
        return float(jiwer.wer(references, predictions))
    except Exception as e:
        logging.error("Failed to compute aggregated WER: %s", e)
        return 1.0

def generate_summary_metrics(
    df: pd.DataFrame, 
    model_col: str, 
    framework_col: str, 
    snr_col: str
) -> List[Dict[str, Any]]:
    """
    Generates a structured JSON-compatible summary of the WER results.

    Args:
        df (pd.DataFrame): The processed dataframe containing normalized columns.
        model_col (str): Column name for the model architecture.
        framework_col (str): Column name for the framework.
        snr_col (str): Column name for the target SNR.

    Returns:
        List[Dict[str, Any]]: A list of summary dictionaries per model.
    """
    summaries: List[Dict[str, Any]] = []
    
    for (model, framework), group in df.groupby([model_col, framework_col]):
        total_samples: int = len(group)
        
        valid_mask: pd.Series = (
            ~group["prediction"].astype(str).str.startswith("[ERROR") & 
            (group["normalized_ground_truth"] != "")
        )
        valid_group: pd.DataFrame = group[valid_mask]
        valid_samples: int = len(valid_group)
        
        overall_wer: float = compute_aggregated_wer(
            valid_group["normalized_ground_truth"].tolist(),
            valid_group["normalized_prediction"].tolist()
        )
        
        stratified_wer: List[Dict[str, Union[float, int]]] = []
        if snr_col in valid_group.columns:
            for snr, snr_group in valid_group.groupby(snr_col):
                stratum_wer: float = compute_aggregated_wer(
                    snr_group["normalized_ground_truth"].tolist(),
                    snr_group["normalized_prediction"].tolist()
                )
                stratified_wer.append({
                    "target_snr_db": float(snr),
                    "sample_count": int(len(snr_group)),
                    "wer": float(stratum_wer)
                })
                
        stratified_wer.sort(key=lambda x: x["target_snr_db"], reverse=True)
        
        summaries.append({
            "model_architecture": str(model),
            "framework": str(framework),
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "overall_wer": overall_wer,
            "stratified_wer": stratified_wer
        })
        
    return summaries

def process_evaluation_results(
    input_csv_path: Union[str, Path],
    output_csv_path: Union[str, Path],
    output_json_path: Union[str, Path],
    gt_column: str,
    pred_column: str,
    snr_col: str,
    framework_name: str,
    model_name: str
) -> None:
    """
    Reads results, applies normalization functionally, computes WER, and exports results.

    Args:
        input_csv_path (Union[str, Path]): Path to the input CSV.
        output_csv_path (Union[str, Path]): Path to save detailed results.
        output_json_path (Union[str, Path]): Path to save summary metrics.
        gt_column (str): Ground truth column.
        pred_column (str): Prediction column.
        snr_col (str): Target SNR column.
        framework_name (str): Hardcoded framework name to inject into the dataset.
        model_name (str): Hardcoded model architecture name to inject into the dataset.
    """
    try:
        input_path: Path = Path(input_csv_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        raw_df: pd.DataFrame = pd.read_csv(input_path)
        
        required_cols: set = {gt_column, pred_column, snr_col}
        missing_cols: set = required_cols - set(raw_df.columns)
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        pipeline: jiwer.Compose = get_normalization_pipeline()

        processed_df: pd.DataFrame = raw_df.assign(
            framework=framework_name,
            model_architecture=model_name,
            normalized_ground_truth=lambda df: df[gt_column].apply(
                lambda x: normalize_text(str(x), pipeline)
            ),
            normalized_prediction=lambda df: df[pred_column].apply(
                lambda x: normalize_text(str(x), pipeline)
            )
        ).assign(
            wer_normalized=lambda df: df.apply(
                lambda row: compute_row_wer(str(row[gt_column]), str(row[pred_column]), pipeline),
                axis=1
            )
        )

        csv_out: Path = Path(output_csv_path)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(csv_out, index=False)
        logging.info("Detailed results saved to %s", csv_out)

        summaries: List[Dict[str, Any]] = generate_summary_metrics(
            processed_df, 
            "model_architecture", 
            "framework", 
            snr_col
        )
        
        json_out: Path = Path(output_json_path)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=4)
        logging.info("Summary metrics saved to %s", json_out)

    except Exception as e:
        logging.error("An unexpected error occurred during processing: %s", e)
        raise

if __name__ == "__main__":
    INPUT_FILE: str = "results/transcription/transcription_whisper_results.csv"
    OUTPUT_CSV: str = "results/transcription/normalized/normalized_transcription_whisper_results.csv"
    OUTPUT_JSON: str = "results/transcription/normalized/evaluation_whisper_normalized_summary.json"

    GROUND_TRUTH_COL: str = "ground_truth_transcript"
    PREDICTION_COL: str = "prediction"
    TARGET_SNR_COL: str = "target_snr_db"
    
    FRAMEWORK_NAME: str = "huggingface"
    MODEL_NAME: str = "whisper-large-v3"
    
    try:
        process_evaluation_results(
            input_csv_path=INPUT_FILE,
            output_csv_path=OUTPUT_CSV,
            output_json_path=OUTPUT_JSON,
            gt_column=GROUND_TRUTH_COL, 
            pred_column=PREDICTION_COL,
            snr_col=TARGET_SNR_COL,
            framework_name=FRAMEWORK_NAME,
            model_name=MODEL_NAME
        )
    except Exception:
        logging.critical("Evaluation script failed.")
        sys.exit(1)