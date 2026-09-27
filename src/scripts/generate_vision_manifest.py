import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


def parse_pickle_payload(payload: Any) -> Dict[str, Dict[str, float]]:
    """
    Parses unpickled ChaLearn annotation objects into a normalized mapping.

    Args:
        payload (Any): Raw unpickled data structure.

    Returns:
        Dict[str, Dict[str, float]]: Standardized mapping of video names to trait scores.
    """
    trait_mapping: Dict[str, Dict[str, float]] = {}

    if isinstance(payload, pd.DataFrame):
        video_col: str = ""
        for col in payload.columns:
            if any(key in str(col).lower() for key in ["video", "name", "file"]):
                video_col = str(col)
                break
        if not video_col:
            video_col = str(payload.columns[0])

        for _, row in payload.iterrows():
            vid_name: str = str(row[video_col]).strip()
            if not vid_name.endswith(".mp4"):
                vid_name = f"{vid_name}.mp4"
            traits: Dict[str, float] = {}
            for col in payload.columns:
                if col != video_col:
                    try:
                        traits[str(col).strip().lower()] = float(row[col])
                    except (ValueError, TypeError):
                        pass
            trait_mapping[vid_name] = traits
        return trait_mapping

    if not isinstance(payload, dict):
        return trait_mapping

    first_val: Any = next(iter(payload.values())) if payload else None

    if isinstance(first_val, dict):
        first_inner_key: str = str(next(iter(first_val.keys()))) if first_val else ""
        if any(trait in first_inner_key.lower() for trait in ["open", "extra", "agree", "neuro", "conscien", "interview"]):
            for vid_name, traits_dict in payload.items():
                v_key: str = str(vid_name).strip()
                if not v_key.endswith(".mp4"):
                    v_key = f"{v_key}.mp4"
                trait_mapping[v_key] = {str(k).strip().lower(): float(v) for k, v in traits_dict.items() if isinstance(v, (int, float))}
        else:
            for trait_name, vid_scores in payload.items():
                t_key: str = str(trait_name).strip().lower()
                if isinstance(vid_scores, dict):
                    for vid_name, score in vid_scores.items():
                        v_key = str(vid_name).strip()
                        if not v_key.endswith(".mp4"):
                            v_key = f"{v_key}.mp4"
                        if v_key not in trait_mapping:
                            trait_mapping[v_key] = {}
                        try:
                            trait_mapping[v_key][t_key] = float(score)
                        except (ValueError, TypeError):
                            pass
    return trait_mapping


def load_ground_truth_mapping(ground_truth_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Parses ground truth directory scanning for PKL and CSV annotation files.

    Args:
        ground_truth_dir (str): Directory containing ground truth annotations.

    Returns:
        Dict[str, Dict[str, float]]: A mapping of video filename to trait scores.
    """
    trait_mapping: Dict[str, Dict[str, float]] = {}
    dir_path: Path = Path(ground_truth_dir)

    if not dir_path.exists() or not dir_path.is_dir():
        logger.error("Ground truth directory not found: %s", ground_truth_dir)
        return trait_mapping

    pkl_files: List[Path] = list(dir_path.rglob("*.pkl"))
    for file_path in pkl_files:
        try:
            with open(file_path, "rb") as f:
                try:
                    payload: Any = pickle.load(f)
                except UnicodeDecodeError:
                    f.seek(0)
                    payload = pickle.load(f, encoding="latin1")

            parsed: Dict[str, Dict[str, float]] = parse_pickle_payload(payload)
            for v_key, traits in parsed.items():
                if v_key not in trait_mapping:
                    trait_mapping[v_key] = {}
                trait_mapping[v_key].update(traits)
            logger.info("Parsed %d video records from %s", len(parsed), file_path)
        except Exception as exc:
            logger.error("Failed to parse pickle file %s: %s", file_path, exc)

    return trait_mapping


def extract_video_paths(source_dirs: List[str], extension: str = ".mp4") -> List[Path]:
    """
    Recursively scans source directory structures for target video files.

    Args:
        source_dirs (List[str]): Root directory paths to scan.
        extension (str, optional): Target file extension. Defaults to ".mp4".

    Returns:
        List[Path]: Consolidated list of resolved Path objects.
    """
    all_paths: List[Path] = []
    for dir_path_str in source_dirs:
        base_path: Path = Path(dir_path_str)
        if not base_path.exists() or not base_path.is_dir():
            logger.warning("Source directory not found or invalid: %s", dir_path_str)
            continue
        all_paths.extend(list(base_path.rglob(f"*{extension}")))
    return all_paths


def parse_metadata_from_path(
    file_path: Path, 
    trait_mapping: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """
    Formats path metadata and links available continuous trait annotations.

    Args:
        file_path (Path): Path of the video file.
        trait_mapping (Dict[str, Dict[str, float]]): Loaded traits mapping.

    Returns:
        Dict[str, Any]: Consolidated metadata dictionary.
    """
    filename: str = file_path.name
    split_name: str = file_path.parent.name
    traits: Dict[str, float] = trait_mapping.get(filename, {})

    metadata: Dict[str, Any] = {
        "video_path": str(file_path.as_posix()),
        "video_name": filename,
        "split": split_name,
    }
    metadata.update(traits)
    return metadata


def generate_vision_manifest(
    source_dirs: List[str], 
    ground_truth_dir: str, 
    output_csv: str
) -> None:
    """
    Orchestrates dataset discovery, annotation alignment, and manifest serialization.

    Args:
        source_dirs (List[str]): Directories containing raw video files.
        ground_truth_dir (str): Root directory containing PKL ground truth files.
        output_csv (str): Output destination path for the generated CSV manifest.
    """
    try:
        trait_mapping: Dict[str, Dict[str, float]] = load_ground_truth_mapping(ground_truth_dir)
        if not trait_mapping:
            logger.warning("Trait mapping is empty. Ground truth variables will be missing.")

        video_files: List[Path] = extract_video_paths(source_dirs)
        if not video_files:
            logger.warning("No video files found in the provided directories.")
            return

        metadata_records: List[Dict[str, Any]] = [
            parse_metadata_from_path(p, trait_mapping) for p in video_files
        ]

        manifest_df: pd.DataFrame = pd.DataFrame(metadata_records)
        output_path: Path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(output_path, index=False)

        logger.info(
            "Successfully generated manifest with %d records at %s", 
            len(manifest_df), 
            output_csv
        )

    except Exception as exc:
        logger.error("Failed to generate vision manifest: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    target_directories: List[str] = [
        "src/data/chalearn"
    ]

    generate_vision_manifest(
        source_dirs=target_directories,
        ground_truth_dir="src/data/chalearn/annotations",
        output_csv="src/data/vision/manifest.csv"
    )