import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class SkeletalConfig:
    """
    Immutable configuration matrix governing the Human-Omni Skeletal Kinematics node.
    Maps local neural weights and parses statistical thresholds.

    Attributes:
        weights_path (Path): Local file system path to the .pth state dictionary.
        feature_dim (int): Latent projection dimensionality for the tri-branch model.
        num_joints (int): Target number of skeletal joints for regression.
        calibration_path (Path): Filesystem path to the skeletal calibration JSON.
        device (str): Compute device for inference.
        temporal_window_size (int): Frame buffer limit for temporal smoothing filters.
    """
    weights_path: Path = Path("/home/abdullahanaz012/final_project/app/models/visual/human_omni.pth")
    feature_dim: int = 512
    num_joints: int = 33
    calibration_path: Path = Path("/home/abdullahanaz012/final_project/app/models/visual/skeletal_calibration.json")
    device: str = "cuda:0"
    temporal_window_size: int = 50
    jitter_very_low: float = field(init=False)
    jitter_low: float = field(init=False)
    jitter_high: float = field(init=False)
    jitter_very_high: float = field(init=False)
    smoothness_very_low: float = field(init=False)
    smoothness_low: float = field(init=False)
    smoothness_high: float = field(init=False)
    smoothness_very_high: float = field(init=False)

    def __post_init__(self) -> None:
        """Enforces strict validation of JSON boundaries and numeric immutability."""
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Model weights missing at {self.weights_path}. Run generate_omni_weights.py first.")

        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string.")

        # Restored temporal window type and boundary validation
        if not isinstance(self.temporal_window_size, int) or isinstance(self.temporal_window_size, bool):
            raise TypeError("temporal_window_size must be an integer.")

        if self.temporal_window_size <= 0:
            raise ValueError("temporal_window_size must be strictly positive.")

        if not self.calibration_path.is_file():
            raise FileNotFoundError(f"Calibration file missing at {self.calibration_path}")

        try:
            with open(self.calibration_path, "r", encoding="utf-8") as file:
                data: Dict[str, Any] = json.load(file)
        except Exception as exc:
            raise RuntimeError(f"Failed to load calibration JSON: {exc}") from exc

        required_parents = ["perceptual_jitter", "temporal_smoothness"]
        required_keys = ["very_low_threshold", "low_threshold", "high_threshold", "very_high_threshold"]

        # 1. Hierarchical Check: Ensure all primary parents exist before extracting keys
        for parent in required_parents:
            if parent not in data or not isinstance(data[parent], dict):
                raise ValueError(f"Calibration JSON missing required parent object: '{parent}'")

        # 2. Value Extraction Check: Map threshold keys safely
        for parent, prefix in zip(required_parents, ["jitter", "smoothness"]):
            parent_data = data[parent]
            for key in required_keys:
                if key not in parent_data:
                    raise ValueError(f"'{parent}' missing required threshold key: '{key}'")
                
                value = parent_data[key]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"Threshold '{key}' in '{parent}' must be a numerical float.")

            v_low: float = float(parent_data["very_low_threshold"])
            low: float = float(parent_data["low_threshold"])
            high: float = float(parent_data["high_threshold"])
            v_high: float = float(parent_data["very_high_threshold"])

            if not (v_low < low < high < v_high):
                raise ValueError(f"Thresholds for '{parent}' must be strictly ascending.")

            object.__setattr__(self, f"{prefix}_very_low", v_low)
            object.__setattr__(self, f"{prefix}_low", low)
            object.__setattr__(self, f"{prefix}_high", high)
            object.__setattr__(self, f"{prefix}_very_high", v_high)