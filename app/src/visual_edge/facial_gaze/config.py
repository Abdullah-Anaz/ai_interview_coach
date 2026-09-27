import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class FacialConfig:
    """
    Immutable configuration matrix governing the Facial & Gaze POSTER++ inference node.
    Maps local neural weights, regression artifacts, and parses statistical quintile thresholds.
    """
    poster_weights_path: Path = Path("/home/abdullahanaz012/final_project/app/models/visual/poster_plus_latent.pth")
    regressor_weights_path: Path = Path("/home/abdullahanaz012/final_project/app/models/visual/facial_poster_regressor.joblib")
    calibration_path: Path = Path("/home/abdullahanaz012/final_project/app/models/visual/facial_calibration.json")
    device: str = "cuda:0"
    inference_batch_size: int = 16
    feature_dim: int = 1024
    calibration_thresholds: Dict[str, Dict[str, float]] = field(init=False)

    def __post_init__(self) -> None:
        """Enforces strict validation of file paths, device types, and calibration JSON boundaries."""
        if not isinstance(self.poster_weights_path, Path):
            raise TypeError("poster_weights_path must be a pathlib.Path instance.")
        if not isinstance(self.regressor_weights_path, Path):
            raise TypeError("regressor_weights_path must be a pathlib.Path instance.")
        if not isinstance(self.calibration_path, Path):
            raise TypeError("calibration_path must be a pathlib.Path instance.")

        if not self.poster_weights_path.is_file():
            raise FileNotFoundError(f"POSTER++ weights not found at {self.poster_weights_path}")
        if not self.regressor_weights_path.is_file():
            raise FileNotFoundError(f"Regressor weights not found at {self.regressor_weights_path}")
        if not self.calibration_path.is_file():
            raise FileNotFoundError(f"Calibration artifact not found at {self.calibration_path}")

        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string.")
        if not self.device.startswith(("cuda", "cpu", "mps")):
            raise ValueError("device must be a valid compute string (e.g., 'cuda:0').")

        if not isinstance(self.inference_batch_size, int) or isinstance(self.inference_batch_size, bool):
            raise TypeError("inference_batch_size must be an integer.")
        if self.inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be strictly positive.")

        if not isinstance(self.feature_dim, int) or isinstance(self.feature_dim, bool):
            raise TypeError("feature_dim must be an integer.")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be strictly positive.")

        try:
            with open(self.calibration_path, "r", encoding="utf-8") as file:
                data: Any = json.load(file)
        except Exception as exc:
            raise RuntimeError(f"Failed to load calibration JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Calibration JSON root must resolve to a standard dictionary.")

        required_keys = ["very_low_threshold", "low_threshold", "high_threshold", "very_high_threshold"]
        validated_thresholds: Dict[str, Dict[str, float]] = {}

        for trait, thresholds in data.items():
            if not isinstance(thresholds, dict):
                raise ValueError(f"Calibration entry for trait '{trait}' must be a dictionary.")

            for key in required_keys:
                if key not in thresholds:
                    raise ValueError(f"Trait '{trait}' missing required threshold key: '{key}'")
                val = thresholds[key]
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise TypeError(f"Threshold '{key}' for trait '{trait}' must be a numerical float.")

            v_low = float(thresholds["very_low_threshold"])
            low = float(thresholds["low_threshold"])
            high = float(thresholds["high_threshold"])
            v_high = float(thresholds["very_high_threshold"])

            if not (v_low < low < high < v_high):
                raise ValueError(f"Thresholds for trait '{trait}' must be strictly ascending.")

            validated_thresholds[str(trait)] = {
                "very_low_threshold": v_low,
                "low_threshold": low,
                "high_threshold": high,
                "very_high_threshold": v_high
            }

        object.__setattr__(self, "calibration_thresholds", validated_thresholds)