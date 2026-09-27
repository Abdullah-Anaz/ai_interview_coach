import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AcousticConfig:
    """
    Configuration parameters governing the acoustic analysis models and trait calibration.

    Attributes:
        wavlm_repo_id (str): Hugging Face repository ID for the WavLM encoder.
        wavlm_revision (str): Specific Git revision or branch for the model weights.
        ridge_model_path (Path): Path to the serialized scikit-learn Ridge model.
        calibration_path (Path): Path to the JSON file containing trait calibration boundaries.
        device (str): Compute device mapping (e.g., "cpu", "cuda:0").
        use_float16 (bool): Flag to enable half-precision tensor mathematics.
        local_weights_dir (Path): Local directory for caching model checkpoints.
        very_low_threshold (float): Upper boundary for the very low category.
        low_threshold (float): Upper boundary for the low category.
        high_threshold (float): Lower boundary for the high category.
        very_high_threshold (float): Lower boundary for the very high category.
    """
    wavlm_repo_id: str = "microsoft/wavlm-large"
    wavlm_revision: str = "refs/pr/7"
    ridge_model_path: Path = field(default_factory=lambda: Path("/home/abdullahanaz012/final_project/app/models/acoustic/ridge_ocean_regressor.joblib"))
    calibration_path: Path = field(default_factory=lambda: Path("/home/abdullahanaz012/final_project/app/models/acoustic/calibration.json"))
    device: str = "cuda:0"
    use_float16: bool = True
    local_weights_dir: Path = field(default_factory=lambda: Path("/home/abdullahanaz012/final_project/app/models/acoustic"))

    very_low_threshold: float = field(init=False)
    low_threshold: float = field(init=False)
    high_threshold: float = field(init=False)
    very_high_threshold: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Validates configuration attributes and dynamically loads calibration thresholds.

        Raises:
            TypeError: If attribute types violate defined constraints.
            ValueError: If string values are empty or thresholds are unordered.
            FileNotFoundError: If the calibration file does not exist.
            RuntimeError: If deserialization of the calibration file fails.
        """
        if not isinstance(self.wavlm_repo_id, str) or not self.wavlm_repo_id.strip():
            raise ValueError("wavlm_repo_id must be a non-empty string.")

        if not isinstance(self.wavlm_revision, str) or not self.wavlm_revision.strip():
            raise ValueError("wavlm_revision must be a non-empty string.")

        if not isinstance(self.ridge_model_path, Path):
            raise TypeError("ridge_model_path must be a pathlib.Path instance.")

        if not isinstance(self.calibration_path, Path):
            raise TypeError("calibration_path must be a pathlib.Path instance.")

        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string.")

        if not isinstance(self.use_float16, bool):
            raise TypeError("use_float16 must be a boolean.")

        if not isinstance(self.local_weights_dir, Path):
            raise TypeError("local_weights_dir must be a pathlib.Path instance.")

        if not self.calibration_path.exists():
            raise FileNotFoundError(f"Calibration file missing at {self.calibration_path}")

        try:
            with open(self.calibration_path, "r", encoding="utf-8") as file:
                calibration_data = json.load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load calibration JSON: {e}") from e

        required_keys = ("very_low_threshold", "low_threshold", "high_threshold", "very_high_threshold")
        for key in required_keys:
            if key not in calibration_data:
                raise ValueError(f"Calibration file missing required key: '{key}'")
            if not isinstance(calibration_data[key], (int, float)):
                raise TypeError(f"Threshold '{key}' must be a numerical value.")

        very_low: float = float(calibration_data["very_low_threshold"])
        low: float = float(calibration_data["low_threshold"])
        high: float = float(calibration_data["high_threshold"])
        very_high: float = float(calibration_data["very_high_threshold"])

        if not (very_low < low < high < very_high):
            raise ValueError(
                f"Thresholds must be strictly ascending: "
                f"{very_low} < {low} < {high} < {very_high}"
            )

        object.__setattr__(self, "very_low_threshold", very_low)
        object.__setattr__(self, "low_threshold", low)
        object.__setattr__(self, "high_threshold", high)
        object.__setattr__(self, "very_high_threshold", very_high)