from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptionConfig:
    """
    Hardware and caching configuration for the Transducer node.

    Attributes:
        model_repo_id (str): Hugging Face repository identifier.
        model_filename (str): The specific .nemo checkpoint filename.
        local_weights_dir (Path): Local directory to cache the downloaded model.
        device (str): Compute device identifier (e.g., 'cuda:0', 'cpu').
        use_float16 (bool): Flag to enforce half-precision for memory optimization.
    """
    model_repo_id: str = "nvidia/parakeet-tdt-0.6b-v3"
    model_filename: str = "parakeet-tdt-0.6b-v3.nemo"
    local_weights_dir: Path = Path("/home/abdullahanaz012/final_project/app/models/asr")
    device: str = "cuda:0"
    use_float16: bool = True

    def __post_init__(self) -> None:
        """
        Validates the structural integrity and bounds of the transcription configuration.

        Raises:
            TypeError: If fields are of incorrect types (like Path or boolean mismatches).
            ValueError: If string fields are empty or improperly typed.
        """
        if not isinstance(self.model_repo_id, str) or not self.model_repo_id.strip():
            raise ValueError("model_repo_id must be a non-empty string.")
        
        if not isinstance(self.model_filename, str) or not self.model_filename.strip():
            raise ValueError("model_filename must be a non-empty string.")
        
        if not isinstance(self.local_weights_dir, Path):
            raise TypeError(f"local_weights_dir must be a Path, got {type(self.local_weights_dir).__name__}.")
        
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string.")
        
        if not isinstance(self.use_float16, bool):
            raise TypeError(f"use_float16 must be a boolean, got {type(self.use_float16).__name__}.")