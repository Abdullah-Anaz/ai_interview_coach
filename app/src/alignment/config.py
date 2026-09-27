from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AlignmentConfig:
    """
    Configuration matrix for the Temporal Aligner's model and environment parameters.

    Attributes:
        model_repo_id (str): Hugging Face repository identifier.
        model_filename (str): The specific .nemo checkpoint filename.
        local_weights_dir (Path): Local directory to cache the downloaded model.
        device (str): Compute device identifier (e.g., 'cuda:0', 'cpu').
        use_float16 (bool): Flag to enforce half-precision for memory optimization.
        video_fps (int): Target frame rate for video synchronization.
        time_stride_sec (float): Step size for the temporal window in seconds.
    """
    model_repo_id: str = "nvidia/parakeet-ctc-0.6b"
    model_filename: str = "parakeet-ctc-0.6b.nemo"
    local_weights_dir: Path = Path("app/models/alignment")
    device: str = "cuda:0"
    use_float16: bool = True
    video_fps: int = 50
    time_stride_sec: float = 0.04

    def __post_init__(self) -> None:
        """
        Validates the type safety, structural integrity, and boundary conditions 
        of the alignment configuration matrix.

        Raises:
            TypeError: If fields fail strict type enforcement.
            ValueError: If fields contain empty, zero, or structurally invalid values.
        """
        if not isinstance(self.model_repo_id, str):
            raise TypeError("model_repo_id must be a string.")
        if not self.model_repo_id.strip():
            raise ValueError("model_repo_id must be a non-empty string.")

        if not isinstance(self.model_filename, str):
            raise TypeError("model_filename must be a string.")
        if not self.model_filename.strip():
            raise ValueError("model_filename must be a non-empty string.")

        if not isinstance(self.local_weights_dir, Path):
            raise TypeError("local_weights_dir must be a Path object.")

        if not isinstance(self.device, str):
            raise TypeError("device must be a string.")
        if not self.device.strip():
            raise ValueError("device must be a non-empty string.")

        if not isinstance(self.use_float16, bool):
            raise TypeError("use_float16 must be a boolean.")

        if not isinstance(self.video_fps, int):
            raise TypeError("video_fps must be an integer.")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be strictly positive.")

        if not isinstance(self.time_stride_sec, float):
            raise TypeError("time_stride_sec must be a float.")
        if self.time_stride_sec <= 0.0:
            raise ValueError("time_stride_sec must be strictly positive.")