from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class IngestionMode(str, Enum):
    """
    Enumeration defining the upstream data ingestion strategy.
    """
    LIVE = "live"
    SIMULATOR = "simulator"


@dataclass(frozen=True)
class IngestionConfig:
    """
    Configuration matrix governing the frontend media capture and synchronization parameters.

    Attributes:
        mode (IngestionMode): The active routing strategy (live hardware vs. file simulation).
        video_fps (float): The target visual extraction frequency.
        audio_sample_rate (int): The target acoustic extraction frequency in Hertz.
        camera_device_index (int): The zero-based OS hardware identifier for the primary webcam.
        audio_device_index (int): The zero-based OS hardware identifier for the primary microphone.
        simulator_file_path (Optional[Path]): The absolute path to the test video if running in SIMULATOR mode.
    """
    mode: IngestionMode
    video_fps: float = 50.0
    audio_sample_rate: int = 16000
    camera_device_index: int = 0
    audio_device_index: int = 0
    simulator_file_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """
        Enforces strict runtime validation of hardware parameters and strategy dependencies.

        Raises:
            TypeError: If configuration parameters violate strict type bounds.
            ValueError: If numeric constraints are breached or required strategy dependencies are missing.
        """
        if not isinstance(self.mode, IngestionMode):
            raise TypeError("mode must be a valid IngestionMode enum.")
        
        if not isinstance(self.video_fps, float):
            raise TypeError("video_fps must be a strictly typed float.")
        if self.video_fps <= 0.0:
            raise ValueError("video_fps must be strictly positive.")

        if not isinstance(self.audio_sample_rate, int) or isinstance(self.audio_sample_rate, bool):
            raise TypeError("audio_sample_rate must be a strictly typed integer.")
        if self.audio_sample_rate <= 0:
            raise ValueError("audio_sample_rate must be strictly positive.")

        if not isinstance(self.camera_device_index, int) or isinstance(self.camera_device_index, bool):
            raise TypeError("camera_device_index must be a strictly typed integer.")
        if self.camera_device_index < 0:
            raise ValueError("camera_device_index must be a non-negative integer.")

        if not isinstance(self.audio_device_index, int) or isinstance(self.audio_device_index, bool):
            raise TypeError("audio_device_index must be a strictly typed integer.")
        if self.audio_device_index < 0:
            raise ValueError("audio_device_index must be a non-negative integer.")

        if self.mode == IngestionMode.SIMULATOR:
            if self.simulator_file_path is None:
                raise ValueError("simulator_file_path is explicitly required when mode is SIMULATOR.")
            if not isinstance(self.simulator_file_path, Path):
                raise TypeError("simulator_file_path must be a valid pathlib.Path object.")