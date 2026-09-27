from dataclasses import dataclass


@dataclass(frozen=True)
class OrchestratorConfig:
    """
    Configuration parameters for the edge orchestration pipeline.

    Attributes:
        audio_sample_rate (int): Target sampling rate for audio extraction.
        video_fps (float): Target frame rate for visual extraction.
        window_duration_sec (float): Total length of each temporal window slice.
        window_stride_sec (float): Temporal step size between consecutive windows.
        queue_timeout_sec (float): Maximum block time when pushing to a full queue.
    """
    audio_sample_rate: int = 16000
    video_fps: float = 50.0
    window_duration_sec: float = 2.0
    window_stride_sec: float = 0.5
    queue_timeout_sec: float = 5.0

    def __post_init__(self) -> None:
        """
        Validates structural types and mathematical constraints of the configuration.

        Raises:
            TypeError: If fields have wrong types.
            ValueError: If any numeric constraints (e.g., negative durations) are violated.
        """
        if not isinstance(self.audio_sample_rate, int):
            raise TypeError(f"audio_sample_rate must be int, got {type(self.audio_sample_rate).__name__}.")
        if self.audio_sample_rate <= 0:
            raise ValueError(f"audio_sample_rate must be positive, got {self.audio_sample_rate}.")

        if not isinstance(self.video_fps, float):
            raise TypeError(f"video_fps must be float, got {type(self.video_fps).__name__}.")
        if self.video_fps <= 0.0:
            raise ValueError(f"video_fps must be positive, got {self.video_fps}.")

        if not isinstance(self.window_duration_sec, float):
            raise TypeError(f"window_duration_sec must be float, got {type(self.window_duration_sec).__name__}.")
        if self.window_duration_sec <= 0.0:
            raise ValueError(f"window_duration_sec must be positive, got {self.window_duration_sec}.")

        if not isinstance(self.window_stride_sec, float):
            raise TypeError(f"window_stride_sec must be float, got {type(self.window_stride_sec).__name__}.")
        if self.window_stride_sec <= 0.0:
            raise ValueError(f"window_stride_sec must be positive, got {self.window_stride_sec}.")

        if not isinstance(self.queue_timeout_sec, float):
            raise TypeError(f"queue_timeout_sec must be float, got {type(self.queue_timeout_sec).__name__}.")
        if self.queue_timeout_sec <= 0.0:
            raise ValueError(f"queue_timeout_sec must be positive, got {self.queue_timeout_sec}.")