import threading
from collections import deque
from typing import Tuple

from app.src.orchestration.types import VideoFrame


class FrameBuffer:
    """
    A thread-safe, bounded circular buffer for managing multimodal visual state.

    Automatically evicts the oldest frames when the maximum capacity is reached
    to prevent memory saturation during prolonged stream ingestion.
    """

    def __init__(self, max_capacity: int) -> None:
        """
        Initializes the frame buffer with a strict upper memory boundary.

        Args:
            max_capacity (int): The maximum number of frames to retain in memory.

        Raises:
            ValueError: If max_capacity is not a positive integer.
            TypeError: If max_capacity is not an integer type.
        """
        if not isinstance(max_capacity, int):
            raise TypeError(f"max_capacity must be an integer, got {type(max_capacity).__name__}.")
        if max_capacity <= 0:
            raise ValueError(f"max_capacity must be strictly positive, got {max_capacity}.")

        self._max_capacity: int = max_capacity
        self._buffer: deque[VideoFrame] = deque(maxlen=max_capacity)
        self._lock: threading.Lock = threading.Lock()

    def append(self, frame: VideoFrame) -> None:
        """
        Safely appends a new frame to the buffer.

        Args:
            frame (VideoFrame): The visual frame to store.

        Raises:
            TypeError: If the provided object is not a VideoFrame instance.
        """
        if not isinstance(frame, VideoFrame):
            raise TypeError(f"Expected VideoFrame instance, got {type(frame).__name__}.")

        with self._lock:
            self._buffer.append(frame)

    def get_frames_in_interval(
        self,
        start_sec: float,
        end_sec: float
    ) -> Tuple[VideoFrame, ...]:
        """
        Retrieves a contiguous sequence of frames falling within a temporal window.

        Extracts frames where the presentation timestamp strictly satisfies:
        start_sec <= timestamp_sec < end_sec.

        Args:
            start_sec (float): Lower temporal boundary in seconds (inclusive).
            end_sec (float): Upper temporal boundary in seconds (exclusive).

        Returns:
            Tuple[VideoFrame, ...]: Immutable sequence of bounded visual frames.

        Raises:
            ValueError: If temporal boundaries are inverted or negative.
        """
        if start_sec < 0.0:
            raise ValueError(f"start_sec must be non-negative, got {start_sec}.")
        if end_sec <= start_sec:
            raise ValueError(
                f"end_sec ({end_sec}) must be strictly greater than start_sec ({start_sec})."
            )

        with self._lock:
            return tuple(f for f in self._buffer if start_sec <= f.timestamp_sec < end_sec)

    def clear(self) -> None:
        """
        Atomically purges all frames from the buffer.
        """
        with self._lock:
            self._buffer.clear()

    @property
    def capacity(self) -> int:
        """
        Returns the absolute maximum frame capacity of the buffer.
        """
        return self._max_capacity

    @property
    def current_size(self) -> int:
        """
        Returns the exact number of frames currently held in the buffer.
        """
        with self._lock:
            return len(self._buffer)