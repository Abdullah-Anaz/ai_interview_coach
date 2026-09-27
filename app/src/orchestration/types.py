from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SessionMetadata:
    """
    Metadata describing the current interview session and question context.

    Attributes:
        session_id (str): Unique identifier for the interview session.
        question_id (str): Identifier for the specific question prompt.
        question_text (str): Verbatim text of the question presented to the candidate.
        target_competency (str): Core competency or trait evaluated by this question.
        timestamp_utc (str): ISO 8601 UTC timestamp of session creation.
    """
    session_id: str
    question_id: str
    question_text: str
    target_competency: str
    timestamp_utc: str

    def __post_init__(self) -> None:
        """
        Validates structural completeness and types of SessionMetadata.

        Raises:
            TypeError: If any field violates strict string typing.
            ValueError: If any field is empty or solely whitespace.
        """
        fields = {
            "session_id": self.session_id,
            "question_id": self.question_id,
            "question_text": self.question_text,
            "target_competency": self.target_competency,
            "timestamp_utc": self.timestamp_utc,
        }
        for name, value in fields.items():
            if not isinstance(value, str):
                raise TypeError(f"Field '{name}' must be of type str.")
            if not value.strip():
                raise ValueError(f"Field '{name}' cannot be empty or solely whitespace.")


@dataclass(frozen=True)
class VideoFrame:
    """
    Representation of an individual visual frame normalized for downstream vision models.

    Attributes:
        frame_index (int): Monotonically increasing zero-based frame index.
        timestamp_sec (float): Presentation timestamp in seconds relative to recording start.
        data (np.ndarray): Decoded RGB array of shape (H, W, 3) and dtype uint8.
    """
    frame_index: int
    timestamp_sec: float
    data: np.ndarray

    def __post_init__(self) -> None:
        """
        Verifies that a VideoFrame complies with shape, dtype, and timestamp constraints.

        Raises:
            TypeError: If data is not a numpy ndarray.
            ValueError: If indices/timestamps are negative, or shape/dtype is invalid.
        """
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer.")
        if not isinstance(self.timestamp_sec, (int, float)) or self.timestamp_sec < 0.0:
            raise ValueError("timestamp_sec must be non-negative.")
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray.")
        if self.data.ndim != 3 or self.data.shape[2] != 3:
            raise ValueError("data must have 3 dimensions with shape (H, W, 3).")
        if self.data.dtype != np.uint8:
            raise ValueError("data must have dtype uint8.")


@dataclass(frozen=True)
class AudioChunk:
    """
    Representation of an audio slice normalized for acoustic and ASR models.

    Attributes:
        sample_rate (int): Sampling rate in Hertz (nominally 16000).
        timestamp_sec (float): Start time offset of this chunk in seconds.
        data (np.ndarray): 1D single-channel PCM float32 array normalized to [-1.0, 1.0].
    """
    sample_rate: int
    timestamp_sec: float
    data: np.ndarray

    def __post_init__(self) -> None:
        """
        Verifies that an AudioChunk complies with sample rate, dimension, and dtype constraints.

        Raises:
            TypeError: If data is not a numpy ndarray.
            ValueError: If rate/timestamp is invalid, or array shape/dtype is incorrect.
        """
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        if not isinstance(self.timestamp_sec, (int, float)) or self.timestamp_sec < 0.0:
            raise ValueError("timestamp_sec must be non-negative.")
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray.")
        if self.data.ndim != 1:
            raise ValueError("data must be a 1-dimensional array.")
        if self.data.dtype != np.float32:
            raise ValueError("data must have dtype float32.")


@dataclass(frozen=True)
class SynchronizedWindowSlice:
    """
    Aligned multimodal temporal window emitted to downstream analysis branches.

    Attributes:
        window_id (int): Zero-based sequential identifier of the sliding window.
        start_time_sec (float): Lower temporal boundary of the window in seconds.
        end_time_sec (float): Upper temporal boundary of the window in seconds.
        audio (AudioChunk): Audio slice spanning [start_time_sec, end_time_sec].
        frames (Tuple[VideoFrame, ...]): Sequence of visual frames captured within the interval.
        is_terminal (bool): True if this window represents the final slice of the recording.
    """
    window_id: int
    start_time_sec: float
    end_time_sec: float
    audio: AudioChunk
    frames: Tuple[VideoFrame, ...]
    is_terminal: bool = False

    def __post_init__(self) -> None:
        """
        Ensures temporal consistency and component validity of a SynchronizedWindowSlice.

        Raises:
            TypeError: If the container violates type signatures.
            ValueError: If temporal boundaries are inverted.
        """
        if not isinstance(self.window_id, int) or self.window_id < 0:
            raise ValueError("window_id must be a non-negative integer.")
        if self.start_time_sec < 0.0:
            raise ValueError("start_time_sec must be non-negative.")
        if self.end_time_sec <= self.start_time_sec:
            raise ValueError("end_time_sec must be strictly greater than start_time_sec.")
        if not isinstance(self.frames, tuple):
            raise TypeError("frames must be provided as a tuple.")


@dataclass(frozen=True)
class WordTimestamp:
    """
    Representation of a single transcribed word and its temporal boundaries.

    Attributes:
        word (str): The verbatim transcribed word or disfluency.
        start_time_sec (float): Start time of the word in seconds.
        end_time_sec (float): End time of the word in seconds.
        confidence (float): The model's confidence score for this token [0.0, 1.0].
    """
    word: str
    start_time_sec: float
    end_time_sec: float
    confidence: float

    def __post_init__(self) -> None:
        """
        Validates structural and temporal boundaries of a WordTimestamp.

        Raises:
            TypeError: If string typing is violated.
            ValueError: If bounds are inverted, confidence is out of range, or word is empty.
        """
        if not isinstance(self.word, str) or not self.word.strip():
            raise ValueError("word must be a non-empty string.")
        if self.start_time_sec < 0.0:
            raise ValueError("start_time_sec must be non-negative.")
        if self.end_time_sec <= self.start_time_sec:
            raise ValueError("end_time_sec must be strictly greater than start_time_sec.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0.")


@dataclass(frozen=True)
class TranscriptSegment:
    """
    A contiguous sequence of transcribed words mapping to a candidate's utterance.

    Attributes:
        segment_id (int): Zero-based sequential identifier for the utterance.
        words (Tuple[WordTimestamp, ...]): The sequence of timestamped words.
        is_final (bool): True if this represents a completed utterance boundary.
    """
    segment_id: int
    words: Tuple[WordTimestamp, ...]
    is_final: bool = False

    def __post_init__(self) -> None:
        """
        Validates temporal consistency across a sequence of transcribed words.

        Raises:
            TypeError: If the container violates type signatures.
            ValueError: If identifier is negative.
        """
        if not isinstance(self.segment_id, int) or self.segment_id < 0:
            raise ValueError("segment_id must be a non-negative integer.")
        if not isinstance(self.words, tuple):
            raise TypeError("words must be a tuple.")


@dataclass(frozen=True)
class AlignedWord:
    """
    Represents a single transcribed word flawlessly synchronized across time and visual frames.

    Attributes:
        word (str): The verbatim transcribed word or disfluency.
        start_time_sec (float): Absolute start time of the word.
        end_time_sec (float): Absolute end time of the word.
        start_frame_idx (int): Corresponding 50 Hz visual frame start index.
        end_frame_idx (int): Corresponding 50 Hz visual frame end index.
        confidence (float): Alignment confidence score [0.0, 1.0].
    """
    word: str
    start_time_sec: float
    end_time_sec: float
    start_frame_idx: int
    end_frame_idx: int
    confidence: float

    def __post_init__(self) -> None:
        """
        Validates the strict temporal and index boundaries of the synchronized word.

        Raises:
            TypeError: If word is not a string.
            ValueError: If temporal or frame index boundaries are negative or inverted.
        """
        if not isinstance(self.word, str):
            raise TypeError("word must be a string.")
        if self.start_time_sec < 0.0 or self.end_time_sec < 0.0:
            raise ValueError("Temporal boundaries must be non-negative.")
        if self.start_time_sec > self.end_time_sec:
            raise ValueError("start_time_sec cannot exceed end_time_sec.")
        if self.start_frame_idx < 0 or self.end_frame_idx < 0:
            raise ValueError("Visual frame indices must be non-negative.")
        if self.start_frame_idx > self.end_frame_idx:
            raise ValueError("start_frame_idx cannot exceed end_frame_idx.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be a probability between 0.0 and 1.0.")


@dataclass(frozen=True)
class SynchronizedSegment:
    """
    The unified multimodal payload distributed to the parallel analysis edges.

    Attributes:
        segment_id (int): Sequence identifier mapped to the source utterance.
        words (Tuple[AlignedWord, ...]): The strict sequence of aligned words.
        audio_slice_start_sec (float): Start bound for the corresponding acoustic slice.
        audio_slice_end_sec (float): End bound for the corresponding acoustic slice.
        visual_frame_start_idx (int): Start bound for the corresponding video frame array.
        visual_frame_end_idx (int): End bound for the corresponding video frame array.
        is_final (bool): Flag indicating if this completes an interaction turn.
    """
    segment_id: int
    words: Tuple[AlignedWord, ...]
    audio_slice_start_sec: float
    audio_slice_end_sec: float
    visual_frame_start_idx: int
    visual_frame_end_idx: int
    is_final: bool

    def __post_init__(self) -> None:
        """
        Ensures the overarching segment boundaries are logically sound.

        Raises:
            TypeError: If the tuple or boolean flag violate type constraints.
            ValueError: If indices/times are negative or inverted.
        """
        if self.segment_id < 0:
            raise ValueError("segment_id must be non-negative.")
        if not isinstance(self.words, tuple):
            raise TypeError("words must be a tuple of AlignedWord instances.")
        if self.audio_slice_start_sec < 0.0 or self.audio_slice_end_sec < 0.0:
            raise ValueError("Audio slice boundaries must be non-negative.")
        if self.audio_slice_start_sec > self.audio_slice_end_sec:
            raise ValueError("audio_slice_start_sec cannot exceed audio_slice_end_sec.")
        if self.visual_frame_start_idx < 0 or self.visual_frame_end_idx < 0:
            raise ValueError("Visual frame indices must be non-negative.")
        if self.visual_frame_start_idx > self.visual_frame_end_idx:
            raise ValueError("visual_frame_start_idx cannot exceed visual_frame_end_idx.")
        if not isinstance(self.is_final, bool):
            raise TypeError("is_final must be a boolean.")