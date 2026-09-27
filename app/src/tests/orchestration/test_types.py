from typing import Any, Tuple

import numpy as np
import pytest

from app.src.orchestration.types import (
    AudioChunk,
    SessionMetadata,
    SynchronizedWindowSlice,
    VideoFrame,
    WordTimestamp,
    TranscriptSegment,
    AlignedWord,
    SynchronizedSegment,
)


def test_SessionMetadata_expected_behaviour_for_valid_instance() -> None:
    """
    Validates that a correctly populated SessionMetadata instance passes validation seamlessly.
    """
    # Instantiation should succeed without raising any errors
    SessionMetadata(
        session_id="sess_123",
        question_id="q_1",
        question_text="Tell me about a time you failed.",
        target_competency="Resilience",
        timestamp_utc="2026-09-05T13:00:00Z"
    )


def test_SessionMetadata_raises_for_invalid_field_type() -> None:
    """
    Validates that passing non-string values to string fields raises a TypeError upon creation.
    """
    with pytest.raises(TypeError) as exc_info:
        SessionMetadata(
            session_id=123,  # type: ignore
            question_id="q_1",
            question_text="Tell me about a time you failed.",
            target_competency="Resilience",
            timestamp_utc="2026-09-05T13:00:00Z"
        )

    assert "must be of type str" in str(exc_info.value)


def test_SessionMetadata_raises_for_empty_string() -> None:
    """
    Validates that providing an empty or whitespace-only string raises a ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        SessionMetadata(
            session_id="sess_123",
            question_id="q_1",
            question_text="   ",
            target_competency="Resilience",
            timestamp_utc="2026-09-05T13:00:00Z"
        )

    assert "cannot be empty" in str(exc_info.value)


def test_VideoFrame_expected_behaviour_for_valid_instance() -> None:
    """
    Validates that a correctly formatted VideoFrame passes validation seamlessly.
    """
    VideoFrame(
        frame_index=0,
        timestamp_sec=0.033,
        data=np.zeros((720, 1280, 3), dtype=np.uint8)
    )


def test_VideoFrame_raises_for_negative_index() -> None:
    """
    Validates that a negative frame index raises a ValueError upon creation.
    """
    with pytest.raises(ValueError) as exc_info:
        VideoFrame(
            frame_index=-1,
            timestamp_sec=0.033,
            data=np.zeros((720, 1280, 3), dtype=np.uint8)
        )

    assert "frame_index must be a non-negative integer" in str(exc_info.value)


def test_VideoFrame_raises_for_negative_timestamp() -> None:
    """
    Validates that a negative timestamp raises a ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        VideoFrame(
            frame_index=1,
            timestamp_sec=-0.5,
            data=np.zeros((720, 1280, 3), dtype=np.uint8)
        )

    assert "timestamp_sec must be non-negative" in str(exc_info.value)


def test_VideoFrame_raises_for_invalid_dimensions() -> None:
    """
    Validates that a video frame array lacking exactly 3 channels raises a ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        VideoFrame(
            frame_index=0,
            timestamp_sec=0.0,
            data=np.zeros((720, 1280), dtype=np.uint8)
        )

    assert "must have 3 dimensions with shape (H, W, 3)" in str(exc_info.value)


def test_VideoFrame_raises_for_invalid_numpy_dtype() -> None:
    """
    Validates that a video frame array not utilizing uint8 raises a ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        VideoFrame(
            frame_index=0,
            timestamp_sec=0.0,
            data=np.zeros((720, 1280, 3), dtype=np.float32)
        )

    assert "must have dtype uint8" in str(exc_info.value)


def test_AudioChunk_expected_behaviour_for_valid_instance() -> None:
    """
    Validates that a correctly formatted AudioChunk passes validation seamlessly.
    """
    AudioChunk(
        sample_rate=16000,
        timestamp_sec=1.5,
        data=np.zeros((16000,), dtype=np.float32)
    )


def test_AudioChunk_raises_for_invalid_sample_rate() -> None:
    """
    Validates that a non-positive sample rate raises a ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        AudioChunk(
            sample_rate=0,
            timestamp_sec=0.0,
            data=np.zeros((1024,), dtype=np.float32)
        )

    assert "sample_rate must be a positive integer" in str(exc_info.value)


def test_AudioChunk_raises_for_invalid_dimensions() -> None:
    """
    Validates that an audio array with more than 1 dimension raises a ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        AudioChunk(
            sample_rate=16000,
            timestamp_sec=0.0,
            data=np.zeros((1024, 2), dtype=np.float32)
        )

    assert "must be a 1-dimensional array" in str(exc_info.value)


def test_AudioChunk_raises_for_invalid_numpy_dtype() -> None:
    """
    Validates that an audio array not utilizing float32 raises a ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        AudioChunk(
            sample_rate=16000,
            timestamp_sec=0.0,
            data=np.zeros((1024,), dtype=np.float64)
        )

    assert "must have dtype float32" in str(exc_info.value)


def test_SynchronizedWindowSlice_expected_behaviour_for_valid_instance() -> None:
    """
    Validates that a correctly constructed SynchronizedWindowSlice passes validation seamlessly.
    """
    chunk: AudioChunk = AudioChunk(
        sample_rate=16000,
        timestamp_sec=0.0,
        data=np.zeros((32000,), dtype=np.float32)
    )
    frames: Tuple[VideoFrame, ...] = (
        VideoFrame(frame_index=0, timestamp_sec=0.0, data=np.zeros((720, 1280, 3), dtype=np.uint8)),
        VideoFrame(frame_index=1, timestamp_sec=0.02, data=np.zeros((720, 1280, 3), dtype=np.uint8))
    )
    SynchronizedWindowSlice(
        window_id=0,
        start_time_sec=0.0,
        end_time_sec=2.0,
        audio=chunk,
        frames=frames,
        is_terminal=False
    )


def test_SynchronizedWindowSlice_raises_for_inverted_temporal_boundaries() -> None:
    """
    Validates that inverted or equal start and end times raise a ValueError upon creation.
    """
    chunk: AudioChunk = AudioChunk(
        sample_rate=16000,
        timestamp_sec=0.0,
        data=np.zeros((1024,), dtype=np.float32)
    )
    
    with pytest.raises(ValueError) as exc_info:
        SynchronizedWindowSlice(
            window_id=1,
            start_time_sec=2.0,
            end_time_sec=1.0,
            audio=chunk,
            frames=tuple(),
            is_terminal=False
        )

    assert "must be strictly greater than start_time_sec" in str(exc_info.value)


def test_SynchronizedWindowSlice_raises_for_non_tuple_frames() -> None:
    """
    Validates that providing frames as a list rather than a tuple raises a TypeError.
    """
    chunk: AudioChunk = AudioChunk(
        sample_rate=16000,
        timestamp_sec=0.0,
        data=np.zeros((1024,), dtype=np.float32)
    )
    frames: Any = [
        VideoFrame(frame_index=0, timestamp_sec=0.0, data=np.zeros((720, 1280, 3), dtype=np.uint8))
    ]
    
    with pytest.raises(TypeError) as exc_info:
        SynchronizedWindowSlice(
            window_id=1,
            start_time_sec=0.0,
            end_time_sec=2.0,
            audio=chunk,
            frames=frames,
            is_terminal=False
        )

    assert "frames must be provided as a tuple" in str(exc_info.value)


def test_WordTimestamp_expected_behaviour_for_valid_instance() -> None:
    WordTimestamp(word="um", start_time_sec=1.0, end_time_sec=1.5, confidence=0.95)


def test_WordTimestamp_raises_for_inverted_times() -> None:
    with pytest.raises(ValueError) as exc_info:
        WordTimestamp(word="hello", start_time_sec=2.0, end_time_sec=1.5, confidence=0.99)
        
    assert "strictly greater than" in str(exc_info.value)


def test_TranscriptSegment_expected_behaviour_for_valid_sequence() -> None:
    words = (
        WordTimestamp(word="hello", start_time_sec=0.0, end_time_sec=0.5, confidence=0.99),
        WordTimestamp(word="world", start_time_sec=0.6, end_time_sec=1.0, confidence=0.98)
    )
    TranscriptSegment(segment_id=0, words=words, is_final=True)


def test_TranscriptSegment_raises_for_negative_segment_id() -> None:
    with pytest.raises(ValueError) as exc_info:
        TranscriptSegment(segment_id=-1, words=tuple(), is_final=True)
        
    assert "segment_id must be a non-negative integer" in str(exc_info.value)


def test_AlignedWord_expected_behaviour_for_valid_input() -> None:
    """
    Validates that a correctly formatted AlignedWord instantiates cleanly.
    """
    word = AlignedWord(
        word="synergy",
        start_time_sec=1.5,
        end_time_sec=2.0,
        start_frame_idx=75,
        end_frame_idx=100,
        confidence=0.98
    )
    
    assert word.word == "synergy"
    assert word.start_frame_idx == 75


def test_AlignedWord_raises_for_invalid_time_boundaries() -> None:
    """
    Validates constraints preventing paradoxical time definitions (start > end).
    """
    with pytest.raises(ValueError) as exc_info:
        AlignedWord(
            word="drift",
            start_time_sec=3.0,
            end_time_sec=2.0,
            start_frame_idx=150,
            end_frame_idx=200,
            confidence=0.9
        )
        
    assert "start_time_sec cannot exceed end_time_sec" in str(exc_info.value)


def test_AlignedWord_raises_for_invalid_frame_boundaries() -> None:
    """
    Validates constraints preventing negative frame indices.
    """
    with pytest.raises(ValueError) as exc_info:
        AlignedWord(
            word="offset",
            start_time_sec=0.0,
            end_time_sec=0.5,
            start_frame_idx=-5,
            end_frame_idx=25,
            confidence=0.99
        )
        
    assert "Visual frame indices must be non-negative" in str(exc_info.value)


def test_SynchronizedSegment_expected_behaviour_for_valid_input() -> None:
    """
    Validates structural integrity of the macro payload containing multiple aligned words.
    """
    word_1 = AlignedWord("hello", 0.0, 0.5, 0, 25, 0.99)
    word_2 = AlignedWord("world", 0.5, 1.0, 25, 50, 0.95)
    
    segment = SynchronizedSegment(
        segment_id=1,
        words=(word_1, word_2),
        audio_slice_start_sec=0.0,
        audio_slice_end_sec=1.0,
        visual_frame_start_idx=0,
        visual_frame_end_idx=50,
        is_final=False
    )
    
    assert segment.segment_id == 1
    assert len(segment.words) == 2


def test_SynchronizedSegment_raises_for_invalid_words_type() -> None:
    """
    Validates type safety preventing mutation (requires tuple, rejects list).
    """
    word = AlignedWord("fail", 0.0, 0.5, 0, 25, 0.9)
    
    with pytest.raises(TypeError) as exc_info:
        SynchronizedSegment(
            segment_id=1,
            words=[word],  # type: ignore 
            audio_slice_start_sec=0.0,
            audio_slice_end_sec=0.5,
            visual_frame_start_idx=0,
            visual_frame_end_idx=25,
            is_final=True
        )
        
    assert "words must be a tuple" in str(exc_info.value)


def test_SynchronizedSegment_raises_for_paradoxical_audio_slice() -> None:
    """
    Validates boundary logic on the overarching audio slice limits.
    """
    with pytest.raises(ValueError) as exc_info:
        SynchronizedSegment(
            segment_id=2,
            words=(),
            audio_slice_start_sec=10.0,
            audio_slice_end_sec=5.0,
            visual_frame_start_idx=500,
            visual_frame_end_idx=250,
            is_final=False
        )
        
    assert "audio_slice_start_sec cannot exceed audio_slice_end_sec" in str(exc_info.value)