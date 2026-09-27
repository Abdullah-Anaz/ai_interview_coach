from typing import Any, Iterator, List, Tuple, Union

import numpy as np
import pytest

from app.src.orchestration.types import (
    AudioChunk,
    SynchronizedWindowSlice,
    VideoFrame,
)
from app.src.orchestration.window_manager import (
    _extract_audio_slice,
    _extract_video_slice,
    generate_sliding_windows,
)


def _generate_synthetic_stream(
    duration_sec: float,
    chunk_duration_sec: float = 0.5
) -> Iterator[Union[VideoFrame, AudioChunk]]:
    """
    Constructs a deterministic, interleaved stream of pure data objects.

    Args:
        duration_sec (float): Total length of the simulated stream.
        chunk_duration_sec (float): Duration of each audio chunk packet.

    Yields:
        Union[VideoFrame, AudioChunk]: Sequential media packets.
    """
    fps: float = 50.0
    sample_rate: int = 16000
    current_time: float = 0.0
    frame_index: int = 0

    while current_time < duration_sec:
        audio_samples: int = int(chunk_duration_sec * sample_rate)
        yield AudioChunk(
            sample_rate=sample_rate,
            timestamp_sec=current_time,
            data=np.zeros((audio_samples,), dtype=np.float32)
        )

        frames_in_chunk: int = int(chunk_duration_sec * fps)
        for i in range(frames_in_chunk):
            yield VideoFrame(
                frame_index=frame_index,
                timestamp_sec=current_time + (i / fps),
                data=np.zeros((10, 10, 3), dtype=np.uint8)
            )
            frame_index += 1

        current_time += chunk_duration_sec


def test_extract_audio_slice_expected_behaviour_for_exact_bounds() -> None:
    """
    Validates precise extraction of an audio sub-array spanning multiple chunks.
    """
    chunks: List[AudioChunk] = [
        AudioChunk(sample_rate=16000, timestamp_sec=0.0, data=np.ones((8000,), dtype=np.float32)),
        AudioChunk(sample_rate=16000, timestamp_sec=0.5, data=np.ones((8000,), dtype=np.float32))
    ]

    slice_out: AudioChunk = _extract_audio_slice(chunks, start_sec=0.25, end_sec=0.75, sample_rate=16000)

    assert slice_out.timestamp_sec == 0.25
    assert len(slice_out.data) == 8000
    assert slice_out.sample_rate == 16000


def test_extract_audio_slice_expected_behaviour_for_empty_buffer() -> None:
    """
    Validates graceful fallback to an empty array when requested from an empty buffer.
    """
    slice_out: AudioChunk = _extract_audio_slice([], start_sec=1.0, end_sec=2.0, sample_rate=16000)

    assert slice_out.timestamp_sec == 1.0
    assert len(slice_out.data) == 0
    assert slice_out.sample_rate == 16000


def test_extract_audio_slice_expected_behaviour_for_out_of_bounds_request() -> None:
    """
    Validates array clamping when the requested interval exceeds available buffer data.
    """
    chunks: List[AudioChunk] = [
        AudioChunk(sample_rate=16000, timestamp_sec=0.0, data=np.ones((8000,), dtype=np.float32))
    ]

    slice_out: AudioChunk = _extract_audio_slice(chunks, start_sec=0.0, end_sec=2.0, sample_rate=16000)

    assert len(slice_out.data) == 8000


def test_extract_video_slice_expected_behaviour_for_boundary_filtering() -> None:
    """
    Validates strict [start_sec, end_sec) interval logic for video frames.
    """
    frames: List[VideoFrame] = [
            VideoFrame(frame_index=0, timestamp_sec=0.0, data=np.zeros((720, 1280, 3), dtype=np.uint8)),
            VideoFrame(frame_index=1, timestamp_sec=0.5, data=np.zeros((720, 1280, 3), dtype=np.uint8)),
            VideoFrame(frame_index=2, timestamp_sec=1.0, data=np.zeros((720, 1280, 3), dtype=np.uint8)),
            VideoFrame(frame_index=3, timestamp_sec=1.5, data=np.zeros((720, 1280, 3), dtype=np.uint8))
    ]

    filtered: Tuple[VideoFrame, ...] = _extract_video_slice(frames, start_sec=0.5, end_sec=1.5)

    assert len(filtered) == 2
    assert filtered[0].timestamp_sec == 0.5
    assert filtered[1].timestamp_sec == 1.0


def test_generate_sliding_windows_expected_behaviour_for_continuous_stream() -> None:
    """
    Validates correct emission of overlapping windows from a continuous multi-second stream.
    """
    stream: Iterator[Union[VideoFrame, AudioChunk]] = _generate_synthetic_stream(duration_sec=3.0)

    windows: List[SynchronizedWindowSlice] = list(
        generate_sliding_windows(stream, window_duration_sec=2.0, window_stride_sec=0.5)
    )

    assert len(windows) == 4
    
    assert windows[0].start_time_sec == 0.0
    assert windows[0].end_time_sec == 2.0
    assert not windows[0].is_terminal

    assert windows[1].start_time_sec == 0.5
    assert windows[1].end_time_sec == 2.5
    assert not windows[1].is_terminal

    assert windows[-1].start_time_sec == 1.5
    assert windows[-1].end_time_sec == 3.0
    assert windows[-1].is_terminal


def test_generate_sliding_windows_expected_behaviour_for_short_stream() -> None:
    """
    Validates that a stream shorter than the window duration still yields a valid terminal slice.
    """
    stream: Iterator[Union[VideoFrame, AudioChunk]] = _generate_synthetic_stream(duration_sec=1.0)

    windows: List[SynchronizedWindowSlice] = list(
        generate_sliding_windows(stream, window_duration_sec=2.0, window_stride_sec=0.5)
    )

    assert len(windows) == 1
    assert windows[0].start_time_sec == 0.0
    assert windows[0].end_time_sec == 1.0
    assert windows[0].is_terminal
    assert len(windows[0].audio.data) == 16000
    assert len(windows[0].frames) == 50


def test_generate_sliding_windows_raises_for_negative_duration() -> None:
    """
    Validates constraint enforcement for non-positive window durations.
    """
    stream: Iterator[Union[VideoFrame, AudioChunk]] = _generate_synthetic_stream(duration_sec=1.0)

    with pytest.raises(ValueError) as exc_info:
        list(generate_sliding_windows(stream, window_duration_sec=-1.0, window_stride_sec=0.5))

    assert "window_duration_sec must be positive" in str(exc_info.value)


def test_generate_sliding_windows_raises_for_zero_stride() -> None:
    """
    Validates constraint enforcement for non-positive window strides.
    """
    stream: Iterator[Union[VideoFrame, AudioChunk]] = _generate_synthetic_stream(duration_sec=1.0)

    with pytest.raises(ValueError) as exc_info:
        list(generate_sliding_windows(stream, window_duration_sec=2.0, window_stride_sec=0.0))

    assert "window_stride_sec must be positive" in str(exc_info.value)


def test_generate_sliding_windows_raises_for_invalid_packet_type() -> None:
    """
    Validates type safety guard when the stream yields an unmapped object type.
    """
    invalid_stream: List[Any] = ["this_is_a_string_not_a_packet"]

    with pytest.raises(TypeError) as exc_info:
        list(generate_sliding_windows(iter(invalid_stream), window_duration_sec=2.0, window_stride_sec=0.5))

    assert "Unrecognized stream packet type" in str(exc_info.value)