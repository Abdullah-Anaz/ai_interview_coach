from typing import Iterator, List, Tuple, Union

import numpy as np

from app.src.orchestration.types import (
    AudioChunk,
    SynchronizedWindowSlice,
    VideoFrame,
)


def _extract_audio_slice(
    chunks: List[AudioChunk],
    start_sec: float,
    end_sec: float,
    sample_rate: int
) -> AudioChunk:
    """
    Extracts a contiguous audio array for a specific temporal interval from buffered chunks.

    Args:
        chunks (List[AudioChunk]): Accumulated audio chunks.
        start_sec (float): Desired start time of the slice in seconds.
        end_sec (float): Desired end time of the slice in seconds.
        sample_rate (int): The audio sampling rate in Hertz.

    Returns:
        AudioChunk: A new bounded AudioChunk containing exactly the requested time slice.
    """
    if not chunks:
        return AudioChunk(
            sample_rate=sample_rate,
            timestamp_sec=start_sec,
            data=np.array([], dtype=np.float32)
        )

    first_chunk_start: float = chunks[0].timestamp_sec
    concatenated_data: np.ndarray = np.concatenate([c.data for c in chunks])

    start_idx: int = int(round((start_sec - first_chunk_start) * sample_rate))
    end_idx: int = int(round((end_sec - first_chunk_start) * sample_rate))

    start_idx = max(0, start_idx)
    end_idx = max(0, min(end_idx, len(concatenated_data)))

    return AudioChunk(
        sample_rate=sample_rate,
        timestamp_sec=start_sec,
        data=concatenated_data[start_idx:end_idx]
    )


def _extract_video_slice(
    frames: List[VideoFrame],
    start_sec: float,
    end_sec: float
) -> Tuple[VideoFrame, ...]:
    """
    Filters a sequence of visual frames falling strictly within a temporal interval.

    Args:
        frames (List[VideoFrame]): Accumulated video frames.
        start_sec (float): Lower temporal bound in seconds (inclusive).
        end_sec (float): Upper temporal bound in seconds (exclusive).

    Returns:
        Tuple[VideoFrame, ...]: A tuple of VideoFrame objects bounded by the interval.
    """
    return tuple(f for f in frames if start_sec <= f.timestamp_sec < end_sec)


def generate_sliding_windows(
    stream: Iterator[Union[VideoFrame, AudioChunk]],
    window_duration_sec: float = 2.0,
    window_stride_sec: float = 0.5
) -> Iterator[SynchronizedWindowSlice]:
    """
    Consumes an interleaved media stream and emits overlapping synchronized windows.

    Args:
        stream (Iterator[Union[VideoFrame, AudioChunk]]): The normalized output of the demuxer.
        window_duration_sec (float): The total temporal length of each window slice.
        window_stride_sec (float): The temporal advancement step between consecutive windows.

    Yields:
        SynchronizedWindowSlice: Aligned audio and video structures.

    Raises:
        ValueError: If duration or stride constraints are violated.
        TypeError: If the input stream yields unrecognized objects.
    """
    if window_duration_sec <= 0.0:
        raise ValueError(f"window_duration_sec must be positive, got {window_duration_sec}")
    if window_stride_sec <= 0.0:
        raise ValueError(f"window_stride_sec must be positive, got {window_stride_sec}")

    audio_buffer: List[AudioChunk] = []
    video_buffer: List[VideoFrame] = []

    window_id: int = 0
    window_start_sec: float = 0.0
    max_audio_time_sec: float = 0.0
    sample_rate: int = 16000

    for packet in stream:
        if isinstance(packet, AudioChunk):
            sample_rate = packet.sample_rate
            audio_buffer.append(packet)
            duration: float = len(packet.data) / packet.sample_rate
            max_audio_time_sec = packet.timestamp_sec + duration
        elif isinstance(packet, VideoFrame):
            video_buffer.append(packet)
        else:
            raise TypeError(f"Unrecognized stream packet type: {type(packet).__name__}")

        while max_audio_time_sec >= window_start_sec + window_duration_sec:
            window_end_sec: float = window_start_sec + window_duration_sec

            audio_slice: AudioChunk = _extract_audio_slice(
                chunks=audio_buffer,
                start_sec=window_start_sec,
                end_sec=window_end_sec,
                sample_rate=sample_rate
            )
            video_slice: Tuple[VideoFrame, ...] = _extract_video_slice(
                frames=video_buffer,
                start_sec=window_start_sec,
                end_sec=window_end_sec
            )

            yield SynchronizedWindowSlice(
                window_id=window_id,
                start_time_sec=window_start_sec,
                end_time_sec=window_end_sec,
                audio=audio_slice,
                frames=video_slice,
                is_terminal=False
            )

            window_start_sec += window_stride_sec
            window_id += 1

            audio_buffer = [
                c for c in audio_buffer
                if (c.timestamp_sec + (len(c.data) / c.sample_rate)) > window_start_sec
            ]
            video_buffer = [
                f for f in video_buffer
                if f.timestamp_sec >= window_start_sec
            ]

    if audio_buffer and max_audio_time_sec > window_start_sec:
        audio_slice_terminal: AudioChunk = _extract_audio_slice(
            chunks=audio_buffer,
            start_sec=window_start_sec,
            end_sec=max_audio_time_sec,
            sample_rate=sample_rate
        )
        video_slice_terminal: Tuple[VideoFrame, ...] = _extract_video_slice(
            frames=video_buffer,
            start_sec=window_start_sec,
            end_sec=max_audio_time_sec
        )

        yield SynchronizedWindowSlice(
            window_id=window_id,
            start_time_sec=window_start_sec,
            end_time_sec=max_audio_time_sec,
            audio=audio_slice_terminal,
            frames=video_slice_terminal,
            is_terminal=True
        )