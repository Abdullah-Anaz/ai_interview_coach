from typing import Tuple

import numpy as np
import pytest

from app.src.orchestration.frame_buffer import FrameBuffer
from app.src.orchestration.types import VideoFrame


def _create_dummy_frame(index: int, timestamp: float) -> VideoFrame:
    """
    Constructs a deterministic synthetic VideoFrame for testing.
    """
    return VideoFrame(
        frame_index=index,
        timestamp_sec=timestamp,
        data=np.zeros((10, 10, 3), dtype=np.uint8)
    )


def test_init_expected_behaviour_for_valid_capacity() -> None:
    """
    Validates correct state initialization when provided a positive integer capacity.
    """
    buffer: FrameBuffer = FrameBuffer(max_capacity=500)
    
    assert buffer.capacity == 500
    assert buffer.current_size == 0


def test_init_raises_for_invalid_type() -> None:
    """
    Validates type safety guard when instantiating with a non-integer capacity.
    """
    with pytest.raises(TypeError) as exc_info:
        FrameBuffer(max_capacity=500.5)  # type: ignore
        
    assert "max_capacity must be an integer" in str(exc_info.value)


def test_init_raises_for_non_positive_capacity() -> None:
    """
    Validates constraint enforcement against zero or negative maximum capacities.
    """
    with pytest.raises(ValueError) as exc_info:
        FrameBuffer(max_capacity=0)
        
    assert "max_capacity must be strictly positive" in str(exc_info.value)


def test_append_expected_behaviour_for_normal_insertion() -> None:
    """
    Validates that a valid VideoFrame is successfully ingested and updates the internal state.
    """
    buffer: FrameBuffer = FrameBuffer(max_capacity=10)
    frame: VideoFrame = _create_dummy_frame(0, 0.0)
    
    buffer.append(frame)
    
    assert buffer.current_size == 1


def test_append_expected_behaviour_for_circular_eviction() -> None:
    """
    Validates the core circular buffer mechanic: evicting the oldest frames upon capacity overflow.
    """
    buffer: FrameBuffer = FrameBuffer(max_capacity=3)
    
    buffer.append(_create_dummy_frame(0, 0.0))
    buffer.append(_create_dummy_frame(1, 1.0))
    buffer.append(_create_dummy_frame(2, 2.0))
    
    assert buffer.current_size == 3
    
    buffer.append(_create_dummy_frame(3, 3.0))
    
    assert buffer.current_size == 3
    
    frames: Tuple[VideoFrame, ...] = buffer.get_frames_in_interval(0.0, 5.0)
    assert len(frames) == 3
    assert frames[0].frame_index == 1
    assert frames[-1].frame_index == 3


def test_append_raises_for_invalid_frame_type() -> None:
    """
    Validates type safety guard preventing ingestion of non-VideoFrame objects.
    """
    buffer: FrameBuffer = FrameBuffer(max_capacity=10)
    
    with pytest.raises(TypeError) as exc_info:
        buffer.append({"frame_index": 0})  # type: ignore
        
    assert "Expected VideoFrame instance" in str(exc_info.value)


def test_get_frames_in_interval_expected_behaviour_for_valid_range() -> None:
    """
    Validates accurate filtering of frames falling strictly within [start_sec, end_sec).
    """
    buffer: FrameBuffer = FrameBuffer(max_capacity=10)
    buffer.append(_create_dummy_frame(0, 0.0))
    buffer.append(_create_dummy_frame(1, 0.5))
    buffer.append(_create_dummy_frame(2, 1.0))
    buffer.append(_create_dummy_frame(3, 1.5))
    
    frames: Tuple[VideoFrame, ...] = buffer.get_frames_in_interval(0.5, 1.5)
    
    assert len(frames) == 2
    assert frames[0].frame_index == 1
    assert frames[1].frame_index == 2


def test_get_frames_in_interval_expected_behaviour_for_empty_match() -> None:
    """
    Validates that querying a temporal interval containing no frames yields an empty tuple.
    """
    buffer: FrameBuffer = FrameBuffer(max_capacity=10)
    buffer.append(_create_dummy_frame(0, 0.0))
    
    frames: Tuple[VideoFrame, ...] = buffer.get_frames_in_interval(1.0, 2.0)
    
    assert len(frames) == 0


def test_get_frames_in_interval_raises_for_negative_start() -> None:
    """
    Validates temporal boundary enforcement against negative start timestamps.
    """
    buffer: FrameBuffer = FrameBuffer(max_capacity=10)
    
    with pytest.raises(ValueError) as exc_info:
        buffer.get_frames_in_interval(-1.0, 2.0)
        
    assert "start_sec must be non-negative" in str(exc_info.value)


def test_get_frames_in_interval_raises_for_inverted_bounds() -> None:
    """
    Validates constraint enforcement preventing mathematically inverted temporal queries.
    """
    buffer: FrameBuffer = FrameBuffer(max_capacity=10)
    
    with pytest.raises(ValueError) as exc_info:
        buffer.get_frames_in_interval(2.0, 1.0)
        
    assert "must be strictly greater than start_sec" in str(exc_info.value)


def test_clear_expected_behaviour_for_purging_buffer() -> None:
    """
    Validates that the clear method atomically resets the buffer state to empty.
    """
    buffer: FrameBuffer = FrameBuffer(max_capacity=10)
    buffer.append(_create_dummy_frame(0, 0.0))
    buffer.append(_create_dummy_frame(1, 0.5))
    
    assert buffer.current_size == 2
    
    buffer.clear()
    
    assert buffer.current_size == 0


def test_current_size_expected_behaviour_for_state_tracking() -> None:
    """
    Validates that the current_size property tracks actual length accurately throughout lifecycles.
    """
    buffer: FrameBuffer = FrameBuffer(max_capacity=2)
    
    assert buffer.current_size == 0
    buffer.append(_create_dummy_frame(0, 0.0))
    assert buffer.current_size == 1
    buffer.append(_create_dummy_frame(1, 1.0))
    assert buffer.current_size == 2
    buffer.append(_create_dummy_frame(2, 2.0))
    assert buffer.current_size == 2