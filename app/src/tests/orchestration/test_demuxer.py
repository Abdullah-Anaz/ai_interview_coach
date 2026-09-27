from pathlib import Path
from typing import List

import numpy as np
import pytest

from app.src.orchestration.demuxer import demux_container
from app.src.orchestration.types import AudioChunk, VideoFrame


def test_demux_container_expected_behaviour_for_standard_media() -> None:
    """
    Validates standard extraction yielding interleaved video and audio chunks from a real MP4.
    """
    media_path = Path("app/src/tests/data/standard_interview.mp4")
    
    results: List = list(demux_container(media_path, target_fps=50.0, target_sample_rate=16000))
    
    video_frames: List[VideoFrame] = [r for r in results if isinstance(r, VideoFrame)]
    audio_chunks: List[AudioChunk] = [r for r in results if isinstance(r, AudioChunk)]
    
    assert len(video_frames) > 0
    assert len(audio_chunks) > 0
    assert video_frames[0].timestamp_sec == 0.0
    assert video_frames[0].data.ndim == 3
    assert audio_chunks[0].data.dtype == np.float32
    assert audio_chunks[0].sample_rate == 16000


def test_demux_container_expected_behaviour_for_video_only_stream() -> None:
    """
    Validates generator functionality when the container possesses only a video track.
    """
    media_path = Path("app/src/tests/data/silent_video.mp4")
    
    results: List = list(demux_container(media_path, target_fps=50.0, target_sample_rate=16000))
    
    video_frames: List[VideoFrame] = [r for r in results if isinstance(r, VideoFrame)]
    audio_chunks: List[AudioChunk] = [r for r in results if isinstance(r, AudioChunk)]
    
    assert len(video_frames) > 0
    assert len(audio_chunks) == 0


def test_demux_container_expected_behaviour_for_audio_only_stream() -> None:
    """
    Validates generator functionality when the container possesses only an audio track.
    """
    media_path = Path("app/src/tests/data/audio_only.wav")
    
    results: List = list(demux_container(media_path, target_fps=50.0, target_sample_rate=16000))
    
    video_frames: List[VideoFrame] = [r for r in results if isinstance(r, VideoFrame)]
    audio_chunks: List[AudioChunk] = [r for r in results if isinstance(r, AudioChunk)]
    
    assert len(video_frames) == 0
    assert len(audio_chunks) > 0


def test_demux_container_expected_behaviour_for_target_fps_upsampling() -> None:
    """
    Validates the zero-order hold temporal logic mapping a lower FPS source to 50Hz.
    """
    media_path = Path("app/src/tests/data/low_fps_video.mp4")
    
    results: List = list(demux_container(media_path, target_fps=50.0))
    video_frames: List[VideoFrame] = [r for r in results if isinstance(r, VideoFrame)]
    
    assert len(video_frames) >= 2
    time_delta: float = video_frames[1].timestamp_sec - video_frames[0].timestamp_sec
    assert np.isclose(time_delta, 0.02, atol=1e-4)


def test_demux_container_expected_behaviour_for_audio_resampling() -> None:
    """
    Validates the FFmpeg resampler correctly forces the target sample rate.
    """
    media_path = Path("app/src/tests/data/high_sample_rate.wav")
    
    results: List = list(demux_container(media_path, target_sample_rate=16000))
    audio_chunks: List[AudioChunk] = [r for r in results if isinstance(r, AudioChunk)]
    
    assert len(audio_chunks) > 0
    assert all(chunk.sample_rate == 16000 for chunk in audio_chunks)


def test_demux_container_expected_behaviour_for_short_duration_media() -> None:
    """
    Validates boundary condition handling for media files lasting less than one second.
    """
    media_path = Path("app/src/tests/data/fractional_second_video.mp4")
    
    results: List = list(demux_container(media_path))
    
    assert len(results) > 0


def test_demux_container_raises_for_nonexistent_file() -> None:
    """
    Validates upstream error propagation when PyAV encounters a missing file.
    """
    media_path = Path("app/src/tests/data/does_not_exist.mp4")
    
    with pytest.raises(RuntimeError) as exc_info:
        list(demux_container(media_path))
        
    assert "FFmpeg pipeline failure" in str(exc_info.value)


def test_demux_container_raises_for_corrupt_or_invalid_format() -> None:
    """
    Validates robust exception handling during invalid file parsing.
    """
    media_path = Path("app/src/tests/data/invalid_format.txt")
    
    with pytest.raises(RuntimeError) as exc_info:
        list(demux_container(media_path))
        
    assert "FFmpeg pipeline failure" in str(exc_info.value)


def test_demux_container_raises_for_empty_file() -> None:
    """
    Validates robust exception handling when attempting to parse a zero-byte file.
    """
    media_path = Path("app/src/tests/data/empty_file.mp4")
    
    with pytest.raises(RuntimeError) as exc_info:
        list(demux_container(media_path))
        
    assert "FFmpeg pipeline failure" in str(exc_info.value)