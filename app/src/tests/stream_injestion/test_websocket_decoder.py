import io
from typing import List, Union

import av
import numpy as np
import pytest

from app.src.orchestration.types import AudioChunk, VideoFrame
from app.src.stream_ingestion.websocket_decoder import decode_websocket_payload


# =====================================================================
# NATIVE MEDIA GENERATION FIXTURES (Bypassing FFmpeg Strictness)
# =====================================================================

@pytest.fixture
def multiplexed_media_bytes() -> bytes:
    """Synthesizes a valid, fully multiplexed WebM container natively."""
    buffer = io.BytesIO()
    with av.open(buffer, mode='w', format='webm') as container:
        video_stream = container.add_stream('vp8', rate=10)
        video_stream.width = 320
        video_stream.height = 240
        video_stream.pix_fmt = 'yuv420p'
        
        # Enforce experimental flag to bypass FFmpeg Vorbis compilation errors
        audio_stream = container.add_stream('vorbis', rate=16000, options={'strict': 'experimental'})
        
        for _ in range(5):
            v_frame = av.VideoFrame(width=320, height=240, format='yuv420p')
            for packet in video_stream.encode(v_frame):
                container.mux(packet)
                
            a_frame = av.AudioFrame(format='fltp', layout='mono', samples=1600)
            a_frame.sample_rate = 16000
            for packet in audio_stream.encode(a_frame):
                container.mux(packet)
                
        for packet in video_stream.encode():
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)
            
    return buffer.getvalue()


@pytest.fixture
def video_only_media_bytes() -> bytes:
    """Synthesizes a valid WebM container containing only a VP8 video stream."""
    buffer = io.BytesIO()
    with av.open(buffer, mode='w', format='webm') as container:
        video_stream = container.add_stream('vp8', rate=10)
        video_stream.width = 320
        video_stream.height = 240
        video_stream.pix_fmt = 'yuv420p'
        
        for _ in range(5):
            v_frame = av.VideoFrame(width=320, height=240, format='yuv420p')
            for packet in video_stream.encode(v_frame):
                container.mux(packet)
                
        for packet in video_stream.encode():
            container.mux(packet)
            
    return buffer.getvalue()


@pytest.fixture
def audio_only_media_bytes() -> bytes:
    """Synthesizes a valid WebM container containing only a Vorbis acoustic stream."""
    buffer = io.BytesIO()
    with av.open(buffer, mode='w', format='webm') as container:
        audio_stream = container.add_stream('vorbis', rate=16000, options={'strict': 'experimental'})
        
        for _ in range(5):
            a_frame = av.AudioFrame(format='fltp', layout='mono', samples=1600)
            a_frame.sample_rate = 16000
            for packet in audio_stream.encode(a_frame):
                container.mux(packet)
                
        for packet in audio_stream.encode():
            container.mux(packet)
            
    return buffer.getvalue()


@pytest.fixture
def empty_container_bytes() -> bytes:
    """
    Synthesizes an empty WebM container devoid of any valid A/V streams using a strict Matroska hex signature.
    This bypasses the initial 'empty byte string' check and forces PyAV to parse 0 available streams.
    """
    return bytes.fromhex(
        "1a45dfa39f4286810142f7810142f2810442f381084282847765626d"
        "42878102428581021853806780"
    )


# =====================================================================
# EXPECTED BEHAVIOUR & FORMATTING
# =====================================================================

def test_decode_websocket_payload_expected_behaviour_for_multiplexed_stream(
    multiplexed_media_bytes: bytes
) -> None:
    """Verifies that visual and acoustic frames are flawlessly extracted and cast to strict pipeline DTOs."""
    results: List[Union[VideoFrame, AudioChunk]] = list(
        decode_websocket_payload(
            payload=multiplexed_media_bytes,
            target_fps=20.0,
            target_sample_rate=16000
        )
    )
    
    assert len(results) > 0
    
    video_frames = [r for r in results if isinstance(r, VideoFrame)]
    audio_chunks = [r for r in results if isinstance(r, AudioChunk)]
    
    assert len(video_frames) > 0
    assert len(audio_chunks) > 0
    
    assert video_frames[0].data.shape == (240, 320, 3)
    assert video_frames[0].data.dtype == np.uint8
    
    assert audio_chunks[0].data.ndim == 1
    assert audio_chunks[0].data.dtype == np.float32


def test_decode_websocket_payload_expected_behaviour_for_video_only_stream(
    video_only_media_bytes: bytes
) -> None:
    """Verifies the pipeline extracts robust spatial matrices even if the audio component drops."""
    results: List[Union[VideoFrame, AudioChunk]] = list(
        decode_websocket_payload(payload=video_only_media_bytes)
    )
    
    video_frames = [r for r in results if isinstance(r, VideoFrame)]
    audio_chunks = [r for r in results if isinstance(r, AudioChunk)]
    
    assert len(video_frames) > 0
    assert len(audio_chunks) == 0


def test_decode_websocket_payload_expected_behaviour_for_audio_only_stream(
    audio_only_media_bytes: bytes
) -> None:
    """Verifies the pipeline extracts flawless acoustic chunks even if the camera matrix fails."""
    results: List[Union[VideoFrame, AudioChunk]] = list(
        decode_websocket_payload(payload=audio_only_media_bytes)
    )
    
    video_frames = [r for r in results if isinstance(r, VideoFrame)]
    audio_chunks = [r for r in results if isinstance(r, AudioChunk)]
    
    assert len(video_frames) == 0
    assert len(audio_chunks) > 0


def test_decode_websocket_payload_expected_behaviour_for_timestamp_offset(
    multiplexed_media_bytes: bytes
) -> None:
    """Verifies that disjointed network chunks correctly carry global temporal sequencing mappings."""
    base_time: float = 120.5
    base_frame: int = 1500
    
    results = list(
        decode_websocket_payload(
            payload=multiplexed_media_bytes,
            target_fps=10.0,
            target_sample_rate=16000,
            base_timestamp_sec=base_time,
            base_frame_idx=base_frame
        )
    )
    
    video_frames = [r for r in results if isinstance(r, VideoFrame)]
    assert video_frames[0].frame_index == base_frame
    assert video_frames[0].timestamp_sec >= base_time


# =====================================================================
# EDGE CASES & VALUE BOUNDARY VIOLATIONS
# =====================================================================

def test_decode_websocket_payload_raises_for_invalid_payload_type() -> None:
    """Tests structural rejection of unmapped interface objects masking as byte arrays."""
    with pytest.raises(TypeError, match="payload must be strictly typed as bytes"):
        list(decode_websocket_payload(payload="corrupted_string", target_fps=50.0))  # type: ignore


def test_decode_websocket_payload_raises_for_empty_payload() -> None:
    """Tests structural rejection of empty network transmissions."""
    with pytest.raises(ValueError, match="payload cannot be an empty byte string"):
        list(decode_websocket_payload(payload=b""))


def test_decode_websocket_payload_raises_for_zero_fps(multiplexed_media_bytes: bytes) -> None:
    """Tests rejection of mathematically impossible visual extraction frequencies (0)."""
    with pytest.raises(ValueError, match="target_fps must be strictly positive"):
        list(decode_websocket_payload(payload=multiplexed_media_bytes, target_fps=0.0))


def test_decode_websocket_payload_raises_for_negative_fps(multiplexed_media_bytes: bytes) -> None:
    """Tests rejection of inverted visual extraction frequencies."""
    with pytest.raises(ValueError, match="target_fps must be strictly positive"):
        list(decode_websocket_payload(payload=multiplexed_media_bytes, target_fps=-10.0))


def test_decode_websocket_payload_raises_for_zero_sample_rate(multiplexed_media_bytes: bytes) -> None:
    """Tests rejection of mathematically impossible acoustic sampling rates (0)."""
    with pytest.raises(ValueError, match="target_sample_rate must be strictly positive"):
        list(decode_websocket_payload(payload=multiplexed_media_bytes, target_sample_rate=0))


def test_decode_websocket_payload_raises_for_negative_sample_rate(multiplexed_media_bytes: bytes) -> None:
    """Tests rejection of inverted acoustic sampling rates."""
    with pytest.raises(ValueError, match="target_sample_rate must be strictly positive"):
        list(decode_websocket_payload(payload=multiplexed_media_bytes, target_sample_rate=-16000))


def test_decode_websocket_payload_raises_for_negative_offset(multiplexed_media_bytes: bytes) -> None:
    """Tests rejection of inverted network continuity timestamps."""
    with pytest.raises(ValueError, match="base_timestamp_sec cannot be negative"):
        list(decode_websocket_payload(payload=multiplexed_media_bytes, base_timestamp_sec=-5.0))


def test_decode_websocket_payload_raises_for_negative_frame_index(multiplexed_media_bytes: bytes) -> None:
    """Tests rejection of inverted continuous global frame indices."""
    with pytest.raises(ValueError, match="base_frame_idx cannot be a negative integer"):
        list(decode_websocket_payload(payload=multiplexed_media_bytes, base_frame_idx=-1))


# =====================================================================
# ERROR CASES: MEMORY & HARDWARE DEGRADATION
# =====================================================================

def test_decode_websocket_payload_raises_for_no_valid_streams(empty_container_bytes: bytes) -> None:
    """Verifies internal PyAV gracefully catches the underlying EOF parsing error on completely empty valid containers."""
    with pytest.raises(RuntimeError, match="Corrupted or unsupported media payload"):
        list(decode_websocket_payload(payload=empty_container_bytes))


def test_decode_websocket_payload_raises_for_corrupted_bytes() -> None:
    """Verifies internal PyAV C-bindings gracefully catch and map junk byte payloads to RuntimeErrors."""
    corrupted_payload = b"this_is_not_a_valid_webm_or_mp4_binary_sequence"
    
    with pytest.raises(RuntimeError, match="Corrupted or unsupported media payload"):
        list(decode_websocket_payload(payload=corrupted_payload))