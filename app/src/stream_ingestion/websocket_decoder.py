import io
import logging
from typing import Iterator, Union

import av
import numpy as np

from app.src.orchestration.types import AudioChunk, VideoFrame

logger = logging.getLogger(__name__)


def decode_websocket_payload(
    payload: bytes,
    target_fps: float = 50.0,
    target_sample_rate: int = 16000,
    base_timestamp_sec: float = 0.0,
    base_frame_idx: int = 0
) -> Iterator[Union[VideoFrame, AudioChunk]]:
    """
    Decodes an isolated raw binary media payload into normalized temporal data transfer objects.

    Operates entirely in-memory to prevent disk I/O latency. Applies temporal offsets to 
    maintain continuous time synchronization across fragmented network packets.

    Args:
        payload (bytes): The raw binary media chunk received via the WebSocket connection.
        target_fps (float): Normalized temporal extraction frequency for visual frames.
        target_sample_rate (int): Normalized sampling rate for the acoustic stream.
        base_timestamp_sec (float): Continuous elapsed time offset to append to this specific payload.
        base_frame_idx (int): The starting index mapping to maintain global visual sequence continuity.

    Yields:
        Union[VideoFrame, AudioChunk]: Strictly typed and dimensionally bounded multimodal slices.

    Raises:
        TypeError: If the payload is not a valid byte string.
        ValueError: If constraints like positive FPS, sampling rates, or empty payloads are violated.
        RuntimeError: If decoding fails or the byte payload is structurally corrupted.
    """
    if not isinstance(payload, bytes):
        raise TypeError(f"payload must be strictly typed as bytes, got {type(payload).__name__}.")
    if not payload:
        raise ValueError("payload cannot be an empty byte string.")
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be strictly positive, got {target_fps}.")
    if target_sample_rate <= 0:
        raise ValueError(f"target_sample_rate must be strictly positive, got {target_sample_rate}.")
    if base_timestamp_sec < 0.0:
        raise ValueError("base_timestamp_sec cannot be negative.")
    if base_frame_idx < 0:
        raise ValueError("base_frame_idx cannot be a negative integer.")

    target_interval_sec: float = 1.0 / target_fps
    next_target_time_sec: float = base_timestamp_sec
    current_frame_index: int = base_frame_idx

    try:
        buffer: io.BytesIO = io.BytesIO(payload)
        with av.open(buffer, format="webm") as container:
            audio_stream = container.streams.audio[0] if container.streams.audio else None
            video_stream = container.streams.video[0] if container.streams.video else None

            streams = [s for s in [audio_stream, video_stream] if s is not None]
            if not streams:
                raise RuntimeError("No valid audio or video streams detected in payload.")

            if video_stream:
                video_stream.thread_type = "AUTO"

            resampler = None
            if audio_stream:
                resampler = av.AudioResampler(
                    format="flt",
                    layout="mono",
                    rate=target_sample_rate
                )

            for packet in container.demux(streams):
                for frame in packet.decode():
                    time_sec: float = float(frame.time) if frame.time is not None else 0.0
                    absolute_time_sec: float = base_timestamp_sec + time_sec

                    if packet.stream.type == "video":
                        rgb_data: np.ndarray = frame.to_ndarray(format="rgb24")
                        
                        while next_target_time_sec <= absolute_time_sec:
                            yield VideoFrame(
                                frame_index=current_frame_index,
                                timestamp_sec=next_target_time_sec,
                                data=rgb_data
                            )
                            next_target_time_sec += target_interval_sec
                            current_frame_index += 1

                    elif packet.stream.type == "audio" and resampler is not None:
                        for resampled_frame in resampler.resample(frame):
                            res_time_sec: float = (
                                float(resampled_frame.time) if resampled_frame.time is not None else 0.0
                            )
                            yield AudioChunk(
                                sample_rate=target_sample_rate,
                                timestamp_sec=base_timestamp_sec + res_time_sec,
                                data=resampled_frame.to_ndarray().flatten().astype(np.float32)
                            )

            if resampler is not None:
                for resampled_frame in resampler.resample(None):
                    res_time_sec: float = (
                        float(resampled_frame.time) if resampled_frame.time is not None else 0.0
                    )
                    yield AudioChunk(
                        sample_rate=target_sample_rate,
                        timestamp_sec=base_timestamp_sec + res_time_sec,
                        data=resampled_frame.to_ndarray().flatten().astype(np.float32)
                    )

    except Exception as exc:
        if isinstance(exc, RuntimeError) and "No valid audio or video streams" in str(exc):
            raise
        logger.error("Media payload decoding failure: %s", exc)
        raise RuntimeError(f"Corrupted or unsupported media payload: {exc}") from exc