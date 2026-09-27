import logging
from pathlib import Path
from typing import BinaryIO, Iterator, List, Optional, Union

import av
import numpy as np

from app.src.orchestration.types import AudioChunk, VideoFrame

logger = logging.getLogger(__name__)


def demux_container(
    source: Union[str, Path, BinaryIO],
    target_fps: float = 50.0,
    target_sample_rate: int = 16000
) -> Iterator[Union[VideoFrame, AudioChunk]]:
    """
    Extracts and normalizes audio and video streams from a media source.

    Operates as a generator, yielding interleaved audio and video packets 
    to prevent memory saturation when processing large video files.

    Args:
        source (Union[str, Path, BinaryIO]): File path or binary stream of the media container.
        target_fps (float): Normalized temporal extraction frequency for video frames.
        target_sample_rate (int): Normalized sampling rate for the audio stream.

    Yields:
        Union[VideoFrame, AudioChunk]: Strongly-typed media slices normalized to pipeline constraints.

    Raises:
        RuntimeError: If decoding fails or no compatible streams are detected.
    """
    target_interval_sec: float = 1.0 / target_fps
    next_target_time_sec: float = 0.0
    frame_index: int = 0
    
    source_target = str(source) if isinstance(source, Path) else source

    try:
        with av.open(source_target) as container:
            audio_stream: Optional[av.AudioStream] = (
                container.streams.audio[0] if container.streams.audio else None
            )
            video_stream: Optional[av.VideoStream] = (
                container.streams.video[0] if container.streams.video else None
            )
            
            streams: List[av.Stream] = [s for s in [audio_stream, video_stream] if s is not None]
            
            if not streams:
                raise RuntimeError("No valid audio or video streams detected in the container.")

            if video_stream:
                video_stream.thread_type = "AUTO"

            resampler: Optional[av.AudioResampler] = None
            if audio_stream:
                resampler = av.AudioResampler(
                    format="flt",
                    layout="mono",
                    rate=target_sample_rate
                )

            for packet in container.demux(streams):
                for frame in packet.decode():
                    time_sec: float = float(frame.time) if frame.time is not None else 0.0

                    if packet.stream.type == "video":
                        rgb_data: np.ndarray = frame.to_ndarray(format="rgb24")
                        
                        while next_target_time_sec <= time_sec:
                            yield VideoFrame(
                                frame_index=frame_index,
                                timestamp_sec=next_target_time_sec,
                                data=rgb_data
                            )
                            next_target_time_sec += target_interval_sec
                            frame_index += 1

                    elif packet.stream.type == "audio" and resampler is not None:
                        for resampled_frame in resampler.resample(frame):
                            res_time_sec: float = (
                                float(resampled_frame.time) if resampled_frame.time is not None else 0.0
                            )
                            audio_array: np.ndarray = (
                                resampled_frame.to_ndarray().flatten().astype(np.float32)
                            )
                            yield AudioChunk(
                                sample_rate=target_sample_rate,
                                timestamp_sec=res_time_sec,
                                data=audio_array
                            )

            if resampler is not None:
                for resampled_frame in resampler.resample(None):
                    res_time_sec: float = (
                        float(resampled_frame.time) if resampled_frame.time is not None else 0.0
                    )
                    audio_array: np.ndarray = (
                        resampled_frame.to_ndarray().flatten().astype(np.float32)
                    )
                    yield AudioChunk(
                        sample_rate=target_sample_rate,
                        timestamp_sec=res_time_sec,
                        data=audio_array
                    )

    except Exception as e:
        if type(e).__module__ == "av.error" or isinstance(e, OSError):
            logger.error("FFmpeg demuxing error encountered: %s", e)
            raise RuntimeError(f"FFmpeg pipeline failure: {e}") from e
            
        logger.error("Unexpected error during demuxing: %s", e)
        raise RuntimeError(f"Demuxer failed: {e}") from e