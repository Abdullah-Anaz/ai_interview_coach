import time
from queue import Full, Queue
from typing import Iterator, List, Tuple, Union

import cv2
import numpy as np
import sounddevice as sd

from app.src.orchestration.frame_buffer import FrameBuffer
from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.orchestration.types import (
    AudioChunk,
    SynchronizedWindowSlice,
    VideoFrame,
)
from app.src.orchestration.window_manager import _extract_audio_slice
from app.src.stream_ingestion.config import IngestionConfig, IngestionMode


def _hardware_stream_generator(
    config: IngestionConfig,
    max_duration_sec: float = 3600.0
) -> Iterator[Union[VideoFrame, AudioChunk]]:
    """
    Safely binds to OS-level hardware drivers to yield an interleaved stream of audio and visual data.

    Args:
        config (IngestionConfig): Validated configuration matrix bound to LIVE mode.
        max_duration_sec (float): Absolute maximum runtime before forcing a graceful hardware release.

    Yields:
        Union[VideoFrame, AudioChunk]: Structurally normalized frames and audio arrays.

    Raises:
        RuntimeError: If the designated webcam or microphone fails initialization or disconnects.
    """
    cap = cv2.VideoCapture(config.camera_device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam hardware unavailable at index {config.camera_device_index}.")

    try:
        audio_stream = sd.InputStream(
            device=config.audio_device_index,
            samplerate=config.audio_sample_rate,
            channels=1,
            dtype=np.float32
        )
        audio_stream.start()
    except Exception as exc:
        cap.release()
        raise RuntimeError(f"Microphone hardware unavailable at index {config.audio_device_index}: {exc}") from exc

    start_time: float = time.time()
    frame_idx: int = 0
    frame_interval: float = 1.0 / config.video_fps

    try:
        while (time.time() - start_time) < max_duration_sec:
            current_time: float = time.time() - start_time

            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield VideoFrame(
                frame_index=frame_idx,
                timestamp_sec=current_time,
                data=rgb_frame
            )
            frame_idx += 1

            available_frames: int = audio_stream.read_available
            if available_frames > 0:
                audio_data, _ = audio_stream.read(available_frames)
                yield AudioChunk(
                    sample_rate=config.audio_sample_rate,
                    timestamp_sec=current_time,
                    data=audio_data.flatten()
                )

            elapsed: float = time.time() - start_time
            expected_time: float = frame_idx * frame_interval
            if expected_time > elapsed:
                time.sleep(expected_time - elapsed)

    finally:
        audio_stream.stop()
        audio_stream.close()
        cap.release()


def execute_live_ingestion(
    config: IngestionConfig,
    transcription_queue: Queue,
    window_queue: Queue,
    frame_buffer_capacity: int = 500,
    window_duration_sec: float = 2.0,
    window_stride_sec: float = 0.5,
    queue_timeout_sec: float = 5.0
) -> None:
    """
    Consumes live hardware data and executes real-time sliding window synchronization.

    Args:
        config (IngestionConfig): Validated configuration bound to LIVE mode.
        transcription_queue (Queue): Egress route for normalized audio chunks.
        window_queue (Queue): Egress route for synchronized multimodal temporal slices.
        frame_buffer_capacity (int): Maximum frames retained in the visual ring buffer.
        window_duration_sec (float): Total length of each temporal window slice.
        window_stride_sec (float): Temporal step size between consecutive windows.
        queue_timeout_sec (float): Maximum block time when pushing to a full downstream queue.

    Raises:
        ValueError: If the configuration mode is strictly SIMULATOR.
        RuntimeError: If hardware drops or downstream routing blocks excessively.
    """
    if config.mode != IngestionMode.LIVE:
        raise ValueError("execute_live_ingestion strictly requires IngestionMode.LIVE.")

    if not isinstance(transcription_queue, Queue):
        raise TypeError("transcription_queue must be a queue.Queue.")
    if not isinstance(window_queue, Queue):
        raise TypeError("window_queue must be a queue.Queue.")

    frame_buffer: FrameBuffer = FrameBuffer(max_capacity=frame_buffer_capacity)
    acoustic_buffer: List[AudioChunk] = []

    window_id: int = 0
    window_start_sec: float = 0.0
    max_audio_time_sec: float = 0.0

    try:
        hardware_stream: Iterator[Union[VideoFrame, AudioChunk]] = _hardware_stream_generator(config)

        for packet in hardware_stream:
            if isinstance(packet, AudioChunk):
                try:
                    transcription_queue.put(packet, timeout=queue_timeout_sec)
                except Full as exc:
                    raise RuntimeError("Transcription queue saturated.") from exc

                acoustic_buffer.append(packet)
                duration: float = len(packet.data) / packet.sample_rate
                max_audio_time_sec = packet.timestamp_sec + duration

            elif isinstance(packet, VideoFrame):
                frame_buffer.append(packet)

            while max_audio_time_sec >= window_start_sec + window_duration_sec:
                window_end_sec: float = window_start_sec + window_duration_sec

                audio_slice: AudioChunk = _extract_audio_slice(
                    chunks=acoustic_buffer,
                    start_sec=window_start_sec,
                    end_sec=window_end_sec,
                    sample_rate=config.audio_sample_rate
                )
                video_slice: Tuple[VideoFrame, ...] = frame_buffer.get_frames_in_interval(
                    start_sec=window_start_sec,
                    end_sec=window_end_sec
                )

                window_packet = SynchronizedWindowSlice(
                    window_id=window_id,
                    start_time_sec=window_start_sec,
                    end_time_sec=window_end_sec,
                    audio=audio_slice,
                    frames=video_slice,
                    is_terminal=False
                )

                try:
                    window_queue.put(window_packet, timeout=queue_timeout_sec)
                except Full as exc:
                    raise RuntimeError("Downstream window queue saturated.") from exc

                window_start_sec += window_stride_sec
                window_id += 1

                acoustic_buffer = [
                    c for c in acoustic_buffer
                    if (c.timestamp_sec + (len(c.data) / c.sample_rate)) > window_start_sec
                ]

    except Exception as exc:
        try:
            transcription_queue.put(OrchestratorTerminalSentinel(error=exc), timeout=queue_timeout_sec)
            window_queue.put(OrchestratorTerminalSentinel(error=exc), timeout=queue_timeout_sec)
        except Full:
            pass
        raise RuntimeError(f"Live ingestion failure: {exc}") from exc

    else:
        try:
            transcription_queue.put(OrchestratorTerminalSentinel(error=None), timeout=queue_timeout_sec)
            window_queue.put(OrchestratorTerminalSentinel(error=None), timeout=queue_timeout_sec)
        except Full:
            pass