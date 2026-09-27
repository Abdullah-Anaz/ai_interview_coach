import logging
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue, Empty
from typing import Iterator, List, Optional, Tuple, Union

from app.src.orchestration.config import OrchestratorConfig
from app.src.orchestration.demuxer import demux_container
from app.src.orchestration.frame_buffer import FrameBuffer
from app.src.orchestration.types import (
    AudioChunk,
    SynchronizedWindowSlice,
    VideoFrame,
)
from app.src.orchestration.window_manager import _extract_audio_slice

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorTerminalSentinel:
    """
    Tombstone object injected into output queues to signal stream completion
    or pipeline termination to downstream consumer nodes.
    """
    error: Optional[Exception] = None


def route_media_stream(
    source: Union[str, Path],
    config: OrchestratorConfig,
    transcription_queue: Queue,
    window_queue: Queue,
    frame_buffer_capacity: int = 500
) -> None:
    """
    Decouples and routes incoming demuxed media streams downstream.

    Pushes full-turn audio packets to `transcription_queue` for ASR and
    routes visual frames into a bounded `FrameBuffer`. Concurrently governs
    the emission of `SynchronizedWindowSlice` objects to `window_queue`.

    Args:
        source (Union[str, Path]): File path to the media container.
        config (OrchestratorConfig): Ingestion and temporal windowing parameters.
        transcription_queue (Queue): Queue receiving raw AudioChunks for Parakeet ASR.
        window_queue (Queue): Queue receiving SynchronizedWindowSlice objects.
        frame_buffer_capacity (int): Maximum frames retained in the visual ring buffer.

    Raises:
        TypeError: If queue or configuration constraints are violated.
        RuntimeError: If an unrecoverable decoding or queue overflow error occurs.
    """
    if not isinstance(transcription_queue, Queue):
        raise TypeError(f"transcription_queue must be a queue.Queue, got {type(transcription_queue).__name__}.")
    if not isinstance(window_queue, Queue):
        raise TypeError(f"window_queue must be a queue.Queue, got {type(window_queue).__name__}.")
    if not isinstance(config, OrchestratorConfig):
        raise TypeError(f"config must be an OrchestratorConfig, got {type(config).__name__}.")

    frame_buffer: FrameBuffer = FrameBuffer(max_capacity=frame_buffer_capacity)
    acoustic_buffer: List[AudioChunk] = []

    window_id: int = 0
    window_start_sec: float = 0.0
    max_audio_time_sec: float = 0.0

    try:
        raw_stream: Iterator[Union[VideoFrame, AudioChunk]] = demux_container(
            source=source,
            target_fps=config.video_fps,
            target_sample_rate=config.audio_sample_rate
        )

        for packet in raw_stream:
            if isinstance(packet, AudioChunk):
                # Route continuous audio directly to Transcription queue
                try:
                    transcription_queue.put(packet, timeout=config.queue_timeout_sec)
                except Full as e:
                    raise RuntimeError("Transcription queue saturated.") from e

                # Accumulate in local acoustic buffer for sliding window generation
                acoustic_buffer.append(packet)
                duration: float = len(packet.data) / packet.sample_rate
                max_audio_time_sec = packet.timestamp_sec + duration

            elif isinstance(packet, VideoFrame):
                # Route visual frames directly to the FrameBuffer node
                frame_buffer.append(packet)

            # Slide temporal window across FrameBuffer and acoustic buffer
            while max_audio_time_sec >= window_start_sec + config.window_duration_sec:
                window_end_sec: float = window_start_sec + config.window_duration_sec

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
                    window_queue.put(window_packet, timeout=config.queue_timeout_sec)
                except Full as e:
                    raise RuntimeError("Downstream window queue saturated.") from e

                window_start_sec += config.window_stride_sec
                window_id += 1

                acoustic_buffer = [
                    c for c in acoustic_buffer
                    if (c.timestamp_sec + (len(c.data) / c.sample_rate)) > window_start_sec
                ]

        if acoustic_buffer and max_audio_time_sec > window_start_sec:
            audio_terminal: AudioChunk = _extract_audio_slice(
                chunks=acoustic_buffer,
                start_sec=window_start_sec,
                end_sec=max_audio_time_sec,
                sample_rate=config.audio_sample_rate
            )
            video_terminal: Tuple[VideoFrame, ...] = frame_buffer.get_frames_in_interval(
                start_sec=window_start_sec,
                end_sec=max_audio_time_sec
            )

            terminal_window = SynchronizedWindowSlice(
                window_id=window_id,
                start_time_sec=window_start_sec,
                end_time_sec=max_audio_time_sec,
                audio=audio_terminal,
                frames=video_terminal,
                is_terminal=True
            )
            try:
                window_queue.put(terminal_window, timeout=config.queue_timeout_sec)
            except Full as e:
                raise RuntimeError("Downstream window queue saturated on terminal slice.") from e

    except Exception as e:
        logger.error("Orchestration routing failure: %s", e)
        try:
            transcription_queue.put(OrchestratorTerminalSentinel(error=e), timeout=config.queue_timeout_sec)
        except Full:
            logger.warning("Transcription queue full; dropped error sentinel.")
        try:
            window_queue.put(OrchestratorTerminalSentinel(error=e), timeout=config.queue_timeout_sec)
        except Full:
            logger.warning("Window queue full; dropped error sentinel.")
            
        raise RuntimeError(f"Orchestration failure: {e}") from e

    else:
        try:
            transcription_queue.put(OrchestratorTerminalSentinel(error=None), timeout=config.queue_timeout_sec)
        except Full:
            logger.warning("Transcription queue full; dropped success sentinel.")
        try:
            window_queue.put(OrchestratorTerminalSentinel(error=None), timeout=config.queue_timeout_sec)
        except Full:
            logger.warning("Window queue full; dropped success sentinel.")


def run_orchestrator_node(
    q_transcription_in: Queue,
    q_window_in: Queue,
    q_acoustic_out: Queue,
    q_semantic_out: Queue,
    q_visual_facial_out: Queue,
    q_visual_skeletal_out: Queue,
    timeout_sec: float = 0.1
) -> None:
    """
    Continuous background daemon that multiplexes ingested media streams to the specialized edge queues.

    Polls the raw transcription and synchronized window ingress buffers, safely duplicating 
    and routing references to the independent downstream compute nodes.

    Args:
        q_transcription_in (Queue): Ingress buffer containing raw AudioChunks.
        q_window_in (Queue): Ingress buffer containing SynchronizedWindowSlice objects.
        q_acoustic_out (Queue): Egress buffer routing windows to the acoustic feature extractor.
        q_semantic_out (Queue): Egress buffer routing windows to the ASR/Semantic node.
        q_visual_facial_out (Queue): Egress buffer routing windows to facial kinematics.
        q_visual_skeletal_out (Queue): Egress buffer routing windows to skeletal kinematics.
        timeout_sec (float): Block duration for queue polling to prevent thread starvation.

    Raises:
        TypeError: If any buffer violates the queue.Queue type constraint.
    """
    queues_list: List[Queue] = [
        q_transcription_in, q_window_in, q_acoustic_out, q_semantic_out,
        q_visual_facial_out, q_visual_skeletal_out
    ]
    for q in queues_list:
        if not isinstance(q, Queue):
            raise TypeError(f"All parameters must be queue.Queue, got {type(q).__name__}.")

    active_transcription: bool = True
    active_window: bool = True

    while active_transcription or active_window:
        if active_transcription:
            try:
                # Silently drain continuous audio buffer to prevent saturation
                audio_packet = q_transcription_in.get(timeout=timeout_sec)
                if isinstance(audio_packet, OrchestratorTerminalSentinel):
                    active_transcription = False
            except Empty:
                pass

        if active_window:
            try:
                window_packet = q_window_in.get(timeout=timeout_sec)
                if isinstance(window_packet, OrchestratorTerminalSentinel):
                    active_window = False
                    q_acoustic_out.put(window_packet)
                    q_semantic_out.put(window_packet)  # Routes Window to ASR Edge
                    q_visual_facial_out.put(window_packet)
                    q_visual_skeletal_out.put(window_packet)
                else:
                    q_acoustic_out.put(window_packet)
                    q_semantic_out.put(window_packet)  # Routes Window to ASR Edge
                    q_visual_facial_out.put(window_packet)
                    q_visual_skeletal_out.put(window_packet)
            except Empty:
                pass
            except Exception as exc:
                logger.error("Orchestrator window multiplexing fault: %s", exc)
                q_acoustic_out.put(OrchestratorTerminalSentinel(error=exc))
                q_semantic_out.put(OrchestratorTerminalSentinel(error=exc))
                q_visual_facial_out.put(OrchestratorTerminalSentinel(error=exc))
                q_visual_skeletal_out.put(OrchestratorTerminalSentinel(error=exc))
                active_window = False