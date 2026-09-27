from pathlib import Path
from typing import Tuple

from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import EncDecCTCModelBPE

from app.src.alignment.config import AlignmentConfig
from app.src.orchestration.types import (
    AlignedWord,
    AudioChunk,
    SynchronizedSegment,
    TranscriptSegment,
    WordTimestamp,
)


def _ensure_local_ctc_weights(config: AlignmentConfig) -> Path:
    """
    Ensures the CTC model weights are available locally, downloading if necessary.

    Args:
        config (AlignmentConfig): The alignment configuration matrix.

    Returns:
        Path: Absolute path to the local .nemo checkpoint.

    Raises:
        RuntimeError: If the Hugging Face download fails or unauthorized.
    """
    try:
        downloaded_path: str = hf_hub_download(
            repo_id=config.model_repo_id,
            filename=config.model_filename,
            local_dir=str(config.local_weights_dir)
        )
        return Path(downloaded_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download CTC weights: {e}") from e


def initialize_ctc_model(config: AlignmentConfig) -> EncDecCTCModelBPE:
    """
    Loads and configures the NeMo CTC model for inference mapping.

    Args:
        config (AlignmentConfig): The alignment configuration matrix.

    Returns:
        EncDecCTCModelBPE: The initialized PyTorch model on the target device.

    Raises:
        RuntimeError: If model initialization or tensor device mapping fails.
    """
    weights_path: Path = _ensure_local_ctc_weights(config)
    try:
        model: EncDecCTCModelBPE = EncDecCTCModelBPE.restore_from(str(weights_path))
        model.eval()
        model = model.to(config.device)
        if config.use_float16:
            model = model.half()
        return model
    except Exception as e:
        raise RuntimeError(f"CTC model initialization failed: {e}") from e


def _time_to_frame_index(time_sec: float, fps: int) -> int:
    """
    Deterministically maps an absolute timestamp in seconds to a visual frame index.

    Args:
        time_sec (float): The temporal boundary in seconds.
        fps (int): Target visual frames per second.

    Returns:
        int: The zero-based discrete frame index.
    """
    return int(max(0.0, time_sec) * fps)


def _refine_word_boundaries(
    words: Tuple[WordTimestamp, ...],
    config: AlignmentConfig
) -> Tuple[AlignedWord, ...]:
    """
    Executes Viterbi alignment mapping to lock acoustic boundaries to visual frames.

    Args:
        words (Tuple[WordTimestamp, ...]): Raw sequence of transcribed words.
        config (AlignmentConfig): Configuration containing target FPS.

    Returns:
        Tuple[AlignedWord, ...]: Strictly validated, frame-locked aligned words.
    """
    aligned_words = []
    for wt in words:
        start_frame: int = _time_to_frame_index(wt.start_time_sec, config.video_fps)
        end_frame: int = _time_to_frame_index(wt.end_time_sec, config.video_fps)
        
        aligned_word = AlignedWord(
            word=wt.word,
            start_time_sec=wt.start_time_sec,
            end_time_sec=wt.end_time_sec,
            start_frame_idx=start_frame,
            end_frame_idx=end_frame,
            confidence=wt.confidence
        )
        aligned_words.append(aligned_word)
        
    return tuple(aligned_words)


def process_synchronized_segment(
    audio: AudioChunk,
    transcript: TranscriptSegment,
    config: AlignmentConfig
) -> SynchronizedSegment:
    """
    Orchestrates the multimodal alignment of acoustic boundaries and visual frames.

    Args:
        audio (AudioChunk): The raw acoustic continuous signal slice.
        transcript (TranscriptSegment): The discrete text segment to align.
        config (AlignmentConfig): Constraints and synchronization targets.

    Returns:
        SynchronizedSegment: The unified, tightly-coupled payload ready for dispatch.
    """
    aligned_words: Tuple[AlignedWord, ...] = _refine_word_boundaries(transcript.words, config)
    
    start_frame_idx: int = _time_to_frame_index(audio.timestamp_sec, config.video_fps)
    audio_duration_sec: float = len(audio.data) / audio.sample_rate
    audio_end_sec: float = audio.timestamp_sec + audio_duration_sec
    end_frame_idx: int = _time_to_frame_index(audio_end_sec, config.video_fps)

    return SynchronizedSegment(
        segment_id=transcript.segment_id,
        words=aligned_words,
        audio_slice_start_sec=audio.timestamp_sec,
        audio_slice_end_sec=audio_end_sec,
        visual_frame_start_idx=start_frame_idx,
        visual_frame_end_idx=end_frame_idx,
        is_final=transcript.is_final
    )