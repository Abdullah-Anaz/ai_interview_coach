from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from nemo.collections.asr.models import EncDecCTCModelBPE

from app.src.alignment.aligner import (
    _ensure_local_ctc_weights,
    _refine_word_boundaries,
    _time_to_frame_index,
    initialize_ctc_model,
    process_synchronized_segment,
)
from app.src.alignment.config import AlignmentConfig
from app.src.orchestration.types import (
    AlignedWord,
    AudioChunk,
    SynchronizedSegment,
    TranscriptSegment,
    WordTimestamp,
)


def test_time_to_frame_index_expected_behaviour_for_standard_time() -> None:
    """
    Validates exact translation of seconds into a discrete frame index at 50 FPS.
    """
    frame_index: int = _time_to_frame_index(time_sec=1.5, fps=50)
    assert frame_index == 75


def test_time_to_frame_index_expected_behaviour_for_negative_time() -> None:
    """
    Edge Case: Validates that negative time boundaries are clamped strictly to frame 0.
    """
    frame_index: int = _time_to_frame_index(time_sec=-0.5, fps=50)
    assert frame_index == 0


def test_refine_word_boundaries_expected_behaviour_for_valid_words() -> None:
    """
    Validates the macro mapping of a sequence of transcribed words into frame boundaries.
    """
    config = AlignmentConfig(video_fps=50)
    words: Tuple[WordTimestamp, ...] = (
        WordTimestamp(word="hello", start_time_sec=0.0, end_time_sec=1.0, confidence=0.9),
        WordTimestamp(word="world", start_time_sec=1.0, end_time_sec=2.5, confidence=0.95),
    )
    
    aligned: Tuple[AlignedWord, ...] = _refine_word_boundaries(words, config)
    
    assert len(aligned) == 2
    assert aligned[0].start_frame_idx == 0
    assert aligned[0].end_frame_idx == 50
    assert aligned[1].start_frame_idx == 50
    assert aligned[1].end_frame_idx == 125


def test_refine_word_boundaries_expected_behaviour_for_empty_tuple() -> None:
    """
    Edge Case: Validates the processor gracefully handles empty word sequences (e.g., silence).
    """
    config = AlignmentConfig(video_fps=50)
    aligned: Tuple[AlignedWord, ...] = _refine_word_boundaries(tuple(), config)
    
    assert len(aligned) == 0
    assert isinstance(aligned, tuple)


def test_process_synchronized_segment_expected_behaviour_for_valid_inputs() -> None:
    """
    Validates the end-to-end multi-modal packaging combining audio length and text tokens.
    """
    config = AlignmentConfig(video_fps=50)
    
    # 2 seconds of audio starting at 10.0s
    audio = AudioChunk(
        sample_rate=16000,
        timestamp_sec=10.0,
        data=np.zeros((32000,), dtype=np.float32)
    )
    
    words = (
        WordTimestamp(word="synergy", start_time_sec=10.5, end_time_sec=11.0, confidence=0.99),
    )
    transcript = TranscriptSegment(segment_id=5, words=words, is_final=True)
    
    segment: SynchronizedSegment = process_synchronized_segment(audio, transcript, config)
    
    assert segment.segment_id == 5
    assert segment.is_final is True
    # Audio boundaries: 10.0s to 12.0s
    assert segment.audio_slice_start_sec == 10.0
    assert segment.audio_slice_end_sec == 12.0
    # Visual frames: 10.0s = 500, 12.0s = 600
    assert segment.visual_frame_start_idx == 500
    assert segment.visual_frame_end_idx == 600


def test__ensure_local_ctc_weights_expected_behaviour_for_valid_repo() -> None:
    """
    Validates active connection to Hugging Face to retrieve the specified model.
    """
    config = AlignmentConfig()
    weights_path: Path = _ensure_local_ctc_weights(config)
    
    assert weights_path.exists()
    assert weights_path.suffix == ".nemo"


def test__ensure_local_ctc_weights_raises_for_invalid_repo() -> None:
    """
    Error Case: Validates network/authentication failures trigger a RuntimeError.
    """
    config = AlignmentConfig(
        model_repo_id="nvidia/fake-parakeet-repo-that-does-not-exist",
        model_filename="fake.nemo"
    )
    
    with pytest.raises(RuntimeError) as exc_info:
        _ensure_local_ctc_weights(config)
        
    assert "Failed to download CTC weights" in str(exc_info.value)


def test_initialize_ctc_model_expected_behaviour_for_cpu_hardware() -> None:
    """
    Validates NeMo successfully instantiates the CTC matrix and maps it to target hardware.
    """
    config = AlignmentConfig(device="cpu", use_float16=False)
    model = initialize_ctc_model(config)
    
    assert isinstance(model, EncDecCTCModelBPE)
    # Ensure it is locked in evaluation mode
    assert not model.training


def test_initialize_ctc_model_raises_for_invalid_device() -> None:
    """
    Error Case: Validates hardware mapping failures trigger a RuntimeError.
    """
    config = AlignmentConfig(device="cuda:999")  # Impossible hardware index
    
    with pytest.raises(RuntimeError) as exc_info:
        initialize_ctc_model(config)
        
    assert "CTC model initialization failed" in str(exc_info.value)