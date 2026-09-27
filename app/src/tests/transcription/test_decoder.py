from typing import Tuple

import pytest
import torch

from app.src.orchestration.types import TranscriptSegment, WordTimestamp
from app.src.transcription.decoder import (
    _calculate_word_boundaries,
    _extract_greedy_path,
    decode_transducer_output,
)


def test__calculate_word_boundaries_expected_behaviour_for_valid_tensors() -> None:
    """
    Validates accurate calculation of temporal boundaries, including the correct
    handling of blank tokens (index 0) which advance time without emitting words.
    """
    token_indices: torch.Tensor = torch.tensor([1, 0, 2], dtype=torch.long)
    durations: torch.Tensor = torch.tensor([1, 1, 2], dtype=torch.long)
    start_offset: float = 1.0
    time_stride: float = 0.1

    words: Tuple[WordTimestamp, ...] = _calculate_word_boundaries(
        token_indices=token_indices,
        durations=durations,
        start_offset_sec=start_offset,
        time_stride_sec=time_stride
    )

    assert len(words) == 2

    # Token 1: start 1.0, duration 1 * 0.1 -> end 1.1
    assert words[0].word == "token_1"
    assert pytest.approx(words[0].start_time_sec) == 1.0
    assert pytest.approx(words[0].end_time_sec) == 1.1

    # Token 0 (blank): start 1.1, duration 1 * 0.1 -> end 1.2 (skipped, time advances)
    
    # Token 2: start 1.2, duration 2 * 0.1 -> end 1.4
    assert words[1].word == "token_2"
    assert pytest.approx(words[1].start_time_sec) == 1.2
    assert pytest.approx(words[1].end_time_sec) == 1.4


def test__calculate_word_boundaries_raises_for_multidimensional_tensors() -> None:
    """
    Validates dimensional constraints on input tensors.
    """
    token_indices: torch.Tensor = torch.tensor([[1, 2]], dtype=torch.long)
    durations: torch.Tensor = torch.tensor([1, 1], dtype=torch.long)

    with pytest.raises(ValueError) as exc_info:
        _calculate_word_boundaries(
            token_indices=token_indices,
            durations=durations,
            start_offset_sec=0.0
        )

    assert "1-dimensional" in str(exc_info.value)


def test__calculate_word_boundaries_raises_for_length_mismatch() -> None:
    """
    Validates structural alignment between tokens and their durations.
    """
    token_indices: torch.Tensor = torch.tensor([1, 2], dtype=torch.long)
    durations: torch.Tensor = torch.tensor([1, 1, 1], dtype=torch.long)

    with pytest.raises(ValueError) as exc_info:
        _calculate_word_boundaries(
            token_indices=token_indices,
            durations=durations,
            start_offset_sec=0.0
        )

    assert "identical lengths" in str(exc_info.value)


def test__extract_greedy_path_expected_behaviour_for_valid_logits() -> None:
    """
    Validates extraction of a 1D token path and uniform durations from a 3D logit tensor.
    """
    # 3D Tensor: (batch_size=1, time_steps=4, vocab_size=10)
    logits: torch.Tensor = torch.randn(1, 4, 10)
    raw_tensors: Tuple[torch.Tensor, ...] = (logits,)

    token_indices, durations = _extract_greedy_path(raw_tensors)

    assert isinstance(token_indices, torch.Tensor)
    assert isinstance(durations, torch.Tensor)
    assert token_indices.dim() == 1
    assert token_indices.shape[0] == 4
    assert durations.shape == token_indices.shape
    assert torch.all(durations == 1).item() is True


def test__extract_greedy_path_raises_for_empty_tuple() -> None:
    """
    Validates strict handling of empty output sequences from the model.
    """
    with pytest.raises(ValueError) as exc_info:
        _extract_greedy_path(())

    assert "cannot be empty" in str(exc_info.value)


def test__extract_greedy_path_raises_for_invalid_type() -> None:
    """
    Validates type safety against non-tensor outputs.
    """
    with pytest.raises(TypeError) as exc_info:
        _extract_greedy_path(([1, 2, 3],))  # type: ignore

    assert "Expected PyTorch tensors" in str(exc_info.value)


def test__extract_greedy_path_raises_for_invalid_dimensions() -> None:
    """
    Validates dimensional constraints on the raw model logits.
    """
    # 2D Tensor instead of 3D
    logits: torch.Tensor = torch.randn(10, 10)
    
    with pytest.raises(ValueError) as exc_info:
        _extract_greedy_path((logits,))

    assert "at least 3-dimensional" in str(exc_info.value)


def test_decode_transducer_output_expected_behaviour_for_standard_input() -> None:
    """
    Validates end-to-end decoding of raw tensors into the strict dataclass contract.
    """
    logits: torch.Tensor = torch.randn(1, 5, 10)
    raw_tensors: Tuple[torch.Tensor, ...] = (logits,)
    
    segment: TranscriptSegment = decode_transducer_output(
        raw_tensors=raw_tensors,
        segment_id=42,
        start_offset_sec=10.5,
        is_final=True
    )

    assert isinstance(segment, TranscriptSegment)
    assert segment.segment_id == 42
    assert segment.is_final is True
    assert isinstance(segment.words, tuple)


def test_decode_transducer_output_raises_for_negative_segment_id() -> None:
    """
    Validates constraints preventing invalid sequence identifiers.
    """
    logits: torch.Tensor = torch.randn(1, 5, 10)
    
    with pytest.raises(ValueError) as exc_info:
        decode_transducer_output(
            raw_tensors=(logits,),
            segment_id=-1,
            start_offset_sec=0.0,
            is_final=True
        )

    assert "segment_id must be non-negative" in str(exc_info.value)


def test_decode_transducer_output_raises_for_negative_offset() -> None:
    """
    Validates constraints preventing invalid temporal boundaries.
    """
    logits: torch.Tensor = torch.randn(1, 5, 10)
    
    with pytest.raises(ValueError) as exc_info:
        decode_transducer_output(
            raw_tensors=(logits,),
            segment_id=1,
            start_offset_sec=-5.0,
            is_final=True
        )

    assert "start_offset_sec must be non-negative" in str(exc_info.value)


def test_decode_transducer_output_raises_for_invalid_is_final() -> None:
    """
    Validates type constraints on the terminal boundary flag.
    """
    logits: torch.Tensor = torch.randn(1, 5, 10)
    
    with pytest.raises(TypeError) as exc_info:
        decode_transducer_output(
            raw_tensors=(logits,),
            segment_id=1,
            start_offset_sec=0.0,
            is_final="True"  # type: ignore
        )

    assert "is_final must be a boolean" in str(exc_info.value)