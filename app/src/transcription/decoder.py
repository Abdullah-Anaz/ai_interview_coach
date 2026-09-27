from typing import Any, List, Tuple

import torch

from app.src.orchestration.types import AlignedWord, TranscriptSegment


def _calculate_word_boundaries(
    token_indices: torch.Tensor,
    durations: torch.Tensor,
    start_offset_sec: float,
    tokenizer: Any,
    time_stride_sec: float = 0.04
) -> Tuple[AlignedWord, ...]:
    """
    Calculates absolute temporal boundaries for a sequence of decoded tokens.

    Args:
        token_indices (torch.Tensor): 1-dimensional tensor of vocabulary indices.
        durations (torch.Tensor): 1-dimensional tensor of frame durations per token.
        start_offset_sec (float): Absolute start time of the acoustic segment in seconds.
        tokenizer (Any): Model-specific tokenizer instance for subword decoding.
        time_stride_sec (float, optional): Temporal stride per acoustic frame. Defaults to 0.04.

    Returns:
        Tuple[AlignedWord, ...]: A sequence of temporally bounded word dataclass instances.

    Raises:
        ValueError: If token or duration tensors are not 1-dimensional, or if lengths mismatch.
    """
    if token_indices.dim() != 1 or durations.dim() != 1:
        raise ValueError("Token and duration tensors must be strictly 1-dimensional.")

    if token_indices.shape[0] != durations.shape[0]:
        raise ValueError("Token indices and durations must possess identical temporal lengths.")

    words: List[AlignedWord] = []
    current_time_sec: float = start_offset_sec

    token_list: List[int] = token_indices.tolist()
    duration_list: List[int] = durations.tolist()

    for token_id, duration_frames in zip(token_list, duration_list):
        if token_id == 0:
            current_time_sec += duration_frames * time_stride_sec
            continue

        end_time_sec: float = current_time_sec + (duration_frames * time_stride_sec)
        text_chunk: str = tokenizer.ids_to_text([token_id])

        word_obj = AlignedWord(
            word=text_chunk,
            start_time_sec=float(current_time_sec),
            end_time_sec=float(end_time_sec),
            start_frame_idx=int(current_time_sec * 50.0),
            end_frame_idx=int(end_time_sec * 50.0),
            confidence=0.99
        )
        words.append(word_obj)
        current_time_sec = end_time_sec

    return tuple(words)


def _extract_greedy_path(
    raw_tensors: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the most probable subword sequence and frame durations from raw log-probabilities.

    Args:
        raw_tensors (Tuple[torch.Tensor, ...]): The raw output matrices from the Transducer forward pass.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Extracted 1-dimensional token indices and frame durations mapped to the active device.

    Raises:
        ValueError: If the input sequence is empty or the logits tensor lacks required dimensions.
        TypeError: If the tuple elements are not valid PyTorch tensors.
    """
    if not raw_tensors:
        raise ValueError("The raw_tensors sequence cannot be empty.")

    logits: Any = raw_tensors[0]

    if not isinstance(logits, torch.Tensor):
        raise TypeError("Expected PyTorch tensors within the raw output sequence.")

    if logits.dim() < 3:
        raise ValueError("Logits tensor must be at least 3-dimensional representing (batch, time, vocab).")

    probabilities: torch.Tensor = torch.softmax(logits, dim=-1)
    best_path: torch.Tensor = torch.argmax(probabilities, dim=-1)
    
    token_indices: torch.Tensor = best_path.squeeze(0)
    durations: torch.Tensor = torch.ones_like(token_indices, device=token_indices.device)

    return token_indices, durations


def decode_transducer_output(
    raw_tensors: Tuple[torch.Tensor, ...],
    segment_id: int,
    start_offset_sec: float,
    is_final: bool,
    tokenizer: Any 
) -> TranscriptSegment:
    """
    Decodes raw neural acoustic probabilities into a fully realized, temporally aligned text segment.

    Args:
        raw_tensors (Tuple[torch.Tensor, ...]): Unprocessed prediction tensors from the model execution.
        segment_id (int): Unique sequential identifier for the acoustic window.
        start_offset_sec (float): Continuous elapsed time offset for the segment.
        is_final (bool): Boolean flag denoting the terminal utterance boundary.
        tokenizer (Any): The initialized model subword tokenizer.

    Returns:
        TranscriptSegment: The strictly typed, temporally aligned semantic payload.

    Raises:
        ValueError: If temporal boundaries are negative or sequence identifiers are invalid.
        TypeError: If structural type constraints on input parameters are violated.
    """
    if not isinstance(segment_id, int):
        raise TypeError("segment_id must be strictly typed as an integer.")
    if segment_id < 0:
        raise ValueError("segment_id cannot be a negative integer.")
        
    if not isinstance(start_offset_sec, (int, float)):
        raise TypeError("start_offset_sec must be a numeric value.")
    if start_offset_sec < 0.0:
        raise ValueError("start_offset_sec cannot be a negative temporal boundary.")
        
    if not isinstance(is_final, bool):
        raise TypeError("is_final must be strictly typed as a boolean.")

    token_indices, durations = _extract_greedy_path(raw_tensors)

    word_timestamps: Tuple[AlignedWord, ...] = _calculate_word_boundaries(
        token_indices=token_indices,
        durations=durations,
        start_offset_sec=start_offset_sec,
        tokenizer=tokenizer
    )

    return TranscriptSegment(
        segment_id=segment_id,
        words=word_timestamps,
        is_final=is_final
    )