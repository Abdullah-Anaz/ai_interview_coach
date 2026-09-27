import logging
from queue import Full, Queue
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoFeatureExtractor, WavLMModel

from app.src.acoustic_edge.config import AcousticConfig
from app.src.acoustic_edge.encoder import (
    encode_audio_to_embeddings,
    load_wavlm_components,
)
from app.src.acoustic_edge.generator import (
    generate_competency_descriptors,
    load_ridge_regressor,
)
from app.src.acoustic_edge.types import AcousticDescriptorPayload
from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel
from app.src.orchestration.types import SynchronizedWindowSlice

logger = logging.getLogger(__name__)


def _process_segment(
    item: SynchronizedWindowSlice,
    wavlm_extractor: AutoFeatureExtractor,
    wavlm_model: WavLMModel,
    ridge_model: Ridge,
    config: AcousticConfig
) -> AcousticDescriptorPayload:
    """
    Transforms a single synchronized audio segment into a text-mediated descriptor payload and raw telemetry.

    Args:
        item (SynchronizedWindowSlice): The input data structure containing the raw audio slice.
        wavlm_extractor (AutoFeatureExtractor): The Hugging Face feature pre-processor.
        wavlm_model (WavLMModel): The loaded WavLM neural network.
        ridge_model (Ridge): The trained scikit-learn Ridge regression model.
        config (AcousticConfig): The active configuration matrix.

    Returns:
        AcousticDescriptorPayload: Text-mediated OCEAN descriptors and numerical scores mapped from the audio chunk.

    Raises:
        RuntimeError: If latent extraction or descriptor generation fails.
    """
    try:
        embeddings: torch.Tensor = encode_audio_to_embeddings(
            audio=item.audio,
            extractor=wavlm_extractor,
            model=wavlm_model,
            config=config
        )

        cues: Tuple[str, ...]
        scores: Dict[str, float]
        cues, scores = generate_competency_descriptors(
            embedding=embeddings,
            ridge_model=ridge_model,
            config=config
        )

        return AcousticDescriptorPayload(
            segment_id=item.window_id,           
            start_time_sec=item.start_time_sec,   
            end_time_sec=item.end_time_sec,       
            vocal_cues=cues,
            **scores
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to process acoustic segment {item.window_id}: {exc}") from exc


def _aggregate_payloads(payloads: List[AcousticDescriptorPayload]) -> AcousticDescriptorPayload:
    """
    Calculates the global average of OCEAN descriptors across the full temporal sequence.

    Args:
        payloads (List[AcousticDescriptorPayload]): Sequence of individual segment payloads.

    Returns:
        AcousticDescriptorPayload: Unified terminal payload containing global statistical averages.

    Raises:
        ValueError: If the payload list is explicitly empty.
    """
    if not payloads:
        raise ValueError("Aggregation requires a strictly populated sequence of payloads.")

    ext_mean: float = float(np.mean([p.extraversion_score for p in payloads]))
    neu_mean: float = float(np.mean([p.neuroticism_score for p in payloads]))
    agr_mean: float = float(np.mean([p.agreeableness_score for p in payloads]))
    con_mean: float = float(np.mean([p.conscientiousness_score for p in payloads]))
    ope_mean: float = float(np.mean([p.openness_score for p in payloads]))

    cues_set: set[str] = set()
    cues_list: List[str] = []
    for p in payloads:
        for cue in p.vocal_cues:
            if cue and cue not in cues_set and cue != "UNAVAILABLE":
                cues_set.add(cue)
                cues_list.append(cue)

    if not cues_list:
        cues_list = ["UNAVAILABLE"]

    return AcousticDescriptorPayload(
        segment_id=0,
        start_time_sec=payloads[0].start_time_sec,
        end_time_sec=payloads[-1].end_time_sec,
        vocal_cues=tuple(cues_list[:5]),
        extraversion_score=ext_mean,
        neuroticism_score=neu_mean,
        agreeableness_score=agr_mean,
        conscientiousness_score=con_mean,
        openness_score=ope_mean
    )


def run_acoustic_node(
    input_queue: Queue,
    output_queue: Queue,
    config: AcousticConfig,
    timeout_sec: float = 5.0
) -> None:
    """
    Executes the continuous Acoustic Edge consumer loop using a Stream-Reduce aggregation pattern.

    Args:
        input_queue (Queue): Ingress queue supplying SynchronizedWindowSlice objects.
        output_queue (Queue): Egress queue receiving the unified AcousticDescriptorPayload.
        config (AcousticConfig): Hardware, model, and threshold configuration matrix.
        timeout_sec (float): Block timeout in seconds for queue retrieval and insertion.

    Raises:
        TypeError: If queue or configuration parameters violate expected types.
        RuntimeError: If model initialization fails or downstream queue saturates.
    """
    if not isinstance(input_queue, Queue):
        raise TypeError("input_queue must be an instance of queue.Queue.")
    if not isinstance(output_queue, Queue):
        raise TypeError("output_queue must be an instance of queue.Queue.")
    if not isinstance(config, AcousticConfig):
        raise TypeError("config must be an instance of AcousticConfig.")

    try:
        wavlm_extractor, wavlm_model = load_wavlm_components(config)
        ridge_model: Ridge = load_ridge_regressor(config)
    except Exception as exc:
        output_queue.put(OrchestratorTerminalSentinel(error=exc))
        raise RuntimeError(f"Acoustic node initialization failed: {exc}") from exc

    history_buffer: List[AcousticDescriptorPayload] = []

    while True:
        try:
            item: Any = input_queue.get(timeout=timeout_sec)
        except Exception:
            continue

        if isinstance(item, OrchestratorTerminalSentinel):
            if history_buffer:
                try:
                    final_payload: AcousticDescriptorPayload = _aggregate_payloads(history_buffer)
                    output_queue.put(final_payload, timeout=timeout_sec)
                    logger.info("[ACOUSTIC SUCCESS] Emitted global aggregated payload to Fusion Nexus.")
                except Full as exc:
                    raise RuntimeError(f"Acoustic aggregation failure: queue saturated. {exc}") from exc
                except Exception as exc:
                    try:
                        output_queue.put(OrchestratorTerminalSentinel(error=exc), timeout=timeout_sec)
                    except Full:
                        pass  # Ignore secondary saturation crash
                    raise RuntimeError(f"Acoustic aggregation failure: {exc}") from exc

            try:
                output_queue.put(item, timeout=timeout_sec)
                history_buffer.clear()
                continue
            except Full as exc:
                raise RuntimeError("Output queue saturated during termination sequence.") from exc

        if not isinstance(item, SynchronizedWindowSlice):
            continue

        try:
            payload: AcousticDescriptorPayload = _process_segment(
                item=item,
                wavlm_extractor=wavlm_extractor,
                wavlm_model=wavlm_model,
                ridge_model=ridge_model,
                config=config
            )
            history_buffer.append(payload)
            logger.info("[ACOUSTIC STREAM] Processed segment %d into aggregation buffer.", payload.segment_id)
        except Exception as exc:
            try:
                output_queue.put(OrchestratorTerminalSentinel(error=exc), timeout=timeout_sec)
            except Full:
                pass
            logger.error("[ACOUSTIC ERROR] Processing failed: %s", exc)
            raise RuntimeError(f"Acoustic processing failure: {exc}") from exc