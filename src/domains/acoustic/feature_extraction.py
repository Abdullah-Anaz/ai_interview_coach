import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


def resolve_device(config_device: str, global_device: torch.device) -> torch.device:
    """
    Determines the appropriate execution hardware device based on configuration and availability.

    Args:
        config_device (str): Device identifier string from configurations.
        global_device (torch.device): Fallback device identifier.

    Returns:
        torch.device: The resolved PyTorch hardware device object.
    """
    try:
        config_lower: str = config_device.lower()
        if config_lower in ["cuda", "cpu"]:
            resolved: torch.device = torch.device(config_lower)
            if config_lower == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA unavailable. Falling back to CPU.")
                return torch.device("cpu")
            return resolved
        return global_device
    except Exception as e:
        logger.error("Failed to resolve target device: %s", e)
        return global_device


def initialize_acoustic_model(architecture: str, model_id: str, device: torch.device) -> Tuple[Any, Any, str]:
    """
    Instantiates the target acoustic edge model (WavLM or Emotion2Vec+) for continuous latent extraction.

    Args:
        architecture (str): The architecture identifier ('wavlm' or 'emotion2vec+').
        model_id (str): The specific model weight identifier.
        device (torch.device): Compute device for the model.

    Returns:
        Tuple[Any, Any, str]: The model, the feature extractor (if applicable), and the framework string.
    """
    arch_lower: str = architecture.lower()
    try:
        if "wavlm" in arch_lower:
            from transformers import AutoFeatureExtractor, AutoModel
            
            try:
                extractor = AutoFeatureExtractor.from_pretrained(model_id)
            except Exception:
                logger.warning("Preprocessor config missing for %s. Using microsoft/wavlm-base-plus extractor.", model_id)
                extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")

            if model_id == "microsoft/wavlm-large":
                model = AutoModel.from_pretrained(model_id, revision="refs/pr/7").to(device)
            else:
                model = AutoModel.from_pretrained(model_id).to(device)
                
            model.eval()
            return model, extractor, "huggingface"
            
        elif "emotion2vec" in arch_lower:
            from funasr import AutoModel
            
            model = AutoModel(model=model_id, device=str(device), disable_update=True)
            return model, None, "funasr"
            
        else:
            raise ValueError(f"Prohibited or unsupported architecture requested: {architecture}")
            
    except Exception as e:
        logger.error("Model initialization failed: %s", e)
        raise


def _load_and_resample(path_str: str, target_sr: int) -> torch.Tensor:
    """
    Loads an audio file, resolves paths, extracts a mono channel, and resamples.

    Args:
        path_str (str): The primary file path to the audio file.
        target_sr (int): The target sampling rate for the output waveform.

    Returns:
        torch.Tensor: A 1D tensor representing the mono audio waveform.

    Raises:
        FileNotFoundError: If the audio file cannot be found at primary or fallback paths.
        RuntimeError: If torchaudio fails to load or process the audio file.
    """
    try:
        p: Path = Path(path_str)
        if not p.exists():
            alt_p: Path = Path(path_str.replace("/home/abdullah/", "/home/abdullahanaz012/"))
            if alt_p.exists():
                p = alt_p
            else:
                raise FileNotFoundError(f"Audio file missing: {path_str}")

        waveform, sr = torchaudio.load(str(p))

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        return waveform.squeeze(0)

    except Exception as e:
        logger.error("Audio loading failed for %s: %s", path_str, e)
        raise RuntimeError(f"Failed to load audio {path_str}: {e}") from e


def extract_latents_batch(
    audio_paths: List[str],
    model: Any,
    extractor: Any,
    framework: str,
    target_sr: int,
    device: torch.device
) -> Tuple[List[List[float]], float]:
    """
    Extracts high-dimensional latent vectors for a batch of audio files.

    Args:
        audio_paths (List[str]): A list of file paths to the audio samples.
        model (Any): The instantiated acoustic model (HuggingFace or FunASR).
        extractor (Any): The feature extractor for HuggingFace models, or None for FunASR.
        framework (str): The framework identifier ('huggingface' or 'funasr').
        target_sr (int): The required sampling rate for the model.
        device (torch.device): The computation device (CPU or CUDA).

    Returns:
        Tuple[List[List[float]], float]: A tuple containing a list of latent vectors
        (one per audio file) and the total inference duration in seconds.

    Raises:
        ValueError: If an unsupported framework string is provided.
        RuntimeError: If model inference or tensor operations fail.
    """
    latent_vectors: List[List[float]] = []
    start_time: float = time.perf_counter()

    try:
        with torch.no_grad():
            if framework == "huggingface":
                raw_waveforms: List[np.ndarray] = [
                    _load_and_resample(p, target_sr).cpu().numpy() for p in audio_paths
                ]

                inputs: Dict[str, torch.Tensor] = extractor(
                    raw_waveforms,
                    sampling_rate=target_sr,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs: Any = model(**inputs)
                pooled: torch.Tensor = outputs.last_hidden_state.mean(dim=1)
                latent_vectors.extend(pooled.detach().cpu().tolist())

            elif framework == "funasr":
                results: List[Dict[str, Any]] = model.generate(input=audio_paths, granularity="utterance")
                for res in results:
                    feats: Any = res.get("feats")
                    if feats is not None:
                        latent_vectors.append(np.array(feats).flatten().tolist())
                    else:
                        latent_vectors.append([])
            else:
                raise ValueError(f"Unsupported framework: {framework}")

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_time: float = time.perf_counter() - start_time
        return latent_vectors, inference_time

    except Exception as e:
        logger.error("Latent extraction failed for batch: %s", e)
        raise RuntimeError(f"Batch extraction failed: {e}") from e