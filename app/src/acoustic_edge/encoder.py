from typing import Tuple

import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, WavLMConfig

from app.src.acoustic_edge.config import AcousticConfig
from app.src.orchestration.types import AudioChunk


def load_wavlm_components(
    config: AcousticConfig
) -> Tuple[Wav2Vec2FeatureExtractor, WavLMModel]:
    """
    Initializes and maps the WavLM feature extractor and neural network model.

    Args:
        config (AcousticConfig): The active configuration matrix.

    Returns:
        Tuple[AutoFeatureExtractor, WavLMModel]: The instantiated pipeline components.

    Raises:
        RuntimeError: If model initialization or device mapping fails.
    """
    try:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            config.wavlm_repo_id,
            cache_dir=str(config.local_weights_dir)
        )

        arch_config = WavLMConfig.from_pretrained(
            config.wavlm_repo_id,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            cache_dir=str(config.local_weights_dir)
        )
        
        model = WavLMModel.from_pretrained(
            config.wavlm_repo_id,
            revision="refs/pr/7",  # CRITICAL: Instructs HF to use the branch with safetensors
            config=arch_config,    # CRITICAL: Overrides the PR's broken config.json
            use_safetensors=True,  # CRITICAL: Bypasses the PyTorch CVE security block
            cache_dir=str(config.local_weights_dir),
            ignore_mismatched_sizes=False
        )
        
        model.eval()
        model = model.to(config.device)
        
        if config.use_float16:
            model = model.half()
            
        return extractor, model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load WavLM components: {e}") from e


def encode_audio_to_embeddings(
    audio: AudioChunk,
    extractor: Wav2Vec2FeatureExtractor,
    model: WavLMModel,
    config: AcousticConfig
) -> torch.Tensor:
    """
    Executes the PyTorch forward pass and mean pooling to generate dense acoustic embeddings.

    Args:
        audio (AudioChunk): The raw acoustic signal slice.
        extractor (Wav2Vec2FeatureExtractor): The Hugging Face feature pre-processor.
        model (WavLMModel): The loaded WavLM neural network.
        config (AcousticConfig): The active configuration matrix for hardware routing.

    Returns:
        torch.Tensor: A 1D tensor representing the mean-pooled latent features.

    Raises:
        ValueError: If the input audio array is empty or the sample rate is incompatible.
        RuntimeError: If the tensor forward pass fails.
    """
    if audio.data.size == 0:
        raise ValueError("Cannot extract embeddings from an empty audio chunk.")
        
    if audio.sample_rate != extractor.sampling_rate:
        raise ValueError(
            f"Sample rate mismatch: expected {extractor.sampling_rate}, "
            f"got {audio.sample_rate}."
        )

    try:
        inputs = extractor(
            audio.data,
            sampling_rate=audio.sample_rate,
            return_tensors="pt"
        )
        
        input_values: torch.Tensor = inputs.input_values.to(config.device)
        
        if config.use_float16:
            input_values = input_values.half()

        with torch.no_grad():
            outputs = model(input_values)
            
        pooled = outputs.last_hidden_state.mean(dim=1)
        return pooled.squeeze(0).cpu()
        
    except Exception as e:
        raise RuntimeError(f"WavLM forward pass failed: {e}") from e