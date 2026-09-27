import numpy as np
import pytest
import torch
from transformers import AutoFeatureExtractor, WavLMModel

from app.src.acoustic_edge.config import AcousticConfig
from app.src.acoustic_edge.encoder import (
    encode_audio_to_embeddings,
    load_wavlm_components,
)
from app.src.orchestration.types import AudioChunk


@pytest.fixture
def production_config() -> AcousticConfig:
    return AcousticConfig(
        wavlm_repo_id="microsoft/wavlm-large",
        wavlm_revision="refs/pr/7",
        device="cpu",
        use_float16=False
    )


def test_load_wavlm_components_expected_behaviour_for_production_repository(production_config: AcousticConfig) -> None:
    extractor, model = load_wavlm_components(production_config)
    
    # AutoFeatureExtractor is a factory, so we assert its functional interface instead of isinstance
    assert hasattr(extractor, "sampling_rate")
    assert hasattr(extractor, "__call__")
    
    assert isinstance(model, WavLMModel)
    assert model.device.type == "cpu"
    assert not model.training


def test_load_wavlm_components_expected_behaviour_for_float16_precision() -> None:
    fp16_config = AcousticConfig(
        wavlm_repo_id="microsoft/wavlm-large",
        wavlm_revision="refs/pr/7",
        device="cpu",
        use_float16=True
    )
    
    extractor, model = load_wavlm_components(fp16_config)
    assert isinstance(model, WavLMModel)
    assert next(model.parameters()).dtype == torch.float16


def test_load_wavlm_components_raises_for_invalid_repository() -> None:
    invalid_config = AcousticConfig(
        wavlm_repo_id="invalid_user/non_existent_model_12345",
        wavlm_revision="refs/pr/7",
        device="cpu",
        use_float16=False
    )
    
    with pytest.raises(RuntimeError, match="Failed to load WavLM components"):
        load_wavlm_components(invalid_config)


def test_encode_audio_to_embeddings_expected_behaviour_for_valid_audio_chunk(production_config: AcousticConfig) -> None:
    extractor, model = load_wavlm_components(production_config)
    
    sample_rate: int = extractor.sampling_rate
    audio_data: np.ndarray = np.random.randn(sample_rate).astype(np.float32)
    chunk = AudioChunk(data=audio_data, sample_rate=sample_rate, timestamp_sec=0.0)
    
    embedding = encode_audio_to_embeddings(
        audio=chunk, 
        extractor=extractor, 
        model=model, 
        config=production_config
    )
    
    assert isinstance(embedding, torch.Tensor)
    assert embedding.dim() == 1
    assert embedding.shape[0] == model.config.hidden_size


def test_encode_audio_to_embeddings_raises_for_empty_audio_array(production_config: AcousticConfig) -> None:
    extractor, model = load_wavlm_components(production_config)
    
    # Explicitly defining float32 to pass AudioChunk's strict dataclass validation
    empty_data: np.ndarray = np.array([], dtype=np.float32)
    chunk = AudioChunk(data=empty_data, sample_rate=extractor.sampling_rate, timestamp_sec=0.0)
    
    with pytest.raises(ValueError, match="Cannot extract embeddings from an empty audio chunk"):
        encode_audio_to_embeddings(
            audio=chunk, 
            extractor=extractor, 
            model=model, 
            config=production_config
        )


def test_encode_audio_to_embeddings_raises_for_sample_rate_mismatch(production_config: AcousticConfig) -> None:
    extractor, model = load_wavlm_components(production_config)
    
    invalid_sr: int = extractor.sampling_rate + 8000
    audio_data: np.ndarray = np.random.randn(invalid_sr).astype(np.float32)
    chunk = AudioChunk(data=audio_data, sample_rate=invalid_sr, timestamp_sec=0.0)
    
    with pytest.raises(ValueError, match="Sample rate mismatch: expected"):
        encode_audio_to_embeddings(
            audio=chunk, 
            extractor=extractor, 
            model=model, 
            config=production_config
        )


def test_encode_audio_to_embeddings_raises_for_tensor_forward_pass_failure(production_config: AcousticConfig) -> None:
    extractor, model = load_wavlm_components(production_config)
    
    # We construct a valid chunk to bypass __post_init__ validation...
    valid_data: np.ndarray = np.zeros(1, dtype=np.float32)
    chunk = AudioChunk(data=valid_data, sample_rate=extractor.sampling_rate, timestamp_sec=0.0)
    
    # ...then use object.__setattr__ to inject corrupt data to trigger the forward pass failure
    object.__setattr__(chunk, 'data', np.array(["invalid", "string", "data"], dtype=object))
    
    with pytest.raises(RuntimeError, match="WavLM forward pass failed"):
        encode_audio_to_embeddings(
            audio=chunk, 
            extractor=extractor, 
            model=model, 
            config=production_config
        )