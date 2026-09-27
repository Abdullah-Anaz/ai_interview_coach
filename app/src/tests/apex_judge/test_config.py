import os
from typing import Generator

import pytest
import torch

from app.src.apex_judge.config import JudgeConfig, ModelProvider


@pytest.fixture
def hardware_check() -> None:
    """Validates runtime environments but enforces CPU operations for structural configuration allocation."""
    _ = torch.cuda.is_available()


@pytest.fixture
def clean_environment() -> Generator[None, None, None]:
    """
    Ensures a pristine OS environment without using mocks to natively test API key resolution.
    Safely restores the host environment state post-execution.
    """
    original_openai = os.environ.get("OPENAI_API_KEY")
    original_google = os.environ.get("GOOGLE_API_KEY")
    
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]
        
    yield
    
    if original_openai is not None:
        os.environ["OPENAI_API_KEY"] = original_openai
    if original_google is not None:
        os.environ["GOOGLE_API_KEY"] = original_google


# =====================================================================
# EXPECTED BEHAVIOUR & EDGE CASES
# =====================================================================

def test_judge_config_expected_behaviour_for_llama_provider(hardware_check: None) -> None:
    """Tests if the configuration safely instantiates a local model without requiring API keys."""
    config = JudgeConfig(
        provider=ModelProvider.LLAMA,
        model_name="meta-llama/Meta-Llama-3-8B",
        temperature=0.7,
        max_tokens=256
    )
    
    assert config.provider == ModelProvider.LLAMA
    assert config.model_name == "meta-llama/Meta-Llama-3-8B"
    assert config.temperature == 0.7
    assert config.max_tokens == 256
    assert config.api_key is None
    assert config.device == "cuda:0"


def test_judge_config_expected_behaviour_for_explicit_api_keys(hardware_check: None) -> None:
    """Tests if cloud providers correctly initialize when keys are passed explicitly via code."""
    config = JudgeConfig(
        provider=ModelProvider.GPT,
        model_name="gpt-4o",
        temperature=0.0,
        max_tokens=100,
        api_key="sk-test-explicit-key"
    )
    
    assert config.provider == ModelProvider.GPT
    assert config.api_key == "sk-test-explicit-key"


def test_judge_config_expected_behaviour_for_os_environment_resolution(clean_environment: None) -> None:
    """Tests if the configuration dynamically extracts missing keys from the native OS environment."""
    os.environ["GOOGLE_API_KEY"] = "AIzaSyTestNativeResolution"
    
    config = JudgeConfig(
        provider=ModelProvider.GEMINI,
        model_name="gemini-1.5-pro",
        temperature=1.0,
        max_tokens=1024
    )
    
    assert config.provider == ModelProvider.GEMINI
    assert config.api_key == "AIzaSyTestNativeResolution"


def test_judge_config_expected_behaviour_for_temperature_boundaries() -> None:
    """Tests if the configuration accepts exact allowed minimum and maximum temperature float boundaries."""
    config_min = JudgeConfig(ModelProvider.LLAMA, "test", temperature=0.0, max_tokens=10)
    config_max = JudgeConfig(ModelProvider.LLAMA, "test", temperature=2.0, max_tokens=10)
    
    assert config_min.temperature == 0.0
    assert config_max.temperature == 2.0


def test_judge_config_expected_behaviour_for_min_tokens() -> None:
    """Tests if the configuration accepts the absolute lowest allowed generation length."""
    config = JudgeConfig(ModelProvider.LLAMA, "test", temperature=1.0, max_tokens=1)
    assert config.max_tokens == 1


# =====================================================================
# ERROR CASES: TYPE POLLUTION
# =====================================================================

def test_judge_config_raises_for_invalid_provider_type() -> None:
    """Tests if initialization traps standard strings passed instead of strict Enums."""
    with pytest.raises(TypeError, match="provider must be a valid ModelProvider enum."):
        JudgeConfig("gpt", "gpt-4", 0.5, 100)  # type: ignore


def test_judge_config_raises_for_invalid_model_name_type() -> None:
    """Tests if initialization strictly rejects non-string model identifiers."""
    with pytest.raises(ValueError, match="model_name must be a valid string identifier."):
        JudgeConfig(ModelProvider.LLAMA, 12345, 0.5, 100)  # type: ignore


def test_judge_config_raises_for_invalid_temperature_type() -> None:
    """Tests if initialization strictly rejects string types for numerical temperature."""
    with pytest.raises(ValueError, match="temperature must be a float between 0.0 and 2.0."):
        JudgeConfig(ModelProvider.LLAMA, "test", "0.5", 100)  # type: ignore


def test_judge_config_raises_for_invalid_max_tokens_type() -> None:
    """Tests if initialization strictly rejects floats for exact token counts."""
    with pytest.raises(ValueError, match="max_tokens must be a strictly positive integer."):
        JudgeConfig(ModelProvider.LLAMA, "test", 0.5, 100.5)  # type: ignore


def test_judge_config_raises_for_invalid_device_type() -> None:
    """Tests if initialization strictly rejects numerical values mapped to device targets."""
    with pytest.raises(TypeError, match="device must be a string."):
        JudgeConfig(ModelProvider.LLAMA, "test", 0.5, 100, device=0)  # type: ignore


# =====================================================================
# ERROR CASES: VALUE BOUNDARY VIOLATIONS
# =====================================================================

def test_judge_config_raises_for_empty_model_name() -> None:
    """Tests if initialization rejects missing or whitespace-only model identifiers."""
    with pytest.raises(ValueError, match="model_name must be a valid string identifier."):
        JudgeConfig(ModelProvider.LLAMA, "   ", 0.5, 100)


def test_judge_config_raises_for_negative_temperature() -> None:
    """Tests if initialization rejects mathematically impossible deterministic sampling targets."""
    with pytest.raises(ValueError, match="temperature must be a float between 0.0 and 2.0."):
        JudgeConfig(ModelProvider.LLAMA, "test", -0.1, 100)


def test_judge_config_raises_for_excessive_temperature() -> None:
    """Tests if initialization rejects hallucination-inducing overflow sampling targets."""
    with pytest.raises(ValueError, match="temperature must be a float between 0.0 and 2.0."):
        JudgeConfig(ModelProvider.LLAMA, "test", 2.1, 100)


def test_judge_config_raises_for_zero_max_tokens() -> None:
    """Tests if initialization rejects a zero generation cap which mathematically stalls inference."""
    with pytest.raises(ValueError, match="max_tokens must be a strictly positive integer."):
        JudgeConfig(ModelProvider.LLAMA, "test", 0.5, 0)


def test_judge_config_raises_for_negative_max_tokens() -> None:
    """Tests if initialization rejects impossible negative memory allocation targets."""
    with pytest.raises(ValueError, match="max_tokens must be a strictly positive integer."):
        JudgeConfig(ModelProvider.LLAMA, "test", 0.5, -50)


# =====================================================================
# ERROR CASES: AUTHENTICATION DEGRADATION
# =====================================================================

def test_judge_config_raises_for_missing_gpt_api_key(clean_environment: None) -> None:
    """Tests if the configuration triggers an immediate crash when OpenAI routing lacks credentials."""
    with pytest.raises(ValueError, match="API key is required for cloud provider: gpt"):
        JudgeConfig(ModelProvider.GPT, "gpt-4o", 0.5, 100)


def test_judge_config_raises_for_missing_gemini_api_key(clean_environment: None) -> None:
    """Tests if the configuration triggers an immediate crash when Gemini routing lacks credentials."""
    with pytest.raises(ValueError, match="API key is required for cloud provider: gemini"):
        JudgeConfig(ModelProvider.GEMINI, "gemini-1.5-pro", 0.5, 100)