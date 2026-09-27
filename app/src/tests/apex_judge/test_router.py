from typing import Any

import pytest
import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI

from app.src.apex_judge.config import JudgeConfig, ModelProvider
from app.src.apex_judge.router import execute_evaluation, load_llm_backend


@pytest.fixture(scope="module")
def local_gpu_config() -> JudgeConfig:
    """Provides a local HuggingFace configuration for unmocked GPU execution."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA architecture is unavailable. Skipping local LLM tests.")
        
    return JudgeConfig(
        provider=ModelProvider.LLAMA,
        model_name="distilgpt2",
        temperature=0.1,
        max_tokens=10,
        device="cuda:0"
    )


@pytest.fixture(scope="module")
def gpt_dummy_config() -> JudgeConfig:
    """Provides a structurally valid GPT configuration with an invalid key to test authentication trapping."""
    return JudgeConfig(
        provider=ModelProvider.GPT,
        model_name="gpt-4o",
        temperature=0.0,
        max_tokens=10,
        api_key="sk-dummy-key-for-auth-rejection-testing"
    )


@pytest.fixture(scope="module")
def gemini_dummy_config() -> JudgeConfig:
    """Provides a structurally valid Gemini configuration for allocation testing."""
    return JudgeConfig(
        provider=ModelProvider.GEMINI,
        model_name="gemini-1.5-pro",
        temperature=0.0,
        max_tokens=10,
        api_key="AIzaSyDummyKeyForAllocationTesting"
    )


# =====================================================================
# LOAD LLM BACKEND TESTS: EXPECTED BEHAVIOUR & ALLOCATION
# =====================================================================

def test_load_llm_backend_expected_behaviour_for_llama_allocation(local_gpu_config: JudgeConfig) -> None:
    """Verifies that the router natively allocates a HuggingFace pipeline onto the target hardware."""
    client: Any = load_llm_backend(local_gpu_config)
    assert isinstance(client, HuggingFacePipeline)
    assert client.pipeline is not None


def test_load_llm_backend_expected_behaviour_for_gpt_allocation(gpt_dummy_config: JudgeConfig) -> None:
    """Verifies that the router natively instantiates the LangChain OpenAI interface."""
    client: Any = load_llm_backend(gpt_dummy_config)
    assert isinstance(client, ChatOpenAI)
    assert client.model_name == "gpt-4o"


def test_load_llm_backend_expected_behaviour_for_gemini_allocation(gemini_dummy_config: JudgeConfig) -> None:
    """Verifies that the router natively instantiates the LangChain Google Generative AI interface."""
    client: Any = load_llm_backend(gemini_dummy_config)
    assert isinstance(client, ChatGoogleGenerativeAI)
    assert client.model == "gemini-1.5-pro"


def test_load_llm_backend_raises_for_corrupted_provider(gpt_dummy_config: JudgeConfig) -> None:
    """Verifies that the router catastrophically fails if the provider enum is mutated in memory."""
    corrupted_config = JudgeConfig.__new__(JudgeConfig)
    object.__setattr__(corrupted_config, "provider", "ANTHROPIC")
    object.__setattr__(corrupted_config, "model_name", "claude-3")
    
    with pytest.raises(RuntimeError, match="Failed to load LLM backend"):
        load_llm_backend(corrupted_config)


# =====================================================================
# EXECUTE EVALUATION TESTS: NATIVE INFERENCE & ERROR TRAPPING
# =====================================================================

def test_execute_evaluation_expected_behaviour_for_huggingface_inference(local_gpu_config: JudgeConfig) -> None:
    """Verifies a successful, unmocked forward pass through a local GPU-bound transformer."""
    client: Any = load_llm_backend(local_gpu_config)
    
    prompt: str = "Analyze this simple sentence."
    response: str = execute_evaluation(prompt, client)
    
    assert isinstance(response, str)
    assert len(response) > 0


def test_execute_evaluation_raises_for_api_rejection(gpt_dummy_config: JudgeConfig) -> None:
    """Verifies that the execution wrapper catches native network authentication failures and raises a RuntimeError."""
    client: Any = load_llm_backend(gpt_dummy_config)
    
    prompt: str = "This should fail at the network layer."
    
    with pytest.raises(RuntimeError, match="LLM execution failed"):
        execute_evaluation(prompt, client)


def test_execute_evaluation_raises_for_unrecognized_client_signature() -> None:
    """Verifies that the router structurally rejects unmapped execution clients to prevent downstream crashes."""
    invalid_client: dict = {"invoke": lambda x: "Spoofed"}
    prompt: str = "This should fail at the type checking layer."
    
    with pytest.raises(RuntimeError, match="LLM execution failed: Unrecognized LLM client signature."):
        execute_evaluation(prompt, invalid_client)


def test_execute_evaluation_raises_for_invalid_prompt_type_on_hf(local_gpu_config: JudgeConfig) -> None:
    """Verifies that the execution block wraps underlying HuggingFace type errors safely."""
    client: Any = load_llm_backend(local_gpu_config)
    
    # Injecting None forces a native crash inside the HF pipeline
    with pytest.raises(RuntimeError, match="LLM execution failed"):
        execute_evaluation(None, client)  # type: ignore