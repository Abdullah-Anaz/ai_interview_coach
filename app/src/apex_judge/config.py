import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ModelProvider(str, Enum):
    """Enumeration of supported LLM execution backends."""
    GPT = "gpt"
    GEMINI = "gemini"
    LLAMA = "llama"


@dataclass(frozen=True)
class JudgeConfig:
    """
    Configuration matrix for the Apex LLM Judge.

    Attributes:
        provider (ModelProvider): The backend execution engine to route prompts to.
        model_name (str): The specific model identifier (e.g., 'gpt-4o', 'gemini-1.5-pro', 'meta-llama/Meta-Llama-3-8B').
        temperature (float): The sampling temperature governing output determinism.
        max_tokens (int): The absolute generation token limit.
        api_key (Optional[str]): The authentication token for cloud providers.
        device (str): Compute device mapping for local model execution (e.g., "cuda:0").
    """
    provider: ModelProvider
    model_name: str
    temperature: float
    max_tokens: int
    api_key: Optional[str] = None
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        """
        Enforces runtime validation of configuration state and environment requirements.

        Raises:
            ValueError: If numerical constraints are violated or required keys are missing.
            TypeError: If attribute types are incorrect.
        """
        if not isinstance(self.provider, ModelProvider):
            raise TypeError("provider must be a valid ModelProvider enum.")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a valid string identifier.")
        if not isinstance(self.temperature, (int, float)) or not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be a float between 0.0 and 2.0.")
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a strictly positive integer.")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string.")

        if self.provider in (ModelProvider.GPT, ModelProvider.GEMINI) and not self.api_key:
            env_key = os.getenv("OPENAI_API_KEY") if self.provider == ModelProvider.GPT else os.getenv("GOOGLE_API_KEY")
            if not env_key:
                raise ValueError(f"API key is required for cloud provider: {self.provider.value}")
            object.__setattr__(self, "api_key", env_key)