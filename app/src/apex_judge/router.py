from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from transformers import pipeline

from app.src.apex_judge.config import JudgeConfig, ModelProvider


def load_llm_backend(config: JudgeConfig) -> Any:
    """
    Initializes the execution client based on the specified model provider.

    Args:
        config (JudgeConfig): The active configuration matrix.

    Returns:
        Any: An instantiated LangChain interface or HuggingFace pipeline.

    Raises:
        RuntimeError: If client initialization or GPU allocation fails.
    """
    try:
        if config.provider == ModelProvider.GPT:
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key  # type: ignore
            )

        if config.provider == ModelProvider.GEMINI:
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                google_api_key=config.api_key  # type: ignore
            )

        if config.provider == ModelProvider.LLAMA:
            hf_pipeline = pipeline(
                "text-generation",
                model=config.model_name,
                device=config.device,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                model_kwargs={"torch_dtype": "auto"}
            )
            return HuggingFacePipeline(pipeline=hf_pipeline)

        raise ValueError(f"Unsupported provider: {config.provider}")

    except Exception as exc:
        provider_name = getattr(config.provider, 'value', str(config.provider))
        raise RuntimeError(f"Failed to load LLM backend {provider_name}: {exc}") from exc


def execute_evaluation(prompt: str, client: Any) -> str:
    """
    Executes a blocking forward pass through the loaded LLM backend.

    Args:
        prompt (str): The compiled InstructERC prompt.
        client (Any): The instantiated LangChain interface.

    Returns:
        str: The raw generated coaching feedback.

    Raises:
        RuntimeError: If the execution forward pass collapses.
    """
    try:
        if isinstance(client, BaseChatModel):
            response = client.invoke([HumanMessage(content=prompt)])
            return str(response.content)
        
        if isinstance(client, HuggingFacePipeline):
            response = client.invoke(prompt)
            return str(response)
            
        raise TypeError("Unrecognized LLM client signature.")
    except Exception as exc:
        raise RuntimeError(f"LLM execution failed: {exc}") from exc