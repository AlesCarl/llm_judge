import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def load_model(
    llm_backend: str = "openai",
    model: str = "gpt-5-mini",
    temperature: float | None = None,
) -> BaseChatModel:
    if llm_backend == "ollama":
        return ChatOllama(
            model=model,
            temperature=temperature if temperature is not None else 0,
            validate_model_on_init=True,
            base_url=os.getenv("OLLAMA_API_URL"),
        )

    if llm_backend == "openai":
        kwargs = {"model_name": model}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatOpenAI(**kwargs)

    if llm_backend == "deepseek":
        kwargs = {"model": model, "base_url": "https://api.deepseek.com"}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatDeepSeek(**kwargs)

    raise ValueError(f"Unsupported llm backend: {llm_backend}")
