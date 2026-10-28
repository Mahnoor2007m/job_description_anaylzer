"""LLM factory: Groq or Gemini."""
from typing import Optional

from app.config import DEFAULT_PROVIDER, GEMINI_MODEL, GROQ_MODEL


class ConfigError(Exception):
    """Raised when the LLM is not configured correctly (e.g. missing API key)."""


def get_llm(provider: Optional[str] = None, api_key: Optional[str] = None):
    import os

    provider = (provider or DEFAULT_PROVIDER).lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ConfigError("GROQ_API_KEY is missing. Add it to your .env file or the sidebar.")
        return ChatGroq(model=GROQ_MODEL, api_key=key, temperature=0)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ConfigError("GOOGLE_API_KEY is missing. Add it to your .env file or the sidebar.")
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=key, temperature=0)

    raise ConfigError(f"Unknown provider '{provider}'. Use 'groq' or 'gemini'.")
