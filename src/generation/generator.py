import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from src.config import settings

def get_chat_model(provider: Optional[str] = None) -> BaseChatModel:
    """
    Returns a standard production LangChain Chat model instance (e.g. Gemini or Claude).
    Raises an exception if the required provider or API key is missing.
    """
    prov = (provider or settings.LLM_PROVIDER).lower()

    if prov in ["gemini", "google"]:
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "[Generator Error] GEMINI_API_KEY is not configured. "
                "Please set GEMINI_API_KEY in your .env file or environment."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=api_key,
                temperature=0.0
            )
        except Exception as e:
            raise RuntimeError(f"[Generator Error] Failed to initialize ChatGoogleGenerativeAI ({settings.GEMINI_MODEL}): {e}")

    elif prov in ["claude", "anthropic"]:
        api_key = settings.CLAUDE_API_KEY or os.environ.get("CLAUDE_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "[Generator Error] CLAUDE_API_KEY is not configured. "
                "Please set CLAUDE_API_KEY in your .env file or environment."
            )
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=settings.CLAUDE_MODEL,
                anthropic_api_key=api_key,
                temperature=0.0
            )
        except Exception as e:
            raise RuntimeError(f"[Generator Error] Failed to initialize ChatAnthropic ({settings.CLAUDE_MODEL}): {e}")

    else:
        raise ValueError(f"[Generator Error] Unsupported LLM provider '{prov}'. Expected 'gemini' or 'claude'.")

