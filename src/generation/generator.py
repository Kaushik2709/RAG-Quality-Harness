"""Grounded generation with a validated structured-answer contract."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import settings
from src.generation.citations import StructuredAnswer, parse_structured_answer
from src.generation.prompt_builder import build_structured_prompt_template


def get_chat_model(provider: Optional[str] = None) -> BaseChatModel:
    """Return a provider client. Raises explicitly when configuration is missing."""
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
                temperature=0.0,
                timeout=settings.LLM_TIMEOUT_SECONDS,
                max_retries=settings.LLM_MAX_RETRIES,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"[Generator Error] Failed to initialize ChatGoogleGenerativeAI ({settings.GEMINI_MODEL}): {exc}"
            ) from exc

    if prov in ["claude", "anthropic"]:
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
                temperature=0.0,
                timeout=settings.LLM_TIMEOUT_SECONDS,
                max_retries=settings.LLM_MAX_RETRIES,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"[Generator Error] Failed to initialize ChatAnthropic ({settings.CLAUDE_MODEL}): {exc}"
            ) from exc

    raise ValueError(f"[Generator Error] Unsupported LLM provider '{prov}'. Expected 'gemini' or 'claude'.")


@dataclass
class GenerationResult:
    structured: StructuredAnswer
    model_id: str
    usage: Dict[str, Any] = field(default_factory=dict)
    usage_source: str = "unavailable"
    latency_ms: float = 0.0
    attempts: int = 1
    raw_text: str = ""


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _is_transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in ("429", "503", "resource_exhausted", "unavailable", "timeout", "temporar"))


def _usage_from_message(message: Any, context_text: str, answer_text: str) -> tuple[Dict[str, Any], str]:
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        normalized = {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        if normalized["total_tokens"] == 0:
            normalized["total_tokens"] = normalized["input_tokens"] + normalized["output_tokens"]
        return normalized, "provider"
    return {
        "input_tokens": max(1, len(context_text) // 4),
        "output_tokens": max(1, len(answer_text) // 4),
        "total_tokens": max(2, len(context_text) // 4 + len(answer_text) // 4),
    }, "estimate"


def generate_structured(
    query: str,
    context_text: str,
    *,
    chat_model: BaseChatModel,
    model_id: Optional[str] = None,
    max_retries: Optional[int] = None,
) -> GenerationResult:
    """Generate and validate JSON. Non-JSON or contract-violating output raises;
    it is never silently returned as a plain answer."""
    retries = settings.LLM_MAX_RETRIES if max_retries is None else max_retries
    prompt = build_structured_prompt_template()
    chain = prompt | chat_model
    started = time.perf_counter()
    last_error: Optional[BaseException] = None
    message = None
    attempts = 0
    for attempt in range(retries + 1):
        attempts = attempt + 1
        try:
            message = chain.invoke({"query": query, "context": context_text})
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries and _is_transient(exc):
                time.sleep(min(2.0, 0.5 * (attempt + 1)))
                continue
            raise
    if message is None:
        raise last_error or RuntimeError("generation produced no message")

    raw = _message_text(message)
    structured = parse_structured_answer(raw)
    usage, usage_source = _usage_from_message(message, context_text, raw)
    resolved_model_id = model_id or (
        settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini" else settings.CLAUDE_MODEL
    )
    return GenerationResult(
        structured=structured,
        model_id=resolved_model_id,
        usage=usage,
        usage_source=usage_source,
        latency_ms=(time.perf_counter() - started) * 1000,
        attempts=attempts,
        raw_text=raw,
    )
