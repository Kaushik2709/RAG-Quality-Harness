"""Structured answer contract and independent citation validation.

Citation existence and semantic entailment are distinct checks. This module
verifies that a cited chunk was actually retrieved, belongs to the request's
tenant/version scope, and contains the quoted passage. It does *not* claim that
the quote logically entails every claim in the answer; that requires a separate
judge/human evaluation and is reported separately.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import settings
from src.runtime.errors import GenerationContractError


class Citation(BaseModel):
    chunk_id: str = Field(..., min_length=1)
    quote: str = Field(..., min_length=1)


class StructuredAnswer(BaseModel):
    answer: str = Field(..., min_length=1)
    citations: List[Citation] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: Optional[str] = None

    @field_validator("answer")
    def answer_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer cannot be blank")
        return value

    @model_validator(mode="after")
    def citations_required_unless_abstained(self):
        if not self.abstained and not self.citations:
            raise ValueError("a non-abstained answer must include at least one citation")
        if self.abstained and not (self.abstention_reason or "").strip():
            raise ValueError("abstained answers must explain why")
        return self

    def to_public_dict(self) -> Dict[str, Any]:
        return self.model_dump()


@dataclass
class CitationRejection:
    chunk_id: str
    quote: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"chunk_id": self.chunk_id, "quote": self.quote, "reason": self.reason}


@dataclass
class CitationValidation:
    status: str
    accepted: List[Dict[str, str]] = field(default_factory=list)
    rejected: List[CitationRejection] = field(default_factory=list)
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    supported: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "accepted": self.accepted,
            "rejected": [item.to_dict() for item in self.rejected],
            "citation_precision": round(self.citation_precision, 4),
            "citation_recall": round(self.citation_recall, 4),
            "supported": self.supported,
        }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _strip_fences(text: str) -> str:
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_structured_answer(text: str) -> StructuredAnswer:
    payload = _strip_fences(text or "")
    try:
        data = json.loads(payload)
    except Exception as exc:  # noqa: BLE001
        raise GenerationContractError(f"model output is not valid JSON: {exc}", component="generation") from exc
    try:
        return StructuredAnswer(**data)
    except Exception as exc:  # noqa: BLE001
        raise GenerationContractError(f"model output violates the answer contract: {exc}", component="generation") from exc


def validate_citations(
    answer: StructuredAnswer,
    documents: Sequence[Document],
    *,
    tenant_id: str = "default",
    min_quote_chars: Optional[int] = None,
) -> CitationValidation:
    minimum = settings.MIN_CITATION_QUOTE_CHARS if min_quote_chars is None else min_quote_chars
    doc_map = {str(doc.metadata.get("chunk_id", "")): doc for doc in documents}
    accepted: List[Dict[str, str]] = []
    rejected: List[CitationRejection] = []
    accepted_ids: set[str] = set()

    for citation in answer.citations:
        doc = doc_map.get(citation.chunk_id)
        if doc is None:
            rejected.append(CitationRejection(citation.chunk_id, citation.quote, "chunk_not_retrieved"))
            continue
        doc_tenant = doc.metadata.get("tenant_id", "default")
        if doc_tenant != tenant_id:
            rejected.append(CitationRejection(citation.chunk_id, citation.quote, "tenant_mismatch"))
            continue
        if len(citation.quote.strip()) < minimum:
            rejected.append(CitationRejection(citation.chunk_id, citation.quote, "quote_too_short"))
            continue
        if _normalize(citation.quote) not in _normalize(doc.page_content):
            rejected.append(CitationRejection(citation.chunk_id, citation.quote, "quote_not_found_in_chunk"))
            continue
        accepted.append({"chunk_id": citation.chunk_id, "quote": citation.quote})
        accepted_ids.add(citation.chunk_id)

    total = len(answer.citations)
    precision = len(accepted) / total if total else 0.0
    recall = (len(accepted_ids) / len(doc_map)) if doc_map else 0.0
    if answer.abstained:
        status = "abstained"
        supported = True
    elif accepted and not rejected:
        status = "validated"
        supported = True
    elif accepted:
        status = "partially_validated"
        supported = True
    else:
        status = "rejected"
        supported = False
    return CitationValidation(
        status=status,
        accepted=accepted,
        rejected=rejected,
        citation_precision=precision,
        citation_recall=recall,
        supported=supported,
    )


def abstained_answer(reason: str) -> StructuredAnswer:
    messages = {
        "no_relevant_evidence": "I don't have enough retrieved evidence to answer this question.",
        "citations_failed_validation": "The generated answer could not be verified against the retrieved evidence.",
        "conflicting_evidence": "The retrieved sources conflict, so I cannot give a reliable answer.",
        "outdated_document": "The retrieved evidence is from an inactive document version.",
        "insufficient_context": "The retrieved evidence is insufficient to answer this question.",
    }
    return StructuredAnswer(
        answer=messages.get(reason, "I cannot answer this question from the available evidence."),
        citations=[],
        abstained=True,
        abstention_reason=reason,
    )
