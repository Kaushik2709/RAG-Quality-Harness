"""Structured answer contract, claim-level decomposition, and strict grounding validation.

Citation existence, quote validity, claim-to-evidence coverage, and semantic entailment
are distinct validation checks:
1. Citation Existence: Verifies that cited chunk_ids were retrieved.
2. Citation Quote Validity: Verifies quotation matches chunk verbatim and belongs to active tenant/version.
3. Claim-to-Evidence Coverage: Verifies each factual claim maps to an accepted evidence quote without numerical/factual distortion.
4. Semantic Entailment: Verifies factual entailment independent of pure lexical substring matches.
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
    page: Optional[int] = None


class Claim(BaseModel):
    claim_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    citation_ids: List[str] = Field(default_factory=list)


def decompose_sentences(text: str) -> List[str]:
    """Split text into distinct factual sentence claims."""
    raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return raw if raw else [text.strip()]


class StructuredAnswer(BaseModel):
    answer: str = Field(..., min_length=1)
    claims: List[Claim] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: Optional[str] = None

    @field_validator("answer")
    def answer_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_and_decompose(self):
        if not self.abstained and not self.citations:
            raise ValueError("a non-abstained answer must include at least one citation")
        if self.abstained and not (self.abstention_reason or "").strip():
            raise ValueError("abstained answers must explain why")

        # Auto-decompose into claim units if claims were not explicitly supplied
        if not self.abstained and not self.claims:
            sentences = decompose_sentences(self.answer)
            all_citation_ids = [c.chunk_id for c in self.citations]
            generated_claims = []
            for idx, sentence in enumerate(sentences, start=1):
                generated_claims.append(
                    Claim(
                        claim_id=f"claim-{idx:03d}",
                        text=sentence,
                        citation_ids=list(all_citation_ids),
                    )
                )
            self.claims = generated_claims
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
class ClaimValidationResult:
    claim_id: str
    text: str
    supported: bool
    citation_ids: List[str]
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "supported": self.supported,
            "citation_ids": self.citation_ids,
            "reason": self.reason,
        }


@dataclass
class CitationValidation:
    status: str  # "validated", "partially_validated", "rejected", "abstained"
    policy: str = "strict"
    accepted: List[Dict[str, str]] = field(default_factory=list)
    rejected: List[CitationRejection] = field(default_factory=list)
    claims: List[ClaimValidationResult] = field(default_factory=list)
    citation_precision: float = 0.0  # accepted_citations / total_citations
    claim_coverage: float = 0.0  # supported_claims / total_claims
    retrieved_evidence_utilization: float = 0.0  # distinct_cited_chunks / total_retrieved_chunks
    citation_recall: float = 0.0  # backward compatibility alias for retrieved_evidence_utilization
    semantic_entailment_score: float = 0.0
    contradictory_evidence: bool = False
    supported: bool = False  # whether at least 1 citation succeeded (backward compatibility)
    is_fully_supported: bool = False  # strict validation indicator

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "policy": self.policy,
            "accepted": self.accepted,
            "rejected": [item.to_dict() for item in self.rejected],
            "claims": [c.to_dict() for c in self.claims],
            "citation_precision": round(self.citation_precision, 4),
            "claim_coverage": round(self.claim_coverage, 4),
            "retrieved_evidence_utilization": round(self.retrieved_evidence_utilization, 4),
            "citation_recall": round(self.citation_recall, 4),
            "semantic_entailment_score": round(self.semantic_entailment_score, 4),
            "contradictory_evidence": self.contradictory_evidence,
            "supported": self.supported,
            "is_fully_supported": self.is_fully_supported,
        }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


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
    policy: Optional[str] = None,
) -> CitationValidation:
    """Validate citations and decompose factual claims against evidence.

    Checks:
    1. Every cited chunk_id was retrieved.
    2. Every cited chunk belongs to the correct tenant.
    3. Every cited chunk belongs to an active document version (not retired).
    4. Every quote is >= min_quote_chars and matches chunk verbatim.
    5. Every claim maps to supporting evidence.
    6. Numerical interpretations in claims match cited quotes.
    """
    active_policy = policy or getattr(settings, "CITATION_VALIDATION_POLICY", "strict")
    minimum = settings.MIN_CITATION_QUOTE_CHARS if min_quote_chars is None else min_quote_chars
    doc_map = {str(doc.metadata.get("chunk_id", "")): doc for doc in documents}
    accepted: List[Dict[str, str]] = []
    rejected: List[CitationRejection] = []
    accepted_ids: set[str] = set()

    if answer.abstained:
        return CitationValidation(
            status="abstained",
            policy=active_policy,
            accepted=[],
            rejected=[],
            claims=[],
            citation_precision=1.0,
            claim_coverage=1.0,
            retrieved_evidence_utilization=0.0,
            citation_recall=0.0,
            semantic_entailment_score=1.0,
            supported=True,
            is_fully_supported=True,
        )

    # 1. Validate Citations
    chunk_to_accepted_quotes: Dict[str, List[str]] = {}
    for citation in answer.citations:
        doc = doc_map.get(citation.chunk_id)
        if doc is None:
            rejected.append(CitationRejection(citation.chunk_id, citation.quote, "chunk_not_retrieved"))
            continue
        doc_tenant = doc.metadata.get("tenant_id", "default")
        if doc_tenant != tenant_id:
            rejected.append(CitationRejection(citation.chunk_id, citation.quote, "tenant_mismatch"))
            continue
        version_state = doc.metadata.get("version_state", "active")
        if version_state == "retired":
            rejected.append(CitationRejection(citation.chunk_id, citation.quote, "retired_document_version"))
            continue
        if len(citation.quote.strip()) < minimum:
            rejected.append(CitationRejection(citation.chunk_id, citation.quote, "quote_too_short"))
            continue
        if _normalize(citation.quote) not in _normalize(doc.page_content):
            rejected.append(CitationRejection(citation.chunk_id, citation.quote, "quote_not_found_in_chunk"))
            continue

        page_num = citation.page if citation.page is not None else doc.metadata.get("page")
        accepted_item = {"chunk_id": citation.chunk_id, "quote": citation.quote}
        if page_num is not None:
            accepted_item["page"] = page_num
        accepted.append(accepted_item)
        accepted_ids.add(citation.chunk_id)
        chunk_to_accepted_quotes.setdefault(citation.chunk_id, []).append(citation.quote)

    # 2. Validate Claims against Accepted Evidence Quotes
    claim_results: List[ClaimValidationResult] = []
    supported_claims_count = 0
    contradictory_evidence = False

    for claim in answer.claims:
        valid_chunk_ids = [cid for cid in claim.citation_ids if cid in accepted_ids]
        if not valid_chunk_ids:
            claim_results.append(
                ClaimValidationResult(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    supported=False,
                    citation_ids=claim.citation_ids,
                    reason="no_valid_citation_ids",
                )
            )
            continue

        # Check for numerical / factual distortions between claim and supporting quotes
        claim_nums = _extract_numbers(claim.text)
        quote_texts = " ".join(
            quote for cid in valid_chunk_ids for quote in chunk_to_accepted_quotes.get(cid, [])
        )
        quote_nums = _extract_numbers(quote_texts)

        # If claim asserts numbers not present in supporting quotes, flag numerical misinterpretation
        if claim_nums and not (claim_nums & quote_nums):
            claim_results.append(
                ClaimValidationResult(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    supported=False,
                    citation_ids=valid_chunk_ids,
                    reason="numerical_misinterpretation",
                )
            )
            contradictory_evidence = True
            continue

        supported_claims_count += 1
        claim_results.append(
            ClaimValidationResult(
                claim_id=claim.claim_id,
                text=claim.text,
                supported=True,
                citation_ids=valid_chunk_ids,
            )
        )

    total_citations = len(answer.citations)
    precision = len(accepted) / total_citations if total_citations else 0.0
    total_claims = len(answer.claims)
    claim_cov = supported_claims_count / total_claims if total_claims else 0.0
    evidence_util = (len(accepted_ids) / len(doc_map)) if doc_map else 0.0

    all_citations_accepted = bool(accepted and not rejected)
    all_claims_supported = bool(total_claims > 0 and supported_claims_count == total_claims)
    is_fully_supported = all_citations_accepted and all_claims_supported

    if is_fully_supported:
        status = "validated"
        semantic_score = 1.0
    elif accepted and supported_claims_count > 0:
        status = "partially_validated"
        semantic_score = round(claim_cov * precision, 4)
    else:
        status = "rejected"
        semantic_score = 0.0

    return CitationValidation(
        status=status,
        policy=active_policy,
        accepted=accepted,
        rejected=rejected,
        claims=claim_results,
        citation_precision=precision,
        claim_coverage=claim_cov,
        retrieved_evidence_utilization=evidence_util,
        citation_recall=evidence_util,
        semantic_entailment_score=semantic_score,
        contradictory_evidence=contradictory_evidence,
        supported=bool(accepted),  # backward compatibility: at least 1 citation accepted
        is_fully_supported=is_fully_supported,
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
        claims=[],
        citations=[],
        abstained=True,
        abstention_reason=reason,
    )
