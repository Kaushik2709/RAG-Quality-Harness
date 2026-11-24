"""Treat retrieved documents as untrusted data with explicit risk policies.

RAG systems concatenate retrieved text into a privileged prompt. A retrieved document
can attempt indirect prompt injection, tool hijacking, credential exfiltration, or
citation manipulation.

Workflow:
Retrieved Document
→ Security Scan
→ Risk Classification
→ Evidence Isolation
→ Context Construction
→ Generation
→ Citation Validation
→ Output Security Validation

Risk Categories:
- normal_evidence
- suspicious_instructions
- tool_execution_instructions
- credential_exfiltration_instructions
- system_prompt_impersonation
- malicious_metadata
- suspicious_encoded_content
- citation_manipulation
- cross_tenant_leak_instruction

Configurable Actions:
- allow
- mark_untrusted (default)
- exclude
- quarantine
"""
from __future__ import annotations

import base64
import html
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from src.config import settings
from src.guardrails.regex_patterns import PROMPT_INJECTION_PATTERNS
from src.observability.metrics import metrics

# Specialized injection detection patterns
_INJECTION_PATTERNS = [
    (
        "system_prompt_impersonation",
        re.compile(r"(?i)(?:<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[system\]|\[developer\]|<<sys>>|system\s*:\s*)"),
    ),
    (
        "tool_execution_instructions",
        re.compile(
            r"(?i)\b(?:call|invoke|execute|run)\s+(?:the\s+)?(?:tool|function|shell|bash|command|subprocess|terminal)\b"
        ),
    ),
    (
        "credential_exfiltration_instructions",
        re.compile(
            r"(?i)\b(?:send|post|upload|exfiltrate|curl|wget|fetch|leak)\b.{0,50}\b(?:api[_ -]?key|secret|token|password|credential|env)\b"
        ),
    ),
    (
        "suspicious_instructions",
        re.compile(
            r"(?i)\b(?:ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions|disregard\s+(?:all\s+)?(?:previous\s+)?(?:rules|prompts|guidelines)|you\s+are\s+now\s+in\s+developer\s+mode|operate\s+without\s+(?:any\s+)?safety\s+constraints|act\s+as\s+dan|developer\s+mode)\b"
        ),
    ),
    (
        "citation_manipulation",
        re.compile(
            r"(?i)\b(?:cite\s+chunk[_ -]?id|claim\s+you\s+found\s+this\s+in|attribute\s+this\s+to\s+chunk|invent\s+a\s+citation)\b"
        ),
    ),
    (
        "cross_tenant_leak_instruction",
        re.compile(
            r"(?i)\b(?:reveal|show|print|list|leak)\s+(?:all\s+)?(?:other\s+tenants?|tenant\s+[a-z0-9_-]+|cross-tenant)\s+(?:data|documents|secrets|files)\b"
        ),
    ),
    (
        "html_script_injection",
        re.compile(r"(?i)(?:<\s*(?:script|iframe|object|embed|svg|img\s+src\s*=\s*['\"]?javascript:)[^>]*>|javascript:)"),
    ),
]

_BASE64_INSTRUCTION_RE = re.compile(r"(?:[A-Za-z0-9+/]{40,}={0,2})")


class RiskCategory(str, Enum):
    NORMAL_EVIDENCE = "normal_evidence"
    SUSPICIOUS_INSTRUCTIONS = "suspicious_instructions"
    TOOL_EXECUTION_INSTRUCTIONS = "tool_execution_instructions"
    CREDENTIAL_EXFILTRATION_INSTRUCTIONS = "credential_exfiltration_instructions"
    SYSTEM_PROMPT_IMPERSONATION = "system_prompt_impersonation"
    MALICIOUS_METADATA = "malicious_metadata"
    SUSPICIOUS_ENCODED_CONTENT = "suspicious_encoded_content"
    CITATION_MANIPULATION = "citation_manipulation"
    CROSS_TENANT_LEAK_INSTRUCTION = "cross_tenant_leak_instruction"


@dataclass
class DocumentRiskAssessment:
    chunk_id: str
    risk_category: str
    confidence: float
    triggered_rules: List[str]
    quarantine_reason: Optional[str] = None
    sanitized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "risk_category": self.risk_category,
            "confidence": round(self.confidence, 4),
            "triggered_rules": self.triggered_rules,
            "quarantine_reason": self.quarantine_reason,
            "sanitized": self.sanitized,
        }


@dataclass
class UntrustedScan:
    flagged: bool = False
    rules: List[str] = field(default_factory=list)
    flagged_chunk_ids: List[str] = field(default_factory=list)
    risk_assessments: List[DocumentRiskAssessment] = field(default_factory=list)
    excluded_chunk_ids: List[str] = field(default_factory=list)
    quarantined_chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flagged": self.flagged,
            "rules": self.rules,
            "flagged_chunk_ids": self.flagged_chunk_ids,
            "risk_assessments": [a.to_dict() for a in self.risk_assessments],
            "excluded_chunk_ids": self.excluded_chunk_ids,
            "quarantined_chunk_ids": self.quarantined_chunk_ids,
        }


class QuarantineRegistry:
    """Thread-safe quarantine storage for suspicious retrieved documents."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: List[Dict[str, Any]] = []

    def record(self, doc: Document, assessment: DocumentRiskAssessment) -> None:
        with self._lock:
            self._items.append({
                "chunk_id": assessment.chunk_id,
                "content_preview": doc.page_content[:200],
                "metadata": dict(doc.metadata or {}),
                "assessment": assessment.to_dict(),
            })

    def list_quarantine(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


quarantine_registry = QuarantineRegistry()


def scan_text(text: str) -> List[str]:
    """Scan text for prompt injection and security override patterns."""
    rules: List[str] = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.pattern.search(text):
            rules.append(pattern.name)
    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            rules.append(name)

    # Check for encoded instructions
    for match in _BASE64_INSTRUCTION_RE.finditer(text):
        token = match.group(0)
        try:
            decoded = base64.b64decode(token).decode("utf-8", errors="ignore")
            if any(
                p.search(decoded)
                for _, p in _INJECTION_PATTERNS[:4]
            ):
                rules.append("suspicious_encoded_content")
                break
        except Exception:
            pass

    return sorted(set(rules))


def assess_document_risk(doc: Document) -> DocumentRiskAssessment:
    chunk_id = str(doc.metadata.get("chunk_id", "unknown"))
    text_rules = scan_text(doc.page_content)
    meta_rules: List[str] = []
    for key, value in (doc.metadata or {}).items():
        if isinstance(value, str):
            meta_rules.extend(scan_text(value))

    combined_rules = sorted(set(text_rules + meta_rules))
    if not combined_rules:
        return DocumentRiskAssessment(
            chunk_id=chunk_id,
            risk_category=RiskCategory.NORMAL_EVIDENCE.value,
            confidence=0.0,
            triggered_rules=[],
        )

    # Prioritize highest risk categories
    category = RiskCategory.SUSPICIOUS_INSTRUCTIONS.value
    if any("credential_exfiltration" in r for r in combined_rules):
        category = RiskCategory.CREDENTIAL_EXFILTRATION_INSTRUCTIONS.value
    elif any("tool_execution" in r for r in combined_rules):
        category = RiskCategory.TOOL_EXECUTION_INSTRUCTIONS.value
    elif any("system_prompt_impersonation" in r for r in combined_rules) or any("role_impersonation" in r for r in combined_rules):
        category = RiskCategory.SYSTEM_PROMPT_IMPERSONATION.value
    elif any("cross_tenant" in r for r in combined_rules):
        category = RiskCategory.CROSS_TENANT_LEAK_INSTRUCTION.value
    elif any("citation_manipulation" in r for r in combined_rules):
        category = RiskCategory.CITATION_MANIPULATION.value
    elif meta_rules and not text_rules:
        category = RiskCategory.MALICIOUS_METADATA.value
    elif any("encoded" in r for r in combined_rules):
        category = RiskCategory.SUSPICIOUS_ENCODED_CONTENT.value

    return DocumentRiskAssessment(
        chunk_id=chunk_id,
        risk_category=category,
        confidence=min(1.0, 0.4 + 0.2 * len(combined_rules)),
        triggered_rules=combined_rules,
    )


def scan_documents(documents: Sequence[Document]) -> UntrustedScan:
    scan = UntrustedScan()
    for doc in documents:
        assessment = assess_document_risk(doc)
        if assessment.risk_category != RiskCategory.NORMAL_EVIDENCE.value:
            scan.flagged = True
            scan.rules = sorted(set(scan.rules + assessment.triggered_rules))
            scan.flagged_chunk_ids.append(assessment.chunk_id)
        scan.risk_assessments.append(assessment)
    return scan


def enforce_evidence_security(
    documents: Sequence[Document],
    policy: Optional[str] = None,
) -> Tuple[List[Document], UntrustedScan]:
    """Apply evidence security policy: allow, mark_untrusted, exclude, or quarantine."""
    active_policy = policy or getattr(settings, "RETRIEVED_DOCUMENT_SECURITY_POLICY", "mark_untrusted")
    scan = scan_documents(documents)
    surviving_docs: List[Document] = []

    for doc, assessment in zip(documents, scan.risk_assessments):
        is_risky = assessment.risk_category != RiskCategory.NORMAL_EVIDENCE.value

        if is_risky:
            metrics.inc(
                "rag_prompt_injection_detections_total",
                {"category": assessment.risk_category},
            )
            metrics.inc(
                "rag_prompt_injection_detected_total",
                {"tenant_id": str(doc.metadata.get("tenant_id", "default")), "source": "retrieved_chunk"},
            )

        if not is_risky:
            surviving_docs.append(doc)
            continue

        if active_policy == "allow":
            surviving_docs.append(doc)
        elif active_policy == "mark_untrusted":
            # Tag metadata and keep for generation with isolated rendering
            doc.metadata["is_untrusted_evidence"] = True
            doc.metadata["risk_category"] = assessment.risk_category
            surviving_docs.append(doc)
        elif active_policy == "exclude":
            scan.excluded_chunk_ids.append(assessment.chunk_id)
            metrics.inc("rag_evidence_excluded_total", {"category": assessment.risk_category})
        elif active_policy == "quarantine":
            assessment.quarantine_reason = f"Flagged by evidence policy: {assessment.risk_category}"
            quarantine_registry.record(doc, assessment)
            scan.quarantined_chunk_ids.append(assessment.chunk_id)
            metrics.inc("rag_evidence_quarantined_total", {"category": assessment.risk_category})

    return surviving_docs, scan


def safe_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Only expose scalar, bounded metadata to prompts. Never include nested objects."""
    allowed = (
        "doc_id",
        "document_version",
        "chunk_id",
        "source",
        "page",
        "section_path",
        "tenant_id",
        "risk_category",
    )
    result = {}
    for key in allowed:
        if key in metadata and isinstance(metadata[key], (str, int, float, bool)):
            result[key] = str(metadata[key])[:200]
    return result


def render_evidence_block(doc: Document) -> str:
    """Render evidence wrapped in strict data delimiters with neutralized control tokens."""
    md = safe_metadata(doc.metadata or {})
    header = " | ".join(
        f"{key}={value}"
        for key, value in md.items()
        if key in {"doc_id", "document_version", "chunk_id", "page", "section_path", "risk_category"}
    )
    content = doc.page_content

    # 1. Neutralize special model control tokens
    content = re.sub(
        r"(?i)(?:<\|[^|>]{0,32}\|>|\[system\]|\[inst\]|<<sys>>|<s>\[inst\]|<\|system\|>|System:)",
        "[neutralized_control_token]",
        content,
    )
    # 2. Escape XML/HTML evidence tags so nested content cannot spoof prompt boundaries
    content = content.replace("</evidence>", "&lt;/evidence&gt;").replace("<evidence>", "&lt;evidence&gt;")
    content = content.replace("</evidence_set>", "&lt;/evidence_set&gt;").replace("<evidence_set>", "&lt;evidence_set&gt;")

    is_untrusted = doc.metadata.get("is_untrusted_evidence", False)
    untrusted_attr = ' is_untrusted="true"' if is_untrusted else ""
    risk_attr = f' risk="{md.get("risk_category", "normal_evidence")}"' if is_untrusted else ""

    return (
        f'<evidence chunk_id="{md.get("chunk_id", "")}"{untrusted_attr}{risk_attr}>\n'
        f"DATA_HEADER: {header}\n"
        f"RAW_TEXT:\n{content}\n"
        f"</evidence>"
    )
