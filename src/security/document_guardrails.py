"""Treat retrieved documents as untrusted input.

RAG systems concatenate retrieved text into a privileged prompt. A document can
therefore attempt a prompt injection just as a user can. We detect instruction
patterns and mark them, and the prompt renders evidence inside explicit data
delimiters with a no-instructions rule. Detection is not treated as proof of
safety: the model contract and citation validation remain the enforcement layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from src.guardrails.regex_patterns import PROMPT_INJECTION_PATTERNS

_EXTRA_PATTERNS = [
    ("embedded_system_prompt", re.compile(r"(?i)\b(system\s+prompt|developer\s+message|hidden\s+instructions?)\b")),
    ("embedded_tool_call", re.compile(r"(?i)\b(call|invoke|execute)\s+(the\s+)?(tool|function|shell|command)\b")),
    ("embedded_exfiltration", re.compile(r"(?i)\b(send|post|upload|exfiltrate)\b.{0,40}\b(api[_ -]?key|secret|token|password)\b")),
    ("role_marker", re.compile(r"(?im)^\s*(system|assistant|developer)\s*:\s*")),
]


@dataclass
class UntrustedScan:
    flagged: bool = False
    rules: List[str] = field(default_factory=list)
    flagged_chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {"flagged": self.flagged, "rules": self.rules, "flagged_chunk_ids": self.flagged_chunk_ids}


def scan_text(text: str) -> List[str]:
    rules: List[str] = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.pattern.search(text):
            rules.append(pattern.name)
    for name, pattern in _EXTRA_PATTERNS:
        if pattern.search(text):
            rules.append(name)
    return sorted(set(rules))


def scan_documents(documents: Sequence) -> UntrustedScan:
    scan = UntrustedScan()
    for doc in documents:
        rules = scan_text(doc.page_content)
        metadata_rules = []
        for key, value in (doc.metadata or {}).items():
            if isinstance(value, str):
                metadata_rules.extend(scan_text(value))
        combined = sorted(set(rules + metadata_rules))
        if combined:
            scan.flagged = True
            scan.rules = sorted(set(scan.rules + combined))
            scan.flagged_chunk_ids.append(str(doc.metadata.get("chunk_id", "")))
    return scan


def safe_metadata(metadata: Dict) -> Dict[str, str]:
    """Only expose scalar, bounded metadata to prompts. Never include nested
    objects or unbounded user metadata."""
    allowed = ("doc_id", "document_version", "chunk_id", "source", "page", "section_path", "tenant_id")
    result = {}
    for key in allowed:
        if key in metadata and isinstance(metadata[key], (str, int, float, bool)):
            result[key] = str(metadata[key])[:200]
    return result


def render_evidence_block(doc) -> str:
    md = safe_metadata(doc.metadata or {})
    header = " | ".join(f"{key}={value}" for key, value in md.items() if key in {"doc_id", "document_version", "chunk_id", "page", "section_path"})
    # Neutralise special chat control tokens so evidence cannot spoof roles.
    text = re.sub(r"<\|[^|>]{0,32}\|>", "[neutralized_control_token]", doc.page_content)
    return f'<evidence chunk_id="{md.get("chunk_id", "")}">\n' f"source: {header}\n" f"{text}\n</evidence>"
