"""Token-aware, deduplicated context assembly."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from src.config import settings
from src.security.document_guardrails import UntrustedScan, render_evidence_block, scan_documents


@dataclass
class ContextBundle:
    chunks: List[Document]
    text: str
    estimated_tokens: int
    tokenizer: str
    token_counts_estimated: bool
    dropped_chunks: int
    truncated: bool
    untrusted_scan: UntrustedScan = field(default_factory=UntrustedScan)
    budget_tokens: int = 0
    utilization: float = 0.0

    def metadata(self) -> Dict[str, object]:
        return {
            "included_chunks": len(self.chunks),
            "estimated_tokens": self.estimated_tokens,
            "tokenizer": self.tokenizer,
            "token_counts_estimated": self.token_counts_estimated,
            "dropped_chunks": self.dropped_chunks,
            "truncated": self.truncated,
            "budget_tokens": self.budget_tokens,
            "utilization": round(self.utilization, 4),
            "untrusted_evidence": self.untrusted_scan.to_dict(),
        }


def _estimate_tokens(text: str) -> int:
    # Conservative approximation used only when no tokenizer is installed.
    return max(1, (len(text) + 3) // 4)


def count_tokens(text: str, tokenizer: Optional[Callable[[str], int]] = None) -> tuple[int, bool]:
    if tokenizer is not None:
        return int(tokenizer(text)), False
    return _estimate_tokens(text), True


def build_context(
    documents: Sequence[Document],
    *,
    max_tokens: Optional[int] = None,
    max_chunks: Optional[int] = None,
    tokenizer: Optional[Callable[[str], int]] = None,
    diversify: Optional[bool] = None,
) -> ContextBundle:
    budget = max_tokens if max_tokens is not None else settings.MAX_CONTEXT_TOKENS
    limit = max_chunks if max_chunks is not None else settings.CONTEXT_MAX_CHUNKS
    diversify = settings.CONTEXT_DIVERSIFY_SOURCES if diversify is None else diversify
    if budget <= 0 or limit <= 0:
        raise ValueError("context budget and chunk limit must be positive")

    # 1. Remove exact duplicate content, retaining the first (highest-ranked) copy.
    seen_hashes = set()
    deduped: List[Document] = []
    dropped = 0
    for doc in documents:
        digest = hashlib.sha256(doc.page_content.strip().encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            dropped += 1
            continue
        seen_hashes.add(digest)
        deduped.append(doc)

    # 2. Optionally spread the budget across sources instead of exhausting it on one document.
    if diversify and len({d.metadata.get("doc_id") for d in deduped}) > 1:
        groups: Dict[str, List[Document]] = {}
        order: List[str] = []
        for doc in deduped:
            key = str(doc.metadata.get("doc_id", ""))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(doc)
        interleaved: List[Document] = []
        cursor = 0
        while True:
            added = False
            for key in order:
                if cursor < len(groups[key]):
                    interleaved.append(groups[key][cursor])
                    added = True
            if not added:
                break
            cursor += 1
        deduped = interleaved

    # 3. Enforce the token budget using the configured counter.
    selected: List[Document] = []
    blocks: List[str] = []
    used = 0
    truncated = False
    for doc in deduped:
        if len(selected) >= limit:
            dropped += 1
            continue
        block = render_evidence_block(doc)
        tokens, _ = count_tokens(block, tokenizer)
        if used + tokens > budget:
            remaining = budget - used
            # The rendered header (chunk id, source) is part of the budget. Only
            # keep a prefix when at least a minimal body fits alongside it.
            header = Document(page_content="", metadata=dict(doc.metadata))
            header_tokens, _ = count_tokens(render_evidence_block(header), tokenizer)
            available = remaining - header_tokens
            chars = max(0, available * 4)
            while chars >= 20:
                prefix_doc = Document(page_content=doc.page_content[:chars], metadata=dict(doc.metadata))
                prefix_block = render_evidence_block(prefix_doc)
                prefix_tokens, _ = count_tokens(prefix_block, tokenizer)
                if used + prefix_tokens <= budget:
                    selected.append(prefix_doc)
                    blocks.append(prefix_block)
                    used += prefix_tokens
                    truncated = True
                    break
                chars = int(chars * 0.8)
            dropped += 1
            continue
        selected.append(doc)
        blocks.append(block)
        used += tokens

    text = "\n\n".join(blocks)
    total_estimated, estimated = count_tokens(text, tokenizer) if text else (0, tokenizer is None)
    scan = scan_documents(selected)
    return ContextBundle(
        chunks=selected,
        text=text,
        estimated_tokens=total_estimated,
        tokenizer="provider_tokenizer" if tokenizer is not None else "estimate:chars/4",
        token_counts_estimated=estimated,
        dropped_chunks=dropped,
        truncated=truncated,
        untrusted_scan=scan,
        budget_tokens=budget,
        utilization=(total_estimated / budget) if budget else 0.0,
    )
