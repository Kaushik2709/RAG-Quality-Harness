"""Golden triple dataset model and serialization."""
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class GoldenTriple(BaseModel):
    query_id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="User query text")
    expected_relevant_doc_ids: List[str] = Field(
        default_factory=list,
        description="Relevant document IDs. May be empty only when answerable=false.",
    )
    expected_relevant_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Relevant chunk IDs if chunk-level annotation is available.",
    )
    reference_answer: Optional[str] = Field(
        default=None, description="Ground truth reference answer (null for unanswerable)"
    )
    expected_abstention_reason: Optional[str] = Field(
        default=None, description="Reason for expected abstention when answerable=false"
    )
    answerable: bool = Field(default=True, description="False for deliberate no-answer cases")
    category: Optional[str] = Field(default=None)
    annotation_status: Optional[str] = Field(default=None)
    split: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata (category, difficulty, source, cluster_id)"
    )

    @field_validator("query")
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty.")
        return v

    @field_validator("expected_relevant_doc_ids", "expected_relevant_chunk_ids")
    def doc_ids_unique_and_nonblank(cls, v: List[str]) -> List[str]:
        if any(not x.strip() for x in v) or len(set(v)) != len(v):
            raise ValueError("expected_relevant IDs must be unique and nonblank")
        return v

    @model_validator(mode="after")
    def harmonize_fields_and_validate(self):
        # Sync top-level and metadata fields
        if self.split and "split" not in self.metadata:
            self.metadata["split"] = self.split
        elif "split" in self.metadata and not self.split:
            self.split = self.metadata["split"]

        if self.annotation_status and "annotation_status" not in self.metadata:
            self.metadata["annotation_status"] = self.annotation_status
        elif "annotation_status" in self.metadata and not self.annotation_status:
            self.annotation_status = self.metadata["annotation_status"]

        if self.category and "category" not in self.metadata:
            self.metadata["category"] = self.category
        elif "category" in self.metadata and not self.category:
            self.category = self.metadata["category"]

        if self.answerable:
            if not self.expected_relevant_doc_ids:
                raise ValueError("answerable queries require at least one relevant document ID")
            if self.reference_answer is not None and not self.reference_answer.strip():
                raise ValueError("reference_answer cannot be blank for answerable queries")
        else:
            if self.expected_relevant_doc_ids:
                raise ValueError("unanswerable queries must not declare relevant document IDs")
        return self


def load_golden_dataset(file_path: str = "eval/golden_dataset.jsonl") -> List[GoldenTriple]:
    triples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                triples.append(GoldenTriple(**data))
            except Exception as e:
                raise ValueError(f"Error parsing line {line_num} in golden dataset ({file_path}): {e}")
    if not triples or len({t.query_id for t in triples}) != len(triples):
        raise ValueError("Dataset must be nonempty with unique query IDs")
    return triples


def save_golden_dataset(triples: List[GoldenTriple], file_path: str = "eval/golden_dataset.jsonl") -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for triple in triples:
            f.write(json.dumps(triple.model_dump()) + "\n")
