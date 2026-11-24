import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class GoldenTriple(BaseModel):
    query_id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="User query text")
    expected_relevant_doc_ids: List[str] = Field(
        default_factory=list,
        description="Relevant document IDs. May be empty only when answerable=false.",
    )
    reference_answer: str = Field(..., description="Ground truth reference answer")
    answerable: bool = Field(default=True, description="False for deliberate no-answer cases")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata (category, difficulty, source)")

    @field_validator("query", "reference_answer")
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query and reference_answer cannot be empty.")
        return v

    @field_validator("expected_relevant_doc_ids")
    def doc_ids_unique_and_nonblank(cls, v: List[str]) -> List[str]:
        if any(not x.strip() for x in v) or len(set(v)) != len(v):
            raise ValueError("expected_relevant_doc_ids must be unique and nonblank")
        return v

    @model_validator(mode="after")
    def answerable_labels_are_consistent(self):
        if self.answerable and not self.expected_relevant_doc_ids:
            raise ValueError("answerable queries require at least one relevant document ID")
        if not self.answerable and self.expected_relevant_doc_ids:
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
