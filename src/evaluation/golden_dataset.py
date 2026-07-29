import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class GoldenTriple(BaseModel):
    query_id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="User query text")
    expected_relevant_doc_ids: List[str] = Field(..., description="List of expected document IDs or chunk IDs that contain the answer")
    reference_answer: str = Field(..., description="Ground truth reference answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata (category, difficulty, source)")

    @field_validator("query", "reference_answer")
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query and reference_answer cannot be empty.")
        return v

    @field_validator("expected_relevant_doc_ids")
    def doc_ids_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("expected_relevant_doc_ids must contain at least one document ID.")
        return v

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
    return triples

def save_golden_dataset(triples: List[GoldenTriple], file_path: str = "eval/golden_dataset.jsonl") -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for triple in triples:
            f.write(json.dumps(triple.model_dump()) + "\n")
