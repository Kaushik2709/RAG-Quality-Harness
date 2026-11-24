import datetime
import json
import math
from typing import Dict, Optional, Sequence
from pydantic import BaseModel, Field, field_validator
from langchain_core.language_models.chat_models import BaseChatModel
from src.generation.generator import get_chat_model
from src.generation.prompt_builder import build_chat_prompt_template
from src.evaluation.judge_prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT_TEMPLATE

JUDGE_PROMPT_VERSION = "judge-v1"


class JudgeResult(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Score 0.0-1.0 measuring factual support in context")
    faithfulness_reasoning: str
    answer_relevancy: float = Field(..., ge=0.0, le=1.0, description="Score 0.0-1.0 measuring query satisfaction")
    answer_relevancy_reasoning: str
    context_precision: float = Field(..., ge=0.0, le=1.0, description="Score 0.0-1.0 measuring context usefulness")
    context_precision_reasoning: str
    judge_model: Optional[str] = None
    judge_prompt_version: str = JUDGE_PROMPT_VERSION
    evaluated_at: Optional[str] = None
    usage_source: str = "unavailable"

    @field_validator("faithfulness", "answer_relevancy", "context_precision")
    def clamp_score(cls, v: float) -> float:
        value = float(v)
        if not math.isfinite(value):
            raise ValueError("judge scores must be finite")
        return max(0.0, min(1.0, value))


class LLMJudge:
    def __init__(self, chat_model: Optional[BaseChatModel] = None, model_id: Optional[str] = None):
        self.chat_model = chat_model or get_chat_model()
        self.model_id = model_id or getattr(self.chat_model, "model_name", type(self.chat_model).__name__)
        self.last_usage_metadata = None
        self.prompt_template = build_chat_prompt_template(
            system_template=JUDGE_SYSTEM_PROMPT,
            human_template=JUDGE_USER_PROMPT_TEMPLATE
        )

    def evaluate(self, query: str, retrieved_context: str, generated_answer: str) -> JudgeResult:
        import time
        chain = self.prompt_template | self.chat_model

        max_retries = 4
        ai_message = None
        for attempt in range(max_retries):
            try:
                ai_message = chain.invoke({
                    "query": query,
                    "retrieved_context": retrieved_context or "None",
                    "generated_answer": generated_answer or "None"
                })
                break
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        sleep_time = (attempt + 1) * 2
                        print(f"[LLMJudge Notice] Google API busy/503. Retrying in {sleep_time}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(sleep_time)
                        continue
                raise e

        self.last_usage_metadata = getattr(ai_message, "usage_metadata", None)
        text = str(ai_message.content).strip() if ai_message else ""

        # Clean JSON codeblock wrappers if present
        if "```" in text:
            import re
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1).strip()
            else:
                lines = text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
            parsed.update(
                {
                    "judge_model": self.model_id,
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "usage_source": "provider" if self.last_usage_metadata else "unavailable",
                }
            )
            return JudgeResult(**parsed)
        except Exception as e:
            raise ValueError(
                f"[LLMJudge Error] Failed to parse LLM Judge response into valid JudgeResult JSON: {e}\nRaw Response:\n{text}"
            )


def calibration_report(
    judge_scores: Sequence[Dict[str, float]],
    human_scores: Sequence[Dict[str, float]],
    dimensions: Sequence[str] = ("faithfulness", "answer_relevancy", "context_precision"),
) -> Dict[str, object]:
    """Compare judge and human scores where both are available.

    A missing human score is reported as unavailable, never imputed. MAE/RMSE
    and signed bias are descriptive; correlation needs variance and enough
    samples to mean anything.
    """
    if len(judge_scores) != len(human_scores):
        raise ValueError("judge and human score lists must have equal length")
    report: Dict[str, object] = {"samples": len(judge_scores)}
    if not judge_scores:
        return {**report, "status": "unavailable", "reason": "no paired scores"}
    for dimension in dimensions:
        pairs = [
            (float(judge[dimension]), float(human[dimension]))
            for judge, human in zip(judge_scores, human_scores)
            if dimension in judge and dimension in human
        ]
        if not pairs:
            report[dimension] = {"status": "unavailable", "reason": "dimension missing"}
            continue
        errors = [judge - human for judge, human in pairs]
        judge_values = [pair[0] for pair in pairs]
        human_values = [pair[1] for pair in pairs]
        report[dimension] = {
            "status": "computed",
            "n": len(pairs),
            "judge_mean": _mean(judge_values),
            "human_mean": _mean(human_values),
            "mae": _mean([abs(error) for error in errors]),
            "rmse": math.sqrt(_mean([error * error for error in errors])),
            "bias_judge_minus_human": _mean(errors),
            "pearson": _pearson(judge_values, human_values),
        }
    report["status"] = "computed"
    return report


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2:
        return None
    left_mean, right_mean = _mean(left), _mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return round(numerator / denominator, 6) if denominator else None

