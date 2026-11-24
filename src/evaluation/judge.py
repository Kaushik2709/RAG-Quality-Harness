"""LLM-as-a-judge evaluation, bias detection, and scientific calibration.

Provides structured judgment metrics, calibration against adjudicated human
ground truth, Cohen's Kappa, percent agreement, position bias detection,
verbosity bias detection, and bootstrap 95% confidence intervals.
"""
from __future__ import annotations

import datetime
import json
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
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


# ----------------------------------------------------------------------
# Agreement & Calibration Statistics
# ----------------------------------------------------------------------
def percent_agreement(rater1_binary: Sequence[int], rater2_binary: Sequence[int]) -> float:
    """Computes exact percent agreement between two raters."""
    if len(rater1_binary) != len(rater2_binary) or not rater1_binary:
        return 0.0
    matches = sum(1 for a, b in zip(rater1_binary, rater2_binary) if a == b)
    return round(matches / len(rater1_binary), 4)


def cohens_kappa(rater1_binary: Sequence[int], rater2_binary: Sequence[int]) -> float:
    """Computes Cohen's Kappa for inter-rater reliability on binary decisions."""
    if len(rater1_binary) != len(rater2_binary) or not rater1_binary:
        return 0.0
    n = len(rater1_binary)
    po = sum(1 for a, b in zip(rater1_binary, rater2_binary) if a == b) / n

    # Marginals
    p1_pos = sum(1 for a in rater1_binary if a == 1) / n
    p1_neg = 1.0 - p1_pos
    p2_pos = sum(1 for b in rater2_binary if b == 1) / n
    p2_neg = 1.0 - p2_pos

    pe = (p1_pos * p2_pos) + (p1_neg * p2_neg)
    if pe == 1.0:
        return 1.0
    kappa = (po - pe) / (1.0 - pe)
    return round(kappa, 4)


def detect_position_bias(
    judge_scores_pos_a: Sequence[float],
    judge_scores_pos_b: Sequence[float],
) -> Dict[str, float]:
    """Detects whether evidence position influences judge ratings (Lost-in-the-Middle effect)."""
    if not judge_scores_pos_a or not judge_scores_pos_b:
        return {"position_bias_delta": 0.0, "mean_pos_a": 0.0, "mean_pos_b": 0.0}
    mean_a = _mean(judge_scores_pos_a)
    mean_b = _mean(judge_scores_pos_b)
    delta = round(mean_a - mean_b, 4)
    return {
        "mean_pos_a": round(mean_a, 4),
        "mean_pos_b": round(mean_b, 4),
        "position_bias_delta": delta,
    }


def detect_verbosity_bias(
    judge_scores: Sequence[float],
    answer_lengths: Sequence[int],
) -> Dict[str, Any]:
    """Measures correlation between response length (verbosity) and judge ratings."""
    if len(judge_scores) != len(answer_lengths) or len(judge_scores) < 2:
        return {"correlation": 0.0, "status": "insufficient_samples"}
    f_lengths = [float(x) for x in answer_lengths]
    corr = _pearson(judge_scores, f_lengths)
    return {
        "correlation": corr or 0.0,
        "status": "computed",
        "has_verbosity_bias": abs(corr or 0.0) > 0.35,
    }


def bootstrap_ci(
    values: Sequence[float],
    samples: int = 1000,
    ci: float = 0.95,
    seed: int = 12345,
) -> Dict[str, float]:
    """Calculates non-parametric bootstrap confidence interval."""
    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        resample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lower_idx = int(((1.0 - ci) / 2.0) * samples)
    upper_idx = int((1.0 - (1.0 - ci) / 2.0) * samples)
    return {
        "mean": round(_mean(values), 4),
        "ci_lower": round(means[max(0, lower_idx)], 4),
        "ci_upper": round(means[min(samples - 1, upper_idx)], 4),
    }


def calibration_report(
    judge_scores: Sequence[Dict[str, float]],
    human_scores: Sequence[Dict[str, float]],
    dimensions: Sequence[str] = ("faithfulness", "answer_relevancy", "context_precision"),
    answer_lengths: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Compare judge and human scores where both are available.

    Computes MAE, RMSE, signed bias, Pearson r, Cohen's Kappa, percent agreement,
    and bootstrap 95% confidence intervals.
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

        # Binary binarization (>= 0.70 is positive decision)
        j_bin = [1 if v >= 0.70 else 0 for v in judge_values]
        h_bin = [1 if v >= 0.70 else 0 for v in human_values]

        ci_res = bootstrap_ci(judge_values, samples=500)

        report[dimension] = {
            "status": "computed",
            "n": len(pairs),
            "judge_mean": round(_mean(judge_values), 4),
            "human_mean": round(_mean(human_values), 4),
            "mae": round(_mean([abs(error) for error in errors]), 4),
            "rmse": round(math.sqrt(_mean([error * error for error in errors])), 4),
            "bias_judge_minus_human": round(_mean(errors), 4),
            "pearson": _pearson(judge_values, human_values),
            "cohens_kappa": cohens_kappa(j_bin, h_bin),
            "percent_agreement": percent_agreement(j_bin, h_bin),
            "bootstrap_95_ci": ci_res,
        }

    if answer_lengths and len(answer_lengths) == len(judge_scores):
        # Measure verbosity bias against faithfulness
        j_faith = [float(j.get("faithfulness", 0.0)) for j in judge_scores]
        report["verbosity_bias"] = detect_verbosity_bias(j_faith, answer_lengths)

    report["status"] = "computed"
    return report


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2:
        return None
    left_mean, right_mean = _mean(left), _mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return round(numerator / denominator, 6) if denominator else None
