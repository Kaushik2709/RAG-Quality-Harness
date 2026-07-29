import json
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.language_models.chat_models import BaseChatModel
from src.generation.generator import get_chat_model
from src.generation.prompt_builder import build_chat_prompt_template
from src.evaluation.judge_prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT_TEMPLATE

class JudgeResult(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Score 0.0-1.0 measuring factual support in context")
    faithfulness_reasoning: str
    answer_relevancy: float = Field(..., ge=0.0, le=1.0, description="Score 0.0-1.0 measuring query satisfaction")
    answer_relevancy_reasoning: str
    context_precision: float = Field(..., ge=0.0, le=1.0, description="Score 0.0-1.0 measuring context usefulness")
    context_precision_reasoning: str

    @field_validator("faithfulness", "answer_relevancy", "context_precision")
    def clamp_score(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

class LLMJudge:
    def __init__(self, chat_model: Optional[BaseChatModel] = None):
        self.chat_model = chat_model or get_chat_model()
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
            return JudgeResult(**parsed)
        except Exception as e:
            raise ValueError(
                f"[LLMJudge Error] Failed to parse LLM Judge response into valid JudgeResult JSON: {e}\nRaw Response:\n{text}"
            )

