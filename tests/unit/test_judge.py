import json
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from src.evaluation.judge import LLMJudge, JudgeResult

class DummyLLMChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        mock_json = json.dumps({
            "faithfulness": 0.95,
            "faithfulness_reasoning": "Answer is fully supported by the retrieved context.",
            "answer_relevancy": 0.90,
            "answer_relevancy_reasoning": "Answer directly answers the question.",
            "context_precision": 0.85,
            "context_precision_reasoning": "Retrieved context is highly relevant."
        })
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=mock_json))])

    @property
    def _llm_type(self) -> str:
        return "dummy_chat_model"

def test_judge_structured_output():
    test_model = DummyLLMChatModel()
    judge = LLMJudge(chat_model=test_model)

    res = judge.evaluate(
        query="What is vector search in Qdrant?",
        retrieved_context="Qdrant provides high performance vector search using HNSW indexing.",
        generated_answer="Vector search in Qdrant uses HNSW indexing for fast approximate nearest neighbor search."
    )

    assert isinstance(res, JudgeResult)
    assert 0.0 <= res.faithfulness <= 1.0
    assert 0.0 <= res.answer_relevancy <= 1.0
    assert 0.0 <= res.context_precision <= 1.0
    assert len(res.faithfulness_reasoning) > 0


