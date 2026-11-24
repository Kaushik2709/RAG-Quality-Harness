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




@pytest.mark.parametrize("response", ["not json", "{}", '{"faithfulness": 2}', '{"faithfulness": NaN}'])
def test_invalid_judge_response(response):
    class InvalidModel(DummyLLMChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])
    with pytest.raises(ValueError):
        LLMJudge(chat_model=InvalidModel()).evaluate("query", "context", "answer")


def test_judge_timeout():
    class TimeoutModel(DummyLLMChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise TimeoutError("provider timeout")
    with pytest.raises(TimeoutError):
        LLMJudge(chat_model=TimeoutModel()).evaluate("query", "context", "answer")


def test_judge_records_model_and_prompt_provenance():
    judge = LLMJudge(chat_model=DummyLLMChatModel(), model_id="judge-test-model")
    result = judge.evaluate("query", "context", "answer")
    assert result.judge_model == "judge-test-model"
    assert result.judge_prompt_version == "judge-v1"
    assert result.evaluated_at
    assert result.usage_source in {"provider", "unavailable"}


def test_calibration_requires_human_scores_and_reports_bias():
    from src.evaluation.judge import calibration_report

    judge_scores = [
        {"faithfulness": .9, "answer_relevancy": .8, "context_precision": .7},
        {"faithfulness": .4, "answer_relevancy": .5, "context_precision": .6},
    ]
    human_scores = [
        {"faithfulness": .8, "answer_relevancy": .9, "context_precision": .7},
        {"faithfulness": .6, "answer_relevancy": .5, "context_precision": .4},
    ]
    report = calibration_report(judge_scores, human_scores)
    assert report["status"] == "computed"
    assert report["samples"] == 2
    assert report["faithfulness"]["mae"] == pytest.approx(.15)
    assert report["faithfulness"]["bias_judge_minus_human"] == pytest.approx(-.05)
    assert report["answer_relevancy"]["pearson"] is not None
    assert calibration_report([], [])["status"] == "unavailable"
    with pytest.raises(ValueError):
        calibration_report(judge_scores, human_scores[:1])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_judge_scores_reject_nonfinite_or_out_of_range(score):
    class InvalidScoreModel(DummyLLMChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            payload = {"faithfulness": score, "answer_relevancy": .5, "context_precision": .5,
                       "faithfulness_reasoning": "test", "answer_relevancy_reasoning": "test", "context_precision_reasoning": "test"}
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(payload)))])
    with pytest.raises(ValueError):
        LLMJudge(chat_model=InvalidScoreModel()).evaluate("query", "context", "answer")
