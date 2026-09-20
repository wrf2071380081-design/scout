"""评测与自进化层测试：裁判去偏、数据飞轮、参数建议（全部离线）。"""

from __future__ import annotations

from scout.evolution import (
    FailureMiner,
    FailureReason,
    FlywheelLedger,
    JudgeCalibration,
    LLMJudge,
    ParamTuner,
    ReviewQueue,
    candidate_id,
    export_preference_pairs,
)
from scout.llm.base import LLMResponse, TokenUsage
from scout.llm.scripted import ScriptedLLM

# —— 裁判 ——


def _scripted(outputs: list[str]) -> ScriptedLLM:
    """把预设输出串成客户端：每个输出对应一次 complete。"""

    return ScriptedLLM(outputs=outputs) if _accepts_outputs() else _queue_llm(outputs)


def _accepts_outputs() -> bool:
    import inspect

    return "outputs" in inspect.signature(ScriptedLLM.__init__).parameters


def _queue_llm(outputs: list[str]):
    """兜底：用一个按序吐出的极简客户端。"""

    class Queued:
        def __init__(self, items: list[str]) -> None:
            self.items = list(items)
            self.calls = 0

        @property
        def model_name(self) -> str:
            return "queued"

        def complete(self, request) -> LLMResponse:
            self.calls += 1
            content = self.items.pop(0) if self.items else ""
            return LLMResponse(content=content, usage=TokenUsage(input_tokens=8, output_tokens=4))

    return Queued(outputs)


RUBRIC_JSON = (
    '{"faithfulness": {"score": 5, "reason": "答案与材料一致"},'
    ' "relevance": {"score": 4, "reason": "直接回应了问题"},'
    ' "conciseness": {"score": 3, "reason": "略有冗余"},'
    ' "verdict": "整体可靠"}'
)


def test_judge_scores_with_rubric() -> None:
    judge = LLMJudge(_scripted([RUBRIC_JSON]))
    verdict = judge.score("云计算标准体系包括哪几部分？", "包括基础、技术等六部分。", "材料：六部分。")
    assert verdict.scores["faithfulness"] == 5
    assert 0 < verdict.overall <= 5
    assert verdict.rationale == "整体可靠"
    assert not verdict.parse_failed


def test_judge_records_model_for_cross_source_check() -> None:
    """裁判模型名要落盘——跨模型对比时必须确认不是同源自评。"""

    judge = LLMJudge(_scripted([RUBRIC_JSON]))
    verdict = judge.score("q", "a", "e")
    assert verdict.judge_model, "缺了裁判模型名就没法判断自我偏好风险"
    assert verdict.to_dict()["judge_model"] == verdict.judge_model


def test_judge_failure_is_visible_not_silent_zero() -> None:
    judge = LLMJudge(_scripted(["这不是 JSON"]))
    verdict = judge.score("q", "a", "e")
    assert verdict.parse_failed is True
    assert verdict.rationale, "评分失败必须留痕，不能悄悄当成 0 分混进评测结论"


def test_pairwise_runs_twice_and_detects_inconsistency() -> None:
    """位置偏见去偏：两次换序结论不一致 → 判平局，而不是采信其中一次。"""

    first = '{"winner": "A", "reason": "更忠实"}'
    second = '{"winner": "A", "reason": "更忠实"}'
    judge = LLMJudge(_scripted([first, second]), use_pairwise_debias=True)
    result = judge.compare("q", "答案甲", "答案乙", "材料")
    assert result["passes"] == 2
    assert result["swapped"] is True
    # 第二次把顺序换过来之后仍然选 A，说明它其实选的是"语义上的乙" → 结论不一致
    assert result["inconsistent"] is True
    assert result["winner"] == "tie"


def test_pairwise_consistent_result_wins() -> None:
    judge = LLMJudge(_scripted(['{"winner": "A", "reason": "好"}', '{"winner": "B", "reason": "好"}']))
    result = judge.compare("q", "甲", "乙", "材料")
    assert result["winner"] == "A", "换序后选 B 恰好印证了原位的 A"


def test_calibration_gates_trust() -> None:
    calibration = JudgeCalibration()
    for _ in range(9):
        calibration.add(4.0, 4.0)
    assert not calibration.trustable(), "样本不足 10 条时裁判分数不可作为决策依据"
    calibration.add(4.5, 4.0)
    assert calibration.trustable()
    assert calibration.to_dict()["agreement_at_1"] == 1.0


def test_calibration_reports_bias() -> None:
    calibration = JudgeCalibration()
    for _ in range(10):
        calibration.add(5.0, 3.0)
    assert calibration.mean_bias == 2.0
    assert not calibration.trustable(), "系统性高估 2 分时不能采信"


# —— 数据飞轮 ——


def test_miner_classifies_failures() -> None:
    miner = FailureMiner()
    found = miner.mine(
        [
            {"question": "该答没答的问题", "outcome": "insufficient_evidence", "retrieved": 5},
            {"question": "支撑率低的问题", "outcome": "answered", "retrieved": 5, "support_rate": 0.2},
            {"question": "重生成过的问题", "outcome": "answered", "retrieved": 4, "regenerations": 1},
            {"question": "检索为空的问题", "outcome": "no_knowledge", "retrieved": 0},
            {"question": "答得很好的问题", "outcome": "answered", "retrieved": 6, "support_rate": 0.9},
        ]
    )
    reasons = {case.reason for case in found}
    assert FailureReason.ABSTAINED in reasons
    assert FailureReason.LOW_SUPPORT in reasons
    assert FailureReason.REGENERATED in reasons
    assert len(found) == 3, "检索为空属于数据覆盖问题，默认不进'系统能力'评测集"


def test_miner_dedups_same_question() -> None:
    """同一道难题每天失败一次，一个月后不该在评测集里出现三十次。"""

    miner = FailureMiner()
    observations = [
        {"question": "同一个反复失败的问题", "outcome": "insufficient_evidence", "retrieved": 3}
    ] * 5
    found = miner.mine(observations)
    assert len(found) == 1
    assert candidate_id("同一个反复失败的问题") == found[0].case_id


def test_review_queue_requires_explicit_approval() -> None:
    miner = FailureMiner()
    cases = miner.mine([{"question": "候选问题", "outcome": "insufficient_evidence", "retrieved": 3}])
    queue = ReviewQueue()
    assert queue.enqueue(cases) == 1
    assert not queue.approved, "自动挖掘不得直接入库——模型判错里混着'本来就不该答'的题"
    queue.approve(cases[0].case_id)
    assert len(queue.approved) == 1


def test_rejection_requires_reason() -> None:
    miner = FailureMiner()
    cases = miner.mine([{"question": "候选问题", "outcome": "insufficient_evidence", "retrieved": 3}])
    queue = ReviewQueue()
    queue.enqueue(cases)
    queue.reject(cases[0].case_id, "库里确实没有这条内容")
    assert queue.rejected[0][1] == "库里确实没有这条内容"
    assert "库里确实没有这条内容" in queue.to_dict()["rejection_reasons"]


def test_ledger_detects_stalled_flywheel() -> None:
    ledger = FlywheelLedger()
    queue = ReviewQueue()
    for _ in range(3):
        ledger.record(mined=0, queue=queue)
    assert ledger.stalled(), "连续三轮无新增 → 该修挖掘规则，而不是继续加数据"


def test_export_preference_pairs_is_portable() -> None:
    pairs = export_preference_pairs([("问题", "好答案", "差答案"), ("问题2", "同", "同")])
    assert len(pairs) == 1, "相同答案不构成偏好对"
    assert set(pairs[0]) == {"prompt", "chosen", "rejected"}


# —— 参数自调 ——


def test_tuner_flags_false_refusal() -> None:
    tuner = ParamTuner()
    suggestions = tuner.suggest(
        {"n_cases": 50, "false_refusal_rate": 0.2, "abstain_rate": 0.3},
        current={"sufficiency_min_coverage": 0.5},
    )
    target = [item for item in suggestions if item.param == "sufficiency_min_coverage"]
    assert target, "误拒率高时必须给出下调门槛的建议"
    assert target[0].proposed < 0.5
    assert not target[0].auto_applicable, "50 条样本不允许自动改参数"


def test_tuner_never_auto_applies_on_small_samples() -> None:
    tuner = ParamTuner(min_samples_for_auto=200)
    suggestions = tuner.suggest(
        {"n_cases": 30, "hallucination_rate": 0.25},
        current={"sufficiency_min_coverage": 0.5},
    )
    assert suggestions and all(not item.auto_applicable for item in suggestions)


def test_tuner_ignores_noise_band() -> None:
    tuner = ParamTuner(noise_band=3.0)
    suggestions = tuner.suggest({"n_cases": 500, "false_refusal_rate": 0.01})
    assert not suggestions, "1pp 的差异落在噪声带内，不该建议改动"


def test_tuner_says_do_not_tune_when_knowledge_is_missing() -> None:
    """空检索率高时明确建议"不要调参"——把数据问题误当模型问题是常见浪费。"""

    tuner = ParamTuner()
    suggestions = tuner.suggest({"n_cases": 300, "empty_retrieval_rate": 0.4})
    assert suggestions
    assert suggestions[0].param == "(不调参)"
    assert not suggestions[0].auto_applicable


def test_tuner_marks_structural_problem() -> None:
    tuner = ParamTuner()
    suggestions = tuner.suggest({"n_cases": 300, "cross_doc_recall": 0.2})
    assert any(item.param == "(结构改动)" for item in suggestions)


def test_tuner_renders_markdown() -> None:
    tuner = ParamTuner()
    rendered = tuner.render(tuner.suggest({"n_cases": 300, "empty_retrieval_rate": 0.4}))
    assert "| 参数 |" in rendered
    assert "不调参" in rendered
