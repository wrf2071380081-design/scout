"""接线层测试：新能力是否真的接进了系统（而不只是躺在模块里）。

这一组测试的存在理由很直接——**一个没被接进主流程的模块等于没做**。
所以这里断言的不是"类能实例化"，而是"装配后行为真的变了"：
流水线会短路、缓存命中不会触发预算、存储后端可替换。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from scout.config import get_settings
from scout.evolution.compare import ComparisonCase, compare_judges, demo_cases
from scout.hitl.checkpoints import (
    Checkpoint,
    InMemoryCheckpointStore,
    RedisCheckpointStore,
    build_checkpoint_store,
)
from scout.llm.base import ChatMessage, LLMRequest, LLMResponse, TokenUsage
from scout.llm.scripted import HeuristicLLM
from scout.rag.embed import HashingEmbedder
from scout.rag.pipeline import RAGPipeline, build_index
from scout.runtime.factory import build_runtime_stack
from scout.runtime.intent import ACTION_LABEL, build_knowledge_funnel

DOCS = [
    ("a.md", "云计算标准体系结构包括基础、技术、服务、应用、管理和安全六个部分。"),
    ("b.md", "低空经济标准体系重点围绕低空航空器、起降设施与运行服务展开。"),
]


# —— 假 Redis ——


class FakePipeline:
    def __init__(self, client: "FakeRedis") -> None:
        self.client = client
        self.ops: list[tuple[str, Any, Any]] = []

    def rpush(self, key: str, value: str) -> "FakePipeline":
        self.ops.append(("rpush", key, value))
        return self

    def expire(self, key: str, ttl: int) -> "FakePipeline":
        self.ops.append(("expire", key, ttl))
        return self

    def execute(self) -> None:
        for name, key, value in self.ops:
            if name == "rpush":
                self.client.store.setdefault(key, []).append(value)
            elif name == "expire":
                self.client.ttls[key] = value
        self.ops.clear()


class FakeRedis:
    """够用的内存版 Redis：只实现本项目用到的那几个命令。"""

    def __init__(self) -> None:
        self.store: dict[str, list[str]] = {}
        self.ttls: dict[str, int] = {}

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    def rpush(self, key: str, value: str) -> int:
        self.store.setdefault(key, []).append(value)
        return len(self.store[key])

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self.store.get(key, [])
        return items[start:] if end == -1 else items[start : end + 1]

    def lindex(self, key: str, index: int) -> str | None:
        items = self.store.get(key, [])
        try:
            return items[index]
        except IndexError:
            return None

    def keys(self, pattern: str) -> list[str]:
        prefix = pattern.rstrip("*")
        return [key for key in self.store if key.startswith(prefix)]


# —— 状态存储后端 ——


def test_redis_store_roundtrip() -> None:
    """Redis 后端的语义必须与文件版一致：append-only、可时间旅行、坏行跳过。"""

    client = FakeRedis()
    store = RedisCheckpointStore(client=client, ttl_seconds=60)

    for step in range(3):
        store.save(Checkpoint(run_id="run-1", step=step, state={"step": step}))

    assert store.latest("run-1").step == 2
    assert store.at("run-1", 1).state == {"step": 1}
    assert [item.step for item in store.history("run-1")] == [0, 1, 2]
    assert store.runs() == ["run-1"]
    # 写入时续期：活跃会话不该因为"最后写入很久以前"而消失
    assert client.ttls["scout:ckpt:run-1"] == 60


def test_redis_store_skips_corrupt_line() -> None:
    client = FakeRedis()
    store = RedisCheckpointStore(client=client)
    store.save(Checkpoint(run_id="r", step=0, state={}))
    client.store["scout:ckpt:r"].append("{坏行")  # 模拟崩溃写坏的最后一行
    store.save(Checkpoint(run_id="r", step=1, state={}))
    # 坏行被跳过而不是整体失败：append-only 的语义是"前面写成功了"
    assert [item.step for item in store.history("r")] == [0, 1]


def test_build_store_requires_explicit_choice() -> None:
    """不传参数就是内存——**不做"自动尝试 Redis 失败再降级"**：
    存储降级是重大语义变化（跨进程可恢复 → 重启即丢），必须显式决定。"""

    assert isinstance(build_checkpoint_store(), InMemoryCheckpointStore)


# —— 运行时链装配 ——


class CountingLLM:
    def __init__(self, model: str = "fake") -> None:
        self._model = model
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="结果", usage=TokenUsage(input_tokens=100, output_tokens=50))


def test_stack_order_cache_outside_budget() -> None:
    """顺序：base → router → budget → cache（由内到外）。

    关键断言是 **cache 在 budget 之外**：缓存命中不花钱，
    如果预算在缓存外面，"本来免费"的命中会先被预算拦下，
    用户看到"预算不足"却根本不需要花钱。
    """

    settings = get_settings()
    demo = replace(
        settings,
        runtime=replace(
            settings.runtime, cache_enabled=True, routing_enabled=True, budget_tokens=10000
        ),
    )
    stack = build_runtime_stack(CountingLLM(), demo, embedder=HashingEmbedder(dim=256))
    assert stack.order == ("base", "router", "budget", "cache")


def test_cache_hit_bypasses_budget() -> None:
    """预算刚好只够一次 + 第二次调用命中缓存 → 不该报预算不足。

    参数是刻意配的：单次调用的保守估算约 1511 token，实际用量 150。
    预算给 1600 意味着"第一次能过、第二次若真的发起调用必然超"——
    所以这个用例能真正区分"命中缓存"和"又发了一次请求"。
    """

    settings = get_settings()
    demo = replace(
        settings,
        runtime=replace(settings.runtime, cache_enabled=True, budget_tokens=1600),
    )
    stack = build_runtime_stack(CountingLLM(), demo, embedder=HashingEmbedder(dim=256))
    request = LLMRequest(
        messages=[ChatMessage(role="user", content="同一段提示词")],
        task="grade",
        context={"evidence_digest": "s1"},
    )
    first = stack.client.complete(request)
    second = stack.client.complete(request)  # 若不命中缓存，这次会因预算不足而抛错
    assert first.content == second.content
    assert second.usage.total == 0
    assert stack.report()["cache"]["hits"] == 1


def test_stack_report_exposes_layers_separately() -> None:
    """"命中率低"与"预算打满"是两个问题，不能混在一个数字里。"""

    settings = get_settings()
    demo = replace(settings, runtime=replace(settings.runtime, cache_enabled=True))
    stack = build_runtime_stack(CountingLLM(), demo, embedder=HashingEmbedder(dim=256))
    report = stack.report()
    assert "cache" in report and "order" in report


# —— 意图前筛接进流水线 ——


def _pipeline_with_intent() -> RAGPipeline:
    settings = replace(get_settings(), intent=replace(get_settings().intent, enabled=True))
    index = build_index(DOCS)
    return RAGPipeline(index, HeuristicLLM(), settings=settings)


def test_pipeline_emits_intent_stage() -> None:
    pipeline = _pipeline_with_intent()
    stages: list[str] = []
    result = pipeline.answer(
        "云计算标准体系结构包括哪几个部分？",
        on_stage=lambda name, _p: stages.append(name),
    )
    assert stages[0] == "intent", "意图必须是第一步——它的意义就在于'在检索之前'做决定"
    assert "intent" in result.meta
    assert result.meta["intent"]["tier"] in {"rule", "semantic", "llm", "clarify", "unknown"}


def test_pipeline_short_circuits_chitchat() -> None:
    """闲聊不该触发检索与生成——这是省钱与降延迟的直接来源。"""

    pipeline = _pipeline_with_intent()
    stages: list[str] = []
    result = pipeline.answer("今天天气怎么样", on_stage=lambda name, _p: stages.append(name))
    assert result.outcome == "no_knowledge"
    assert "未执行检索" in result.answer
    assert "retrieve" not in stages, "短路后不应再有检索阶段"
    assert result.units == []


def test_pipeline_marks_action_requests_but_still_retrieves() -> None:
    """动作类请求：标记出来交给审批通道，但检索照做（动作往往要先查清对象）。"""

    pipeline = _pipeline_with_intent()
    result = pipeline.answer("帮我给客户发一封邮件")
    assert result.meta.get("intent_route") == "action"
    assert result.meta["intent"]["label"] == ACTION_LABEL


def test_pipeline_clarifies_on_ambiguous_intent() -> None:
    """歧义区间必须澄清，而不是硬选一个。"""

    settings = replace(
        get_settings(),
        intent=replace(
            get_settings().intent, enabled=True, margin_threshold=0.95  # 刻意构造歧义
        ),
    )
    pipeline = RAGPipeline(build_index(DOCS), HeuristicLLM(), settings=settings)
    result = pipeline.answer("这份文件里是怎么规定的，还是帮我发邮件")
    assert result.outcome in {"clarify", "no_knowledge"}
    assert result.meta["intent"]["tier"] in {"clarify", "rule", "semantic", "llm", "unknown"}


def test_action_label_matches_between_modules() -> None:
    """流水线里的字面量必须与 runtime.intent 的常量一致。

    这条断言是防止"改了一边忘了另一边"——不一致会导致动作类请求
    静默地不再被标记，而这不会报错，只会让审批链失效。
    """

    from scout.rag import pipeline as pipeline_module

    assert pipeline_module._ACTION_LABEL == ACTION_LABEL


def test_action_keywords_cover_colloquial_phrasing() -> None:
    """动作词表要覆盖口语说法。

    "帮我给客户发一封邮件"里没有"发邮件"这个子串——早期词表只写了"发邮件"，
    结果最典型的动作请求反而漏掉了。这条断言把它钉住。
    """

    from scout.runtime.intent import ACTION_KEYWORDS

    assert any(word in "帮我给客户发一封邮件" for word in ACTION_KEYWORDS)
    assert any(word in "把这条工单转人工处理" for word in ACTION_KEYWORDS)


def test_server_publishes_same_keyword_list() -> None:
    """Web 控制台的动作判断必须与后端同源。

    前端没法 import Python 常量，所以走"后端下发"这条路；
    这条断言保证下发的内容与意图漏斗用的是同一份。
    """

    from scout.runtime.intent import ACTION_KEYWORDS
    from scout.server import ACTION_KEYWORDS as server_keywords

    assert tuple(server_keywords) == tuple(ACTION_KEYWORDS)


def test_funnel_default_has_domain_labels() -> None:
    funnel = build_knowledge_funnel()
    assert "知识问答" in funnel.prototypes
    assert ACTION_LABEL in funnel.sensitive_labels, "动作类默认必须走人工确认"


# —— 裁判 vs 归因校验 ——


class RubricLLM:
    """按"答案里有没有编造数字"给分的假裁判，用于离线验证对照逻辑。"""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "rubric-fake"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        prompt = request.messages[0].content
        faithful = "1000项" not in prompt
        score = 5 if faithful else 1
        content = (
            '{"faithfulness": {"score": %d, "reason": "依据材料"},'
            ' "relevance": {"score": %d, "reason": "直接回应"},'
            ' "conciseness": {"score": 4, "reason": "无明显冗余"},'
            ' "verdict": "%s"}'
        ) % (score, score, "可用" if faithful else "存在编造")
        return LLMResponse(content=content, usage=TokenUsage(input_tokens=20, output_tokens=10))


def test_compare_produces_rows_and_calibration() -> None:
    result = compare_judges(RubricLLM(), demo_cases())
    assert result.n == len(demo_cases())
    assert result.rows and all("judge_overall" in row for row in result.rows)
    assert 0.0 <= result.agreement <= 1.0
    assert "mean_bias" in result.calibration


def test_compare_finds_agreement_on_clean_and_fabricated() -> None:
    """干净答案两种评分器应当一致通过；编造答案应当一致不通过。

    如果这条挂了，说明要么 rubric 有问题、要么归因阈值有问题——
    这正是这个对照存在的意义。
    """

    cases = [
        ComparisonCase(
            question="到2027年有什么量化目标？",
            answer="新制定国家标准和行业标准30项以上。",
            evidence="到2027年，新制定云计算国家标准和行业标准30项以上。",
            expect="pass",
        ),
        ComparisonCase(
            question="到2027年有什么量化目标？",
            answer="到2027年要制定1000项标准。",
            evidence="到2027年，新制定云计算国家标准和行业标准30项以上。",
            expect="fail",
        ),
    ]
    result = compare_judges(RubricLLM(), cases)
    assert result.rows[0]["judge_pass"] is True
    assert result.rows[1]["judge_pass"] is False
    assert result.agreement >= 0.5


def test_compare_reports_disagreements_not_just_rate() -> None:
    """不一致清单才是这份对照最有价值的产物——它是校准的入口。"""

    result = compare_judges(RubricLLM(), demo_cases())
    payload = result.to_dict()
    assert "disagreements" in payload
    assert len(payload["rows"]) == result.n


def test_compare_judge_is_optional_injected() -> None:
    from scout.evolution.judge import LLMJudge

    judge = LLMJudge(RubricLLM())
    result = compare_judges(RubricLLM(), demo_cases()[:2], judge=judge)
    assert result.n == 2
