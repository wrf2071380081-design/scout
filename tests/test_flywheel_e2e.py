"""飞轮端到端契约测试：观测日志 → 挖掘 → 复核 → 数据集增量。

**这一组测试的核心目的，是钉住"观测日志"与"挖掘器"之间的字段契约。**
两边曾经各自都对、接起来就错：挖掘器读 ``retrieved`` / ``support_rate``，
而观测写的是 ``retrieved_sources`` / ``answer_support_rate``。
字段名不一致不会报错——取值全为 None、检索数量当 0、
于是**每条观测都被判成"空检索"并跳过，飞轮转一圈挖出 0 条候选而报告一切正常**。

单测里两边各自都过，只有端到端跑才会暴露。所以这里断言的是**真实管线产出的记录**，
不是手搓的字典。
"""

from __future__ import annotations

from scout.evaluation.dataset import EvalCase, EvalDataset, load_dataset, save_dataset
from scout.evolution import FailureMiner, FailureReason, ReviewQueue
from scout.evolution.observations import (
    ObservationLog,
    observation_from_pipeline,
    mine_report,
)
from scout.llm.scripted import HeuristicLLM
from scout.rag.pipeline import RAGPipeline, build_index
from scout.evaluation.taxonomy import QueryTag

DOCS = [
    ("a.md", "云计算标准体系结构包括基础、技术、服务、应用、管理和安全六个部分。"),
    ("b.md", "低空经济标准体系重点围绕低空航空器、起降设施与运行服务展开。"),
]


def _pipeline() -> RAGPipeline:
    return RAGPipeline(build_index(DOCS), HeuristicLLM())


def test_observation_roundtrip_via_real_pipeline(tmp_path) -> None:
    """真实管线 → 观测 → 落盘 → 读回，字段必须齐全。"""

    pipeline = _pipeline()
    result = pipeline.answer("云计算标准体系结构包括哪几个部分？")
    observation = observation_from_pipeline(result, case_id="c1")

    log = ObservationLog(tmp_path / "obs.jsonl")
    log.append(observation)
    records = log.read()
    assert len(records) == 1

    record = records[0]
    # 这几个字段就是挖掘器的输入契约，缺一个都会让挖掘静默失效
    for key in ("question", "outcome", "answer", "retrieved_sources", "answer_support_rate"):
        assert key in record, f"观测缺少字段 {key}——挖掘器会读不到它"
    assert record["retrieved_sources"], "检索来源要落盘，否则挖掘器判不出'检索为空'"


def test_miner_reads_real_observation_records(tmp_path) -> None:
    """契约测试：**用真实管线产出的记录**喂挖掘器，必须挖得出东西。

    这条断言就是为了拦住那次字段错配——如果挖掘器又只认旧键名，
    它会把这些记录全判成"空检索"，于是一条都挖不出来，而测试会立刻失败。
    """

    pipeline = _pipeline()
    log = ObservationLog(tmp_path / "obs.jsonl")

    # 一条正常回答 + 一条检索为空的拒答
    log.append(observation_from_pipeline(pipeline.answer("云计算标准体系包括哪几个部分？")))
    log.append(observation_from_pipeline(pipeline.answer("完全不相关的问题：明天股市会涨吗？")))

    records = log.read()
    summary = mine_report(records)
    assert summary["total"] == 2

    miner = FailureMiner()
    candidates = miner.mine(records)
    # 第二条应当被判成"检索为空"（进而默认不进候选池），而不是把两条都判成空检索
    assert miner.stats.skipped_no_retrieval <= 1, (
        "两条里最多只有一条是空检索；若两条都算空检索，说明字段名又不一致了"
    )
    assert isinstance(candidates, list)


def test_abstained_observation_becomes_candidate(tmp_path) -> None:
    """拒答要能被识别成"该答没答"，而不是被误判成运行异常。

    ``error_code`` 在本项目里是**类型化结果**的标识（拒答自带 error_code），
    把它当异常信号会让所有拒答都被归到 ERROR，而报告上看起来一切正常。
    """

    record = {
        "question": "某个证据不足的问题",
        "outcome": "insufficient_evidence",
        "error_code": "insufficient_evidence",  # 注意：这不是异常
        "answer": "根据现有资料无法回答该问题。",
        "retrieved_sources": ["doc.md"],
        "answer_support_rate": 0.0,
        "abstained": True,
    }
    miner = FailureMiner()
    candidates = miner.mine([record])
    assert len(candidates) == 1
    assert candidates[0].reason is FailureReason.ABSTAINED
    assert candidates[0].retrieved == 1, "数量要从 retrieved_sources 列表算出来"


def test_real_error_is_still_detected() -> None:
    """反过来：真正的异常必须仍被识别为 ERROR。"""

    miner = FailureMiner()
    candidates = miner.mine(
        [
            {
                "question": "运行崩了的问题",
                "outcome": "",
                "failed": True,
                "retrieved_sources": ["doc.md"],
            }
        ]
    )
    assert candidates and candidates[0].reason is FailureReason.ERROR


def test_low_support_ignores_not_applicable_zero() -> None:
    """``support_rate = 0`` 在拒答样本里是"不适用"，不是"零支撑"。

    把两者混同，会把拒答误报成幻觉——而幻觉率是个会被认真对待的指标。
    """

    miner = FailureMiner()
    candidates = miner.mine(
        [
            {
                "question": "拒答样本",
                "outcome": "insufficient_evidence",
                "retrieved_sources": ["a", "b"],
                "answer_support_rate": 0.0,
            },
            {
                "question": "真的低支撑",
                "outcome": "answered",
                "retrieved_sources": ["a"],
                "answer_support_rate": 0.2,
            },
        ]
    )
    reasons = {case.question: case.reason for case in candidates}
    assert reasons["拒答样本"] is FailureReason.ABSTAINED
    assert reasons["真的低支撑"] is FailureReason.LOW_SUPPORT


def test_flywheel_increment_is_loadable(tmp_path) -> None:
    """增量数据集必须能被主流程加载——写得出但读不回来等于没写。"""

    dataset = EvalDataset(
        name="t",
        cases=[
            EvalCase(
                case_id="c1",
                question="同一个问题",
                tags=[QueryTag.SINGLE_FACT],
                gold_snippets=["片段"],
            ),
            EvalCase(
                case_id="cand-1",
                question="飞轮挖出的新问题",
                tags=[QueryTag.MULTI_HOP],
                gold_snippets=[],
                allow_unknown=True,
                notes="飞轮挖出：abstained｜待补 gold",
            ),
        ],
    )
    path = tmp_path / "ds.json"
    save_dataset(dataset, path)

    reloaded = load_dataset(path)
    assert len(reloaded.cases) == 2
    assert reloaded.validate() == [], "带 allow_unknown 的候选样本必须通过校验"


def test_review_queue_gates_the_increment() -> None:
    """复核队列是增量进评测集的唯一闸门。"""

    miner = FailureMiner()
    candidates = miner.mine(
        [
            {
                "question": "候选问题",
                "outcome": "insufficient_evidence",
                "retrieved_sources": ["a"],
            }
        ]
    )
    queue = ReviewQueue()
    queue.enqueue(candidates)
    assert not queue.approved, "自动挖掘不得直接入库"
    queue.approve(candidates[0].case_id)
    assert len(queue.approved) == 1
