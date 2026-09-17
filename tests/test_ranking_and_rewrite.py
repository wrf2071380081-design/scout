"""排序与查询改写的测试。"""

from __future__ import annotations

from scout.rag.bm25 import BM25Index, idf_weight
from scout.rag.ranking import LexicalReranker, lexical_score, reciprocal_rank_fusion
from scout.rag.rewrite import QueryDefect, RewriteAdvisor, RewriteMethod, apply_plan


def test_rrf_is_deterministic_and_unweighted_by_default() -> None:
    rankings = {"dense": ["a", "b", "c"], "sparse": ["c", "a", "d"]}
    first = reciprocal_rank_fusion(rankings, k=60)
    second = reciprocal_rank_fusion(rankings, k=60)
    assert first == second
    # 两路都进前二 => 分数应高于只被单路命中的候选
    scores = dict(first)
    assert scores["a"] > scores["b"]
    assert scores["c"] > scores["b"]


def test_rrf_weights_change_ordering() -> None:
    """加权 RRF 必须真的能改变结果，否则消融实验没有意义。"""

    rankings = {"dense": ["x", "y"], "sparse": ["y", "x"]}
    balanced = dict(reciprocal_rank_fusion(rankings, k=60))
    assert balanced["x"] == balanced["y"]

    dense_heavy = dict(reciprocal_rank_fusion(rankings, k=60, weights={"dense": 3.0, "sparse": 1.0}))
    assert dense_heavy["x"] > dense_heavy["y"]


def test_rrf_zero_weight_drops_channel() -> None:
    rankings = {"dense": ["a"], "sparse": ["b"]}
    fused = dict(reciprocal_rank_fusion(rankings, weights={"dense": 0.0, "sparse": 1.0}))
    assert "a" not in fused
    assert "b" in fused


def test_lexical_score_ignores_function_words() -> None:
    document = "Redis 采用单线程模型处理命令，避免了上下文切换开销。"
    # 只加功能词不应提升分数
    assert lexical_score("为什么 Redis 快", document) > 0.0
    assert lexical_score("为什么 怎么 如何", document) == 0.0


def test_lexical_reranker_respects_candidate_limit_and_threshold() -> None:
    candidates = [(f"id{i}", "无关内容") for i in range(10)]
    candidates[0] = ("id0", "Redis 的内存存储带来高性能")

    reranker = LexicalReranker(candidate_limit=5)
    outcome = reranker.rerank("Redis 高性能", candidates)
    assert outcome.applied is True
    assert outcome.ordered[0][0] == "id0"
    # 超出 candidate_limit 的部分保持原顺序追加
    assert [item[0] for item in outcome.ordered[-5:]] == [f"id{i}" for i in range(5, 10)]

    strict = LexicalReranker(candidate_limit=10, min_score=0.5)
    strict_outcome = strict.rerank("Redis 高性能", candidates)
    assert strict_outcome.threshold_applied is True
    assert strict_outcome.dropped_count > 0


def test_rewrite_advisor_detects_typo() -> None:
    vocabulary = {"管理", "标准", "云计算", "安全"}
    advisor = RewriteAdvisor(vocabulary=vocabulary, document_frequency={"管理": 3, "标准": 9})
    report = advisor.diagnose("云计算标准体系里的管里标准是什么")
    assert QueryDefect.TYPO in report.defects

    plan = advisor.plan("云计算标准体系里的管里标准是什么", report, llm=None)
    assert plan.method is RewriteMethod.TERM_FIX
    assert plan.term_fixes.get("管里") == "管理"
    assert "管理" in apply_plan("管里标准", plan)


def test_rewrite_advisor_does_not_spuriously_fix_terms() -> None:
    """正常查询不得被"纠错"改坏。

    这是一个曾经真实出现的缺陷：单字之间的编辑距离 1 匹配几乎全是误报，
    "内" 会匹配到 "内存"，于是每个正常查询都被塞进一堆无意义的替换。
    修复后只对长度 ≥ 2 的 token 做纠错。
    """

    from scout.llm.scripted import tokenize

    corpus = "Redis 采用单线程模型，内存存储带来高性能，适合做缓存。"
    vocabulary = set(tokenize(corpus))
    # df 取一个真实语料里合理的量级。用 df=1 这种退化夹具会让所有词都被判成
    # "稀有"，从而误触 TOO_NARROW——那是夹具的问题，不是被测逻辑的问题。
    advisor = RewriteAdvisor(
        vocabulary=vocabulary,
        document_frequency={token: 12 for token in vocabulary},
    )

    query = "单线程 内存 性能"
    plan = advisor.plan(query, advisor.diagnose(query), llm=None)
    assert plan.method is RewriteMethod.NONE
    assert plan.term_fixes == {}


def test_rewrite_advisor_keeps_single_char_queries_untouched() -> None:
    from scout.llm.scripted import tokenize

    vocabulary = set(tokenize("内存 单线程 性能"))
    advisor = RewriteAdvisor(vocabulary=vocabulary)
    report = advisor.diagnose("内存")
    plan = advisor.plan("内存", report, llm=None)
    assert plan.method is not RewriteMethod.TERM_FIX


def test_rewrite_advisor_marks_over_generic_query_as_broad() -> None:
    advisor = RewriteAdvisor(
        vocabulary={"标准"},
        document_frequency={token: 40 for token in ("的", "了", "标准", "是")},
    )
    report = advisor.diagnose("标准是什么")
    assert QueryDefect.TOO_BROAD in report.defects or QueryDefect.ALIAS in report.defects
    plan = advisor.plan("标准是什么", report, llm=None)
    assert plan.triggers is True


def test_idf_weight_rewards_rare_terms() -> None:
    document_frequency = {"建设": 40, "标准": 40, "火星": 0, "殖民": 1}
    corpus_size = 40
    rare = idf_weight(document_frequency, corpus_size, "殖民")
    common = idf_weight(document_frequency, corpus_size, "建设")
    absent = idf_weight(document_frequency, corpus_size, "火星")
    assert rare > common
    assert absent > common
    # 没有语料统计时退化为不加权
    assert idf_weight(None, 0, "建设") == 1.0


def test_bm25_ranks_matching_document_first() -> None:
    index = BM25Index()
    index.fit(
        [
            "Redis 采用单线程模型，使用 IO 多路复用。",
            "Go 的 GMP 调度模型包含 G、M、P。",
            "秒杀系统用 Lua 脚本保证库存扣减原子性。",
        ]
    )
    ranked = index.search("Redis 单线程", top_k=3)
    assert ranked
    assert ranked[0][0] == 0
