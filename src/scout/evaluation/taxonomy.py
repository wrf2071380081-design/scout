"""查询标签体系。

标签不是分类癖。它决定了两件事：

1. **评测集是否能暴露真实短板。** 只用"单事实"类问题测，任何 RAG 都能拿高分；
   真正拉开差距的是多跳、时间版本、来源冲突、近似实体、表格、注入这几类。
2. **失败能不能被归因。** 当整体准确率下降时，只有按标签切片看，
   才能知道是分块坏了（表格类掉）、还是改写坏了（错别字类掉）、
   还是拒答策略坏了（无知识类掉）。

每个标签都对应一个**已知会翻车的机制**，注释里写明了。
"""

from __future__ import annotations

from enum import Enum


class QueryTag(str, Enum):
    """查询标签。"""

    SINGLE_FACT = "single_fact"
    """单一事实。答案存在于一个连续片段内。这是基线，不该有系统在这里失败。"""

    DEFINITION = "definition"
    """定义/概念解释。考察摘要能力，而不是定位能力。"""

    PARAMETER = "parameter"
    """参数/阈值/配置项。数字必须精确，容忍度为零，重排与精确匹配影响最大。"""

    CROSS_DOCUMENT = "cross_document"
    """跨文档综合。考察多路召回与融合，单文档 Top-K 必然不够。"""

    MULTI_HOP = "multi_hop"
    """多跳推理。需要 A→B→C 链式查找，是"子问题分解"与"图结构记忆"的主战场。"""

    COMPARISON = "comparison"
    """对比类。需要同时取到两个对象的证据，只命中一方会给出片面结论。"""

    TIME_VERSION = "time_version"
    """时间/版本敏感。同一指标在不同年份或版本取值不同，考察快照与时效处理。"""

    TABLE = "table"
    """表格。纯文本切分会破坏行列结构，是"分块策略"最容易暴露问题的一类。"""

    CODE = "code"
    """代码/标识符。大小写、下划线、符号必须原样匹配，稀疏检索的主场。"""

    AMBIGUITY = "ambiguity"
    """指代或限定不明。正确行为是**澄清**而不是猜。考察拒答/澄清策略。"""

    NO_KNOWLEDGE = "no_knowledge"
    """知识库未涵盖。正确行为是**明确拒答**，任何编造都是失败。"""

    SOURCE_CONFLICT = "source_conflict"
    """来源冲突。两个文档给出不同说法，考察是否如实呈现分歧而不是擅自择一。"""

    NEAR_ENTITY = "near_entity"
    """近似实体（同名不同主体、型号相近）。考察检索是否被表面相似度误导。"""

    TYPO = "typo"
    """错别字/别名。考察查询改写的**缺陷诊断**能力，也是改写触发率的主要来源。"""

    LONG_QUESTION = "long_question"
    """长问题。考察复杂度路由与子问题分解。"""

    PROMPT_INJECTION = "prompt_injection"
    """检索侧提示注入。考察输入消毒与归因门控，属于安全维度。"""


# 标签的中文说明。报告里直接用，避免读者去翻代码。
TAG_DESCRIPTIONS: dict[QueryTag, str] = {
    QueryTag.SINGLE_FACT: "单一事实定位",
    QueryTag.DEFINITION: "概念/定义解释",
    QueryTag.PARAMETER: "参数与阈值（要求精确）",
    QueryTag.CROSS_DOCUMENT: "跨文档综合",
    QueryTag.MULTI_HOP: "多跳推理",
    QueryTag.COMPARISON: "多对象对比",
    QueryTag.TIME_VERSION: "时间/版本敏感",
    QueryTag.TABLE: "表格结构提取",
    QueryTag.CODE: "代码与标识符",
    QueryTag.AMBIGUITY: "歧义查询（应澄清）",
    QueryTag.NO_KNOWLEDGE: "知识库未涵盖（应拒答）",
    QueryTag.SOURCE_CONFLICT: "来源冲突",
    QueryTag.NEAR_ENTITY: "近似实体干扰",
    QueryTag.TYPO: "错别字/别名",
    QueryTag.LONG_QUESTION: "长问题/多要点",
    QueryTag.PROMPT_INJECTION: "检索侧提示注入",
}

# 期望"拒答/澄清"的标签集合。评测时用来判断"该拒有没有拒"。
REFUSAL_EXPECTED_TAGS: frozenset[QueryTag] = frozenset(
    {QueryTag.NO_KNOWLEDGE, QueryTag.AMBIGUITY}
)

# 期望"必须回答且必须准确"的标签集合。
MUST_ANSWER_TAGS: frozenset[QueryTag] = frozenset(
    {
        QueryTag.SINGLE_FACT,
        QueryTag.DEFINITION,
        QueryTag.PARAMETER,
        QueryTag.CODE,
    }
)


def describe(tag: QueryTag) -> str:
    return TAG_DESCRIPTIONS.get(tag, tag.value)


__all__ = [
    "MUST_ANSWER_TAGS",
    "REFUSAL_EXPECTED_TAGS",
    "TAG_DESCRIPTIONS",
    "QueryTag",
    "describe",
]
