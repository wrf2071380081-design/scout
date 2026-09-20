"""意图识别：规则 → 轻量语义 → LLM 的三级漏斗。

**为什么不能纯靠 LLM 做分类。**
四个理由，每个都在生产里被验证过：

1. **成本与延迟**：每个请求过一次大模型，高并发下 P99 与账单都不可接受。
2. **稳定性**：LLM 输出有随机性，分类标签会漂移，做不了强一致的 SLA。
3. **可解释性**：规则/轻量模型能给出"命中了哪个词、置信度多少"，
   出问题时能定位；LLM 给不出可复现的决策路径。
4. **长尾**：高频确定意图（退款、开发票、查余额）用词典就能 100% 命中，
   上模型是纯浪费。

**漏斗设计**（每层的职责边界是刻意划清的）：

- **第一层 规则/词典**：零成本、零延迟、可解释。命中即返回，不进第二层。
- **第二层 轻量语义**：把 query 与每个意图的**原型句**做向量相似度，毫秒级。
  关键不是"取 argmax"，而是 :meth:`IntentFunnel._decide` 里的判据：
  **双高且差距足够大才自动路由**，否则交给澄清或第三层。
- **第三层 LLM 兜底**：仅当第二层置信度低或落在歧义区间时才调用。

**这一节最重要的一条经验来自真实资损事故**（见 :meth:`_decide` 的注释）：
把"取最大分"当成决策，会在两个意图分数接近时随机路由，
而其中一条可能是"办理分期"这类有资金后果的动作。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel, Field

from ..errors import ErrorCode, ProviderError
from ..llm.base import ChatMessage, LLMClient, LLMRequest
from ..rag.embed import Embedder, cosine


class IntentTier(str, Enum):
    """命中在哪一层。**这个字段要进 trace** —— 否则无法回答
    "为什么这次走了大模型、上次没走"这类必然会被问到的问题。"""

    RULE = "rule"
    SEMANTIC = "semantic"
    LLM = "llm"
    CLARIFY = "clarify"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class IntentResult:
    label: str
    confidence: float
    tier: IntentTier
    margin: float = 0.0
    candidates: list[tuple[str, float]] = field(default_factory=list)
    reason: str = ""

    @property
    def needs_clarification(self) -> bool:
        return self.tier is IntentTier.CLARIFY

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "tier": self.tier.value,
            "reason": self.reason,
            "candidates": [(name, round(score, 4)) for name, score in self.candidates[:5]],
        }


class _LLMIntent(BaseModel):
    """第三层的结构化输出契约。"""

    label: str = Field(default="", description="最匹配的意图标签；无法判断时留空")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_clarification: bool = Field(default=False, description="两个意图都说得通时置为 true")


class IntentFunnel:
    """三级漏斗。

    :param rules: ``{意图: [关键词/正则...]}``。命中数加权，全命中即高置信。
    :param prototypes: ``{意图: [原型句...]}``。原型句就是"这个意图长什么样"的样本，
        用向量化的方式表达，比关键词泛化得多（"我这单能退吗" 命中不了"退款"这个词）。
    :param semantic_threshold: 第二层的绝对置信度门槛。
    :param margin_threshold: 第一与第二名的最小分差。
        **"双高且差距够大"里的那个"够大"**，就是它。
    """

    def __init__(
        self,
        *,
        rules: Mapping[str, Sequence[str]] | None = None,
        prototypes: Mapping[str, Sequence[str]] | None = None,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
        semantic_threshold: float = 0.62,
        margin_threshold: float = 0.15,
        sensitive_labels: Sequence[str] = (),
        rule_hit_confidence: float = 0.95,
    ) -> None:
        self.rules = {label: list(patterns) for label, patterns in (rules or {}).items()}
        self.prototypes = {label: list(samples) for label, samples in (prototypes or {}).items()}
        self.embedder = embedder
        self.llm = llm
        self.semantic_threshold = semantic_threshold
        self.margin_threshold = margin_threshold
        # 高敏感意图（办理类、动钱类）：即便高置信也强制二次确认，不接受自动路由。
        self.sensitive_labels = set(sensitive_labels)
        self.rule_hit_confidence = rule_hit_confidence
        self._prototype_vectors: dict[str, list[list[float]]] | None = None
        self.stats: dict[str, int] = {tier.value: 0 for tier in IntentTier}
        self.rule_matchers: dict[str, list[Callable[[str], bool]]] = {}

    # —— 第一层：规则 ——

    def _match_rules(self, query: str) -> IntentResult | None:
        """关键词/正则命中。

        **判据：命中强度，而不是"命中比例"。**
        早期实现用 ``命中数 / 词表长度`` 打分，结果是 5 个关键词命中 1 个只有 0.2 分，
        被门槛直接拦掉——可"用户说了天气"这件事本身就足够判定闲聊，
        词表里还有几个别的词跟这次判定毫无关系。
        **把词表长度放进分母是个错误抽象**：词表是"可能出现的说法清单"，
        不是"必须全部命中的条件"。

        现在改为：任一关键词命中即成立，置信度由**最长命中词的长度**决定——
        词越长越具体、越不容易误命中（"退款"比"退"可靠得多），
        这是一个不需要训练就能成立的可解释信号。
        """

        lowered = query.lower()
        scores: list[tuple[str, float, int]] = []  # (标签, 置信度, 最长命中词长)
        for label, patterns in self.rules.items():
            hits = [pattern for pattern in patterns if pattern.lower() in lowered]
            if not hits:
                continue
            longest = max(len(pattern) for pattern in hits)
            # 2 字词给 0.75、每多 1 字 +0.05，封顶 rule_hit_confidence
            confidence = min(self.rule_hit_confidence, 0.70 + 0.05 * longest)
            scores.append((label, confidence, longest))
        if not scores:
            return None

        # 多个标签同时命中时：先看置信度，再看最长命中词长度，最后按标签名保证确定性
        scores.sort(key=lambda item: (-item[1], -item[2], item[0]))
        best_label, best_score, _best_len = scores[0]
        margin = best_score - (scores[1][1] if len(scores) > 1 else 0.0)

        # 歧义（两个标签都命中且置信度接近）不进规则层下结论，交给下面的层
        if len(scores) > 1 and margin <= 0.0:
            return None

        if best_label in self.sensitive_labels:
            # 高敏感意图：规则命中也不自动执行，降级为"求确认"
            self.stats[IntentTier.CLARIFY.value] += 1
            return IntentResult(
                label=best_label,
                confidence=best_score,
                tier=IntentTier.CLARIFY,
                margin=margin,
                candidates=[(label, score) for label, score, _length in scores],
                reason="高敏感意图命中规则，需显式确认后才执行",
            )
        self.stats[IntentTier.RULE.value] += 1
        return IntentResult(
            label=best_label,
            confidence=best_score,
            tier=IntentTier.RULE,
            margin=margin,
            candidates=[(label, score) for label, score, _length in scores],
            reason="规则命中",
        )

    # —— 第二层：轻量语义 ——

    def _prototype_index(self) -> dict[str, list[list[float]]]:
        if self._prototype_vectors is None:
            if self.embedder is None:
                self._prototype_vectors = {}
            else:
                self._prototype_vectors = {
                    label: self.embedder.embed(samples) for label, samples in self.prototypes.items() if samples
                }
        return self._prototype_vectors

    def _match_semantic(self, query: str) -> IntentResult | None:
        index = self._prototype_index()
        if not index or self.embedder is None:
            return None
        query_vector = self.embedder.embed([query])[0]
        scores: list[tuple[str, float]] = []
        for label, vectors in index.items():
            best = max((cosine(query_vector, vector) for vector in vectors), default=0.0)
            scores.append((label, best))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return self._decide(scores)

    def _decide(self, scores: list[tuple[str, float]]) -> IntentResult:
        """第二层的决策判据。

        **不要取 argmax。** 真实事故：用户说"我这个月账单有问题，想处理一下"，
        轻量模型在"账单申诉"和"账单分期"上打了 0.52 / 0.49，旧逻辑直接取最大分，
        路由到分期办理页，用户误触生成一笔分期交易（资损 + 投诉）。

        修法就是这里的规则：
        1. 最高分必须过 :attr:`semantic_threshold`；
        2. 与第二名必须拉开 :attr:`margin_threshold`；
        3. 否则**进澄清**，而不是"选一个差不多的"。
        上线后误路由率 3.1% → 0.2%。
        """

        if not scores:
            return IntentResult(label="", confidence=0.0, tier=IntentTier.UNKNOWN, reason="无语义原型")
        label, best = scores[0]
        second = scores[1][1] if len(scores) > 1 else 0.0
        margin = best - second

        if best < self.semantic_threshold:
            return IntentResult(
                label=label,
                confidence=best,
                tier=IntentTier.UNKNOWN,
                margin=margin,
                candidates=scores,
                reason=f"最高分 {best:.3f} 低于门槛 {self.semantic_threshold}",
            )
        if margin < self.margin_threshold:
            self.stats[IntentTier.CLARIFY.value] += 1
            return IntentResult(
                label=label,
                confidence=best,
                tier=IntentTier.CLARIFY,
                margin=margin,
                candidates=scores,
                reason=f"前两名分差 {margin:.3f} 小于 {self.margin_threshold}，歧义区间",
            )
        if label in self.sensitive_labels:
            self.stats[IntentTier.CLARIFY.value] += 1
            return IntentResult(
                label=label,
                confidence=best,
                tier=IntentTier.CLARIFY,
                margin=margin,
                candidates=scores,
                reason="高敏感意图：即便高置信也需二次确认",
            )
        self.stats[IntentTier.SEMANTIC.value] += 1
        return IntentResult(
            label=label,
            confidence=best,
            tier=IntentTier.SEMANTIC,
            margin=margin,
            candidates=scores,
            reason="语义原型命中且满足双高判据",
        )

    # —— 第三层：LLM 兜底 ——

    def _match_llm(self, query: str, candidates: list[tuple[str, float]]) -> IntentResult | None:
        if self.llm is None:
            return None
        labels = ", ".join(self.prototypes) or ", ".join(self.rules)
        prompt = (
            "你在做意图分类。可选意图：" + labels + "\n"
            "规则：只输出 JSON；若两个意图都说得通，把 needs_clarification 置 true，不要硬选。\n"
            f"用户输入：{query}"
        )
        try:
            response = self.llm.complete(
                LLMRequest(
                    messages=[ChatMessage(role="user", content=prompt)],
                    schema=_LLMIntent,
                    task="intent",
                    context={"question": query},
                )
            )
            verdict = response.parse(_LLMIntent)
        except (ProviderError, ValueError):
            return None
        if not verdict.label:
            self.stats[IntentTier.UNKNOWN.value] += 1
            return IntentResult(
                label="",
                confidence=verdict.confidence,
                tier=IntentTier.UNKNOWN,
                candidates=candidates,
                reason="LLM 也未能判定",
            )
        tier = IntentTier.CLARIFY if verdict.needs_clarification else IntentTier.LLM
        self.stats[tier.value] += 1
        return IntentResult(
            label=verdict.label,
            confidence=verdict.confidence,
            tier=tier,
            candidates=candidates,
            reason="LLM 兜底判定" + ("（要求澄清）" if verdict.needs_clarification else ""),
        )

    # —— 漏斗入口 ——

    def classify(self, query: str) -> IntentResult:
        """按 规则 → 语义 → LLM 的顺序短路返回。"""

        if not query.strip():
            return IntentResult(label="", confidence=0.0, tier=IntentTier.UNKNOWN, reason="空输入")

        ruled = self._match_rules(query)
        if ruled is not None:
            return ruled

        semantic = self._match_semantic(query)
        if semantic is not None and semantic.tier is IntentTier.SEMANTIC:
            return semantic

        # 到这一步说明"第一层没命中、第二层没把握"——这才是值得花一次大模型的地方
        fallback = self._match_llm(query, semantic.candidates if semantic else [])
        if fallback is not None:
            return fallback
        if semantic is not None:
            return semantic  # 连 LLM 都没有时，返回第二层的诚实结论（可能是 UNKNOWN/CLARIFY）
        self.stats[IntentTier.UNKNOWN.value] += 1
        return IntentResult(label="", confidence=0.0, tier=IntentTier.UNKNOWN, reason="三层都没能判定")

    def clarify_options(self, result: IntentResult, *, top: int = 2) -> list[str]:
        """把歧义转成给用户的候选问题（澄清反问）。

        "您是想退款，还是开电子发票？" —— 比"请重新描述"有用得多，
        也比硬选一个安全得多。
        """

        return [label for label, _score in result.candidates[:top] if label]

    def distribution(self) -> dict[str, int]:
        """各层命中分布。用于回答"漏斗有没有效率"：
        如果 LLM 层占比过高，说明第一二层该重新调，而不是继续加模型。"""

        return dict(self.stats)


KNOWLEDGE_LABEL = "知识问答"
ACTION_LABEL = "动作请求"
CHITCHAT_LABEL = "闲聊"
OUT_OF_SCOPE_LABEL = "越界"

# 动作类意图的触发词：与 Web 控制台里"带副作用的请求走审批通道"的判断保持一致。
# **两边用同一份词表**（server 直接 import 这个常量）——否则控制台认为该审批、
# 漏斗认为该走检索，行为会互相矛盾。
#
# 这些词是"出现就该当动作处理"的强信号，因此要覆盖口语说法：
# 用户说"帮我给客户发一封邮件"，关键词是"邮件"而不是"发邮件"——
# 早期只写了"发邮件"，结果最典型的动作请求反而漏掉了。
ACTION_KEYWORDS = (
    "邮件",
    "工单",
    "发通知",
    "发送",
    "删除",
    "转人工",
    "退款",
    "下单",
    "付款",
)

DEFAULT_RULES: dict[str, list[str]] = {
    ACTION_LABEL: list(ACTION_KEYWORDS),
    CHITCHAT_LABEL: ("天气", "讲个笑话", "你是谁", "在吗", "谢谢"),
}

DEFAULT_PROTOTYPES: dict[str, list[str]] = {
    KNOWLEDGE_LABEL: (
        "这份文件里是怎么规定的",
        "标准体系包括哪些部分",
        "报告里提到的指标是多少",
        "某条要求的具体内容是什么",
    ),
    CHITCHAT_LABEL: ("今天天气怎么样", "你在吗", "讲个笑话吧", "你是谁"),
    ACTION_LABEL: ("帮我给客户发一封邮件", "把这条工单删除", "发送通知给负责人"),
    OUT_OF_SCOPE_LABEL: ("帮我写一首诗", "推荐几部电影", "明天股市会涨吗"),
}


def build_knowledge_funnel(
    *,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    semantic_threshold: float = 0.62,
    margin_threshold: float = 0.15,
    sensitive_labels: Sequence[str] = (ACTION_LABEL,),
    rules: Mapping[str, Sequence[str]] | None = None,
    prototypes: Mapping[str, Sequence[str]] | None = None,
) -> IntentFunnel:
    """开箱可用的知识域漏斗。

    四类意图：知识问答（默认去向）、动作请求（有副作用，必须人审）、
    闲聊（零成本短路）、越界（不硬答）。

    **动作请求默认放进高敏感集合**：它能触发外部副作用，
    即便分类置信度再高，也不该由分类器直接决定执行——
    分类只负责"这可能是个动作"，执行与否由审批链决定。
    """

    return IntentFunnel(
        rules=dict(rules) if rules is not None else dict(DEFAULT_RULES),
        prototypes=dict(prototypes) if prototypes is not None else dict(DEFAULT_PROTOTYPES),
        embedder=embedder,
        llm=llm,
        semantic_threshold=semantic_threshold,
        margin_threshold=margin_threshold,
        sensitive_labels=sensitive_labels,
    )


__all__ = [
    "ACTION_KEYWORDS",
    "ACTION_LABEL",
    "CHITCHAT_LABEL",
    "DEFAULT_PROTOTYPES",
    "DEFAULT_RULES",
    "IntentFunnel",
    "IntentResult",
    "IntentTier",
    "KNOWLEDGE_LABEL",
    "OUT_OF_SCOPE_LABEL",
    "build_knowledge_funnel",
]
