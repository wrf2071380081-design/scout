"""确定性 LLM 实现，用于离线运行与单元测试。

这里有两个类：

:class:`ScriptedLLM`
    按顺序回放预置响应。用于单元测试中精确控制 Agent 的每一步行为。

:class:`HeuristicLLM`
    一个不依赖网络的"穷人版模型"：用词法重叠、句式规则和简单状态机
    完成 scout 内部的全部模型任务（复杂度判断、相关性打分、改写、作答、校验）。

``HeuristicLLM`` 存在的意义不是"效果好"——它明确效果一般——而是让这个仓库
**克隆下来就能跑通端到端流程**，不需要任何 API Key。评测框架的可信度来自
流程本身可复现，而不是来自某一次调用的分数。生产使用请配置
``SCOUT_LLM_BASE_URL`` 切换到真实模型。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .base import ChatMessage, LLMClient, LLMRequest, LLMResponse, ToolCall, TokenUsage

_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ARITHMETIC = re.compile(r"[\d+\-*/×÷()（）.]{3,}")
_DATE_WORDS = ("今天", "昨天", "明天", "现在", "日期", "几号", "星期", "时间")

_COMPLEX_MARKERS = (
    "对比", "区别", "分别", "以及", "同时", "综合", "多跳", "影响", "原因和",
    "为什么", "如何影响", "比较", "各自", "之间关系", "跨文档",
)
_EVIDENCE_BLOCK = re.compile(r"^\[(\d+)]\s*(.*)$", re.MULTILINE)

# 中文功能词。匹配与打分必须把它们排除，否则"为什么/怎么/如何"这类词
# 会占据查询 token 的绝大部分，让真实内容的权重被稀释到接近零。
FUNCTION_WORDS = frozenset(
    "的了和与及或在是为以对从于其中这那有这个一个什么怎么如何哪些请根据"
    "呢吗吧啊呀我你他她们它您谁何时经常可以能够应该需要是否因为所以如果"
    "但是而且并且以及同时就是都要会能可就也还但而所不没很太更最等一二三"
    "关于下面上述以上问题回答说明内容资料知识库字符串"
)


def content_tokens(text: str) -> set[str]:
    """只保留有实义的 token。

    单词：过滤功能词；双字组：仅当两个字符都是功能词时才过滤
    （避免把"内存""调度"这类由常见字组成的术语误删）。
    """

    result: set[str] = set()
    for token in tokenize(text):
        if len(token) == 1:
            if token in FUNCTION_WORDS:
                continue
        elif len(token) == 2 and all(char in FUNCTION_WORDS for char in token):
            continue
        result.add(token)
    return result


def tokenize(text: str) -> list[str]:
    """轻量分词：拉丁词 + 中文单字 + **连续中文段内**的双字组。

    没有引入分词库是刻意的权衡：双字组已经能提供可用的词法信号，
    而省掉一个含 C 扩展的依赖能让本项目在任何环境秒装。
    真实场景应换成交付级分词器，或直接交给语义向量模型。

    双字组必须在**连续中文段内**生成，不能跨空格与标点：
    否则 "单线程 内存" 会被切出 `程内` 这种跨词伪双字组，
    它在语料里几乎不出现因而 IDF 极高，会污染排序与覆盖率估计
    （实测这类伪词曾把"错别字纠正"触发到正常查询上，把查询改坏）。
    """

    lowered = text.lower()
    tokens: list[str] = [word.lower() for word in _LATIN_WORD.findall(lowered)]
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句，保留标点。"""

    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    return [part.strip() for part in parts if part and part.strip()]


@dataclass(slots=True)
class ScriptedLLM:
    """按顺序回放预置响应。"""

    responses: list[LLMResponse] = field(default_factory=list)
    cursor: int = 0
    requests: list[LLMRequest] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return "scripted"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.cursor >= len(self.responses):
            # 超出预置范围时给出终止性响应，避免测试中出现隐式死循环。
            return LLMResponse(content="", finish_reason="stop")
        response = self.responses[self.cursor]
        self.cursor += 1
        return response


class HeuristicLLM:
    """不依赖网络的确定性实现。

    :param answerable_threshold: 判"证据足以作答"的加权覆盖率阈值。
        应当与 :class:`scout.config.VerifySettings` 的 ``sufficiency_min_coverage``
        保持一致——两处用不同阈值会让"评分器说能答、门控说不能答"，
        白白多做一次改写与生成。
    :param ambiguous_threshold: 判"问题过于模糊、需要澄清"的加权覆盖率阈值
    """

    def __init__(
        self,
        *,
        model: str = "heuristic",
        answerable_threshold: float = 0.5,
        ambiguous_threshold: float = 0.12,
    ) -> None:
        self._model = model
        self.answerable_threshold = answerable_threshold
        self.ambiguous_threshold = ambiguous_threshold

    @property
    def model_name(self) -> str:
        return self._model

    # —— 主入口 ——

    def complete(self, request: LLMRequest) -> LLMResponse:
        handler = getattr(self, f"_task_{request.task}", None)
        if handler is None:
            handler = self._task_answer
        produced = handler(request)
        if isinstance(produced, list):
            return LLMResponse(
                content="",
                tool_calls=produced,
                usage=TokenUsage(input_tokens=0, output_tokens=0),
                model=self._model,
                finish_reason="tool_calls",
            )
        return LLMResponse(
            content=produced,
            usage=TokenUsage(
                input_tokens=sum(len(tokenize(m.content)) for m in request.messages),
                output_tokens=len(tokenize(produced)),
            ),
            model=self._model,
            finish_reason="stop",
        )

    # —— 工具：把正文拼成一段可检索的文本 ——

    @staticmethod
    def _last_text(request: LLMRequest) -> str:
        for message in reversed(request.messages):
            if message.content:
                return message.content
        return ""

    @staticmethod
    def _user_question(request: LLMRequest) -> str:
        """取原始问题。

        优先读 ``request.context['question']``——提示词模板把问题、证据、约束
        拼成一大段文本后，词法分析无法可靠还原原始问题，显式传入才可靠。
        只有在调用方没有提供时才退化为扫描第一条 user 消息。
        """

        explicit = request.context.get("question")
        if isinstance(explicit, str) and explicit.strip():
            return explicit
        for message in request.messages:
            if message.role == "user" and message.content:
                return message.content
        return ""

    @staticmethod
    def _tool_messages(request: LLMRequest) -> list[ChatMessage]:
        return [message for message in request.messages if message.role == "tool"]

    # —— 任务：复杂度判断 ——

    def _task_complexity(self, request: LLMRequest) -> str:
        question = self._user_question(request)
        score = len(question)
        markers = sum(1 for marker in _COMPLEX_MARKERS if marker in question)
        question_marks = question.count("？") + question.count("?")
        complex_question = markers >= 1 or question_marks >= 2 or score >= 60
        return '{"complexity": "%s", "reason": "%s"}' % (
            "complex" if complex_question else "simple",
            f"len={score},markers={markers},q={question_marks}",
        )

    # —— 任务：证据相关性打分与路由 ——

    def _task_grade(self, request: LLMRequest) -> str:
        """按**稀有度加权覆盖率**评估证据。

        权重 = 词长 × IDF。两个因素缺一不可：

        - 词长：长词更具体（"Redis" 比 "快" 重要）。
        - IDF：稀有词更有信息量。政策语料里 "建设""标准" 几乎每篇都出现，
          如果按词面平均计权，一个完全无法回答的问题仅凭命中这两个词
          就能拿到过半覆盖率，从而绕过拒答门控。加权后才符合直觉：
          **覆盖 "火星""殖民" 才算数，覆盖 "建设""标准" 不算。**

        IDF 表通过 ``request.context`` 传入（真实模型会忽略它）。
        这里延迟导入 ``idf_weight`` 是为了打断 llm ↔ rag 的循环依赖。
        """

        from ..rag.bm25 import idf_weight

        text = self._last_text(request)
        question = self._user_question(request)
        blocks = self._split_evidence(text)
        if not blocks:
            return '{"relevance": 0.0, "coverage": 0.0, "answerable": false, "ambiguous": false, "route": "rewrite"}'

        query_tokens = content_tokens(question)
        if not query_tokens:
            # 查询里没有任何实义词（例如"这个指南说了什么"）——这是**歧义**，
            # 不是"可作答"。早期版本在这里返回 answerable=true，
            # 直接导致歧义样本被当成正常问题回答。
            return '{"relevance": 0.0, "coverage": 0.0, "answerable": false, "ambiguous": true, "route": "clarify"}'

        document_frequency = request.context.get("document_frequency")
        corpus_size = int(request.context.get("corpus_size") or 0)

        def weight(token: str) -> float:
            return float(max(len(token), 1)) * idf_weight(document_frequency, corpus_size, token)

        total_weight = sum(weight(token) for token in query_tokens)
        if total_weight <= 0:
            return '{"relevance": 0.0, "coverage": 0.0, "answerable": false, "ambiguous": false, "route": "rewrite"}'

        best = 0.0
        union: set[str] = set()
        for _index, body in blocks:
            body_tokens = content_tokens(body)
            union |= body_tokens
            covered = sum(weight(token) for token in query_tokens if token in body_tokens)
            best = max(best, covered / total_weight)

        coverage = sum(weight(token) for token in query_tokens if token in union) / total_weight
        answerable = best >= self.answerable_threshold
        ambiguous = coverage < self.ambiguous_threshold
        route = "answer" if answerable else ("clarify" if ambiguous else "rewrite")
        return (
            '{"relevance": %.3f, "coverage": %.3f, "answerable": %s, "ambiguous": %s, "route": "%s"}'
            % (round(best, 3), round(coverage, 3), str(answerable).lower(), str(ambiguous).lower(), route)
        )

    # —— 任务：查询改写 ——

    def _task_rewrite(self, request: LLMRequest) -> str:
        question = self._user_question(request)
        has_number = bool(_ARITHMETIC.search(question))
        has_latin = bool(_LATIN_WORD.search(question))
        if has_number or has_latin:
            abstract = re.sub(r"[A-Za-z0-9:：\-—]{2,}", "", question)
            abstract = re.sub(r"\s+", "", abstract) or question
            return (
                '{"method": "step_back", "step_back_question": "%s", "hyde_document": ""}'
                % f"{abstract}的基本原理与通用做法是什么"
            )
        return (
            '{"method": "hyde", "hyde_document": "%s", "step_back_question": ""}'
            % f"针对「{question}」，可参考的资料通常会说明其定义、适用条件与典型做法。"
        )

    # —— 任务：作答 ——

    def _task_answer(self, request: LLMRequest) -> str:
        text = self._last_text(request)
        question = self._user_question(request)
        blocks = self._split_evidence(text)
        if not blocks:
            return "根据现有资料无法回答该问题。"
        query_tokens = content_tokens(question)
        scored: list[tuple[float, str, str]] = []
        for index, body in blocks:
            best_sentence, best_score = "", 0.0
            for sentence in split_sentences(body):
                sentence_tokens = content_tokens(sentence)
                if not sentence_tokens:
                    continue
                score = len(query_tokens & sentence_tokens) / max(len(query_tokens), 1)
                if score > best_score:
                    best_sentence, best_score = sentence, score
            if best_sentence:
                scored.append((best_score, best_sentence, index))
        if not scored:
            return "根据现有资料无法回答该问题。"
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [item for item in scored[:3] if item[0] > 0]
        if not top:
            return "根据现有资料无法回答该问题。"
        lines = [f"{sentence}[{index}]" for _score, sentence, index in top]
        return "根据检索到的资料：\n" + "\n".join(f"- {line}" for line in lines)

    # —— 任务：Agent 工具决策 ——
    #
    # 真实模型在同一次调用里既可能发起工具调用，也可能直接给最终答复；
    # 这里用一个显式状态机复现这个语义：先检索，再按需补日期/计算，最后作答。

    def _task_decide(self, request: LLMRequest) -> str | list[ToolCall]:
        question = self._user_question(request)
        tool_names = {tool.name for tool in (request.tools or [])}
        already = {message.name for message in self._tool_messages(request) if message.name}
        used = len(already)

        if used == 0 and "knowledge_search" in tool_names:
            return [
                ToolCall(
                    name="knowledge_search",
                    arguments={"query": question, "top_k": 8},
                )
            ]
        if "datetime" in tool_names and "datetime" not in already and any(
            word in question for word in _DATE_WORDS
        ):
            return [ToolCall(name="datetime", arguments={"detail": "date"})]
        arithmetic = _ARITHMETIC.search(question)
        if arithmetic and "calculator" in tool_names and "calculator" not in already:
            return [ToolCall(name="calculator", arguments={"expression": arithmetic.group(0)})]
        return self._task_answer(request)

    # —— 任务：子问题分解 ——

    def _task_subquestions(self, request: LLMRequest) -> str:
        """按并列标记切分子问题，切不出多个就原样返回。"""

        question = self._user_question(request)
        parts = re.split(r"(?:以及|并且|同时|，另外|；|;|和(?=[^，。]{6,}))", question)
        candidates = [part.strip(" ，。？?") for part in parts if len(part.strip()) >= 8]
        if len(candidates) < 2:
            candidates = [question.strip()]
        import json

        return json.dumps({"questions": candidates[:4]}, ensure_ascii=False)

    # —— 任务：HyDE 假设性文档 ——

    def _task_hyde(self, request: LLMRequest) -> str:
        question = self._user_question(request)
        return (
            f"围绕「{question}」，相关资料一般会先给出定义与适用范围，"
            f"再说明关键机制、常见参数取值，最后列出实践中的注意事项与对比结论。"
        )

    # —— 任务：退步问题 ——

    def _task_step_back(self, request: LLMRequest) -> str:
        question = self._user_question(request)
        stripped = re.sub(r"[A-Za-z0-9][A-Za-z0-9\-_.:：]{1,}", "", question)
        stripped = re.sub(r"[《》「」“”\"'（）()]", "", stripped)
        stripped = re.sub(r"\s+", "", stripped)
        return f"{stripped or question}的基本原理、适用条件与通用做法是什么"

    # —— 辅助 ——

    @staticmethod
    def _split_evidence(text: str) -> list[tuple[str, str]]:
        """把形如 ``[1] 来源: ...`` 的证据块拆开。"""

        matches = list(_EVIDENCE_BLOCK.finditer(text))
        if not matches:
            return []
        blocks: list[tuple[str, str]] = []
        for position, match in enumerate(matches):
            start = match.end()
            end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
            blocks.append((match.group(1), text[start:end].strip()))
        return blocks


def default_client() -> LLMClient:
    """按配置返回合适的客户端：未配置 base_url 时退化为离线实现。"""

    from ..config import get_settings

    settings = get_settings().llm
    if settings.configured:
        from .openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
            timeout_seconds=settings.timeout_seconds,
            max_attempts=settings.max_attempts,
            temperature=settings.temperature,
        )
    return HeuristicLLM()


__all__ = [
    "FUNCTION_WORDS",
    "HeuristicLLM",
    "ScriptedLLM",
    "content_tokens",
    "default_client",
    "split_sentences",
    "tokenize",
]
