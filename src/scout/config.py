"""运行配置。

所有可调参数集中在这里，从环境变量读取并带安全默认值。
**默认值必须让项目开箱可跑**——不配任何环境变量也能完成离线评测，
这样 CI 和面试官克隆后 ``pytest`` 就能直接通过。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Callable


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), minimum)
    except ValueError:
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(float(raw), minimum)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    """向量器配置。

    ``backend`` 的四种取值：

    - ``auto``（默认）：装了 fastembed 就用本地语义模型，否则退回哈希向量器
    - ``local``：强制本地语义模型（``BAAI/bge-small-zh-v1.5``，约 90MB，CPU 可跑）
    - ``openai``：走 OpenAI 兼容 ``/embeddings`` 接口
    - ``hashing``：强制离线哈希向量器（评测基线与单元测试）

    与 ``LLMSettings`` 的区别值得说明：LLM **可以**离线退化（启发式实现也能产出结构化输出），
    但向量器一旦退化，检索就从"语义"变成"词法"，这是**质量层面的降级而不是可用性层面的降级**。
    所以这里提供 :func:`scout.rag.embed.embedder_status` 把当前真实生效的后端暴露出来，
    并进入评测报告的 environment 段——**降级可以被看见，才不会被误当成真实结果。**
    """

    backend: str = "auto"
    model: str = "BAAI/bge-small-zh-v1.5"
    base_url: str = ""
    api_key: str = ""
    dim: int = 0
    cache_dir: str = ""


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """LLM 客户端配置。

    ``base_url`` 为空时自动退化为 :class:`~scout.llm.scripted.ScriptedLLM`，
    使整套流程可以完全离线运行与测试。
    """

    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    fast_model: str = "gpt-4o-mini"
    timeout_seconds: float = 60.0
    max_attempts: int = 2
    temperature: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.base_url)


@dataclass(frozen=True, slots=True)
class ChunkSettings:
    """三级父子分块参数（**字符级**，不是 token 级）。

    为什么用字符而不是 token：中英混排时 token 估算误差可达 ±40%，
    而分块边界稳定性对召回质量的影响远大于"精确对齐 token 预算"带来的收益。
    三级结构的作用是**解耦检索粒度与生成粒度**——用小块精确命中，用大块补全上下文。
    """

    level1_size: int = 2400
    level1_overlap: int = 400
    level2_size: int = 1600
    level2_overlap: int = 200
    level3_size: int = 800
    level3_overlap: int = 100


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """检索配置。"""

    top_k: int = 8
    candidate_multiplier: int = 3
    rrf_k: int = 60
    auto_merge_enabled: bool = True
    auto_merge_threshold: int = 2
    rerank_enabled: bool = True
    rerank_backend: str = "auto"
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_min_score: float = 0.0
    evidence_budget_chars: int = 12000
    grader_evidence_chars: int = 4800


@dataclass(frozen=True, slots=True)
class AgentSettings:
    """Agent 循环预算。

    预算是 Agent 可控性的第一道防线：max_steps 防止无限循环，
    deadline 防止单次请求无限拉长，tool_call_limit 防止工具滥用。
    """

    max_steps: int = 8
    max_tool_calls: int = 12
    max_repeated_tool_calls: int = 2
    deadline_seconds: float = 120.0
    max_context_chars: int = 12000
    response_reserve_chars: int = 2000

    @property
    def input_budget_chars(self) -> int:
        return max(self.max_context_chars - self.response_reserve_chars, 512)


@dataclass(frozen=True, slots=True)
class MemorySettings:
    """三层记忆配置。"""

    working_max_turns: int = 8
    episodic_max_items: int = 200
    semantic_max_items: int = 500
    default_ttl_seconds: int = 30 * 24 * 3600
    consolidation_min_repeats: int = 2


@dataclass(frozen=True, slots=True)
class VerifySettings:
    """验证与防护配置。

    ``sufficiency_min_coverage`` 与 ``sufficiency_min_focus_coverage`` 是两道独立门：
    前者看"材料整体对不对题"，后者看"问题真正在问的具体概念有没有找到"。
    只保留前者会让政策类语料上的不可回答问题被高频词放行；
    只保留后者会对长问题的部分覆盖过于苛刻。两者都要。
    """

    grounding_min_support: float = 0.6
    sufficiency_min_coverage: float = 0.5
    sufficiency_min_focus_coverage: float = 0.5
    sanitize_enabled: bool = True
    require_citations: bool = True


@dataclass(frozen=True, slots=True)
class Settings:
    """聚合配置。"""

    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    chunking: ChunkSettings = field(default_factory=ChunkSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    verify: VerifySettings = field(default_factory=VerifySettings)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            embedding=EmbeddingSettings(
                backend=_env_str("SCOUT_EMBED_BACKEND", "auto"),
                model=_env_str("SCOUT_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
                base_url=_env_str("SCOUT_EMBED_BASE_URL", ""),
                api_key=_env_str("SCOUT_EMBED_API_KEY", ""),
                dim=_env_int("SCOUT_EMBED_DIM", 0),
                cache_dir=_env_str("SCOUT_EMBED_CACHE_DIR", ""),
            ),
            llm=LLMSettings(
                base_url=_env_str("SCOUT_LLM_BASE_URL", ""),
                api_key=_env_str("SCOUT_LLM_API_KEY", ""),
                model=_env_str("SCOUT_LLM_MODEL", "gpt-4o-mini"),
                fast_model=_env_str("SCOUT_LLM_FAST_MODEL", "gpt-4o-mini"),
                timeout_seconds=_env_float("SCOUT_LLM_TIMEOUT_SECONDS", 60.0, minimum=0.1),
                max_attempts=_env_int("SCOUT_LLM_MAX_ATTEMPTS", 2),
                temperature=_env_float("SCOUT_LLM_TEMPERATURE", 0.0),
            ),
            chunking=ChunkSettings(
                level1_size=_env_int("SCOUT_CHUNK_L1_SIZE", 2400),
                level1_overlap=_env_int("SCOUT_CHUNK_L1_OVERLAP", 400),
                level2_size=_env_int("SCOUT_CHUNK_L2_SIZE", 1600),
                level2_overlap=_env_int("SCOUT_CHUNK_L2_OVERLAP", 200),
                level3_size=_env_int("SCOUT_CHUNK_L3_SIZE", 800),
                level3_overlap=_env_int("SCOUT_CHUNK_L3_OVERLAP", 100),
            ),
            retrieval=RetrievalSettings(
                top_k=_env_int("SCOUT_RETRIEVAL_TOP_K", 8),
                candidate_multiplier=_env_int("SCOUT_RETRIEVAL_CANDIDATE_MULTIPLIER", 3),
                rrf_k=_env_int("SCOUT_RRF_K", 60),
                auto_merge_enabled=_env_bool("SCOUT_AUTO_MERGE_ENABLED", True),
                auto_merge_threshold=_env_int("SCOUT_AUTO_MERGE_THRESHOLD", 2),
                rerank_enabled=_env_bool("SCOUT_RERANK_ENABLED", True),
                rerank_backend=_env_str("SCOUT_RERANK_BACKEND", "auto"),
                rerank_model=_env_str("SCOUT_RERANK_MODEL", "BAAI/bge-reranker-base"),
                rerank_min_score=_env_float("SCOUT_RERANK_MIN_SCORE", 0.0),
                evidence_budget_chars=_env_int("SCOUT_EVIDENCE_BUDGET_CHARS", 12000),
                grader_evidence_chars=_env_int("SCOUT_GRADER_EVIDENCE_CHARS", 4800),
            ),
            agent=AgentSettings(
                max_steps=_env_int("SCOUT_AGENT_MAX_STEPS", 8),
                max_tool_calls=_env_int("SCOUT_AGENT_MAX_TOOL_CALLS", 12),
                max_repeated_tool_calls=_env_int("SCOUT_AGENT_MAX_REPEATED_TOOL_CALLS", 2),
                deadline_seconds=_env_float("SCOUT_AGENT_DEADLINE_SECONDS", 120.0, minimum=1.0),
                max_context_chars=_env_int("SCOUT_AGENT_MAX_CONTEXT_CHARS", 12000),
                response_reserve_chars=_env_int("SCOUT_AGENT_RESPONSE_RESERVE_CHARS", 2000),
            ),
            memory=MemorySettings(
                working_max_turns=_env_int("SCOUT_MEMORY_WORKING_TURNS", 8),
                episodic_max_items=_env_int("SCOUT_MEMORY_EPISODIC_MAX", 200),
                semantic_max_items=_env_int("SCOUT_MEMORY_SEMANTIC_MAX", 500),
                default_ttl_seconds=_env_int("SCOUT_MEMORY_TTL_SECONDS", 30 * 24 * 3600),
                consolidation_min_repeats=_env_int("SCOUT_MEMORY_CONSOLIDATE_MIN", 2),
            ),
            verify=VerifySettings(
                grounding_min_support=_env_float("SCOUT_GROUNDING_MIN_SUPPORT", 0.6),
                sufficiency_min_coverage=_env_float("SCOUT_SUFFICIENCY_MIN_COVERAGE", 0.5),
                sufficiency_min_focus_coverage=_env_float(
                    "SCOUT_SUFFICIENCY_MIN_FOCUS_COVERAGE", 0.5
                ),
                sanitize_enabled=_env_bool("SCOUT_SANITIZE_ENABLED", True),
                require_citations=_env_bool("SCOUT_REQUIRE_CITATIONS", True),
            ),
        )

    def with_overrides(self, **overrides: object) -> Settings:
        """按需覆盖子配置，用于消融实验。"""

        updated = self
        for key, value in overrides.items():
            if not hasattr(updated, key):
                raise AttributeError(f"unknown settings section: {key}")
            if value is not None:
                updated = replace(updated, **{key: value})
        return updated


_SETTINGS_FACTORY: Callable[[], Settings] = Settings.from_env
_cached: Settings | None = None


def get_settings() -> Settings:
    """读取进程级配置（首次调用时从环境变量加载并缓存）。"""

    global _cached
    if _cached is None:
        _cached = _SETTINGS_FACTORY()
    return _cached


def reset_settings_cache() -> None:
    """清空配置缓存。测试与消融实验在改环境变量后需要调用。"""

    global _cached
    _cached = None


__all__ = [
    "AgentSettings",
    "ChunkSettings",
    "LLMSettings",
    "MemorySettings",
    "RetrievalSettings",
    "Settings",
    "VerifySettings",
    "get_settings",
    "reset_settings_cache",
]
