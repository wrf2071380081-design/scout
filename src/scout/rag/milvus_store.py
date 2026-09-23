"""Milvus 稠密向量后端：把"内存索引"换成真实向量数据库，**只换一层**。

**为什么是"只覆写一个方法"，而不是写一个新的索引类。**
``HybridIndex`` 的检索主入口 :meth:`search` 负责的是一整套编排：
分通道召回、RRF 融合、按 id 回查、模式切换（hybrid / dense-only / sparse-only）。
其中真正依赖"向量存在哪里"的，只有稠密通道那一行——
:meth:`HybridIndex._dense_ranked`。

所以 :class:`MilvusHybridIndex` 只覆写这一个方法，其余全部继承：

- **正确的部分不会被重写**（RRF、候选池、去重、合并逻辑一行没动）；
- **两种后端的结果可直接对比**——因为差异只可能来自向量检索那一步，
  这让"换后端前后指标是否一致"成为一个可以验证的问题，而不是一句声称；
- 这正是 ``index.py`` 里那句承诺的字面兑现：
  *"生产环境应把 search() 换成向量数据库……接口形状保持不变，上层无需改动"*。

**为什么集合名由语料指纹派生。**
同一个 Milvus 实例上会并存多份语料（不同评测集、不同版本）。
集合名带上语料指纹，换语料就是换集合，**不会出现"用旧语料检索新问题"这种
最难排查的脏数据事故**——它会返回看似合理的结果。

**为什么不可达时必须显式降级而不能静默回退。**
"检索走了内存"与"检索走了 Milvus"是两种不同的系统状态：
前者数据在进程里、后者在外部服务里，延迟、容量、一致性语义都不同。
静默回退会让报告里的数字与真实部署不一致，且**没有任何痕迹**。
所以降级会写进 ``SearchResult.degraded_code``，让上层能把"这次是内存兜的"看见。
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from ..errors import ErrorCode, ScoutError
from .index import HybridIndex, SearchResult
from ..config import MilvusSettings


def _sanitize_collection_name(raw: str) -> str:
    """Milvus 集合名的合法化。

    约束：必须以字母开头，只能含字母/数字/下划线，长度 ≤ 255。
    语料指纹是 ``sha256:xxxx``，冒号非法——必须清洗，
    否则报错信息会以"名字非法"出现，而真实原因是"我们没处理冒号"。
    """

    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "c_" + cleaned
    return cleaned[:200]


def collection_name_for(prefix: str, corpus_key: str, embedder_name: str) -> str:
    """由「前缀 + 语料身份 + 向量器身份」派生集合名。

    必须带上向量器身份：换向量器意味着维度与语义空间都变了，
    复用同一个集合会拿到**维度不匹配或语义不兼容**的向量。
    这类错误在 Milvus 侧通常是"维度不符"报错，但如果维度恰好相同，
    就会静默返回错误结果——那才是最坏的情况。
    """

    digest = hashlib.sha256(f"{corpus_key}|{embedder_name}".encode("utf-8")).hexdigest()[:16]
    return _sanitize_collection_name(f"{prefix}_{digest}")


@dataclass(slots=True)
class MilvusStatus:
    available: bool
    collection: str = ""
    entities: int = 0
    dimension: int = 0
    error: str = ""
    backend: str = "milvus"

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "backend": self.backend,
            "collection": self.collection,
            "entities": self.entities,
            "dimension": self.dimension,
            "error": self.error,
        }


class MilvusDenseStore:
    """Milvus 稠密向量的最小封装。

    只做四件事：建集合、写入、检索、报状态。**不做融合、不做重排、不管理分块**——
    那些属于上层（HybridIndex 与流水线），放在这里会让这一层变得无法替换。
    """

    def __init__(
        self,
        *,
        uri: str = "http://127.0.0.1:19530",
        token: str = "",
        prefix: str = "scout",
        index_type: str = "HNSW",
        metric_type: str = "COSINE",
        timeout: float = 10.0,
        client: Any = None,
    ) -> None:
        self.uri = uri
        self.token = token
        self.prefix = prefix
        self.index_type = index_type
        self.metric_type = metric_type
        self.timeout = timeout
        self._client = client
        self._collection = ""
        self._dim = 0
        self.last_error = ""

    # —— 连接 ——

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from pymilvus import MilvusClient  # noqa: PLC0415 - 可选依赖，用到才导入
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise ScoutError(
                "未安装 pymilvus，无法使用 Milvus 后端。安装：pip install pymilvus",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
            ) from exc
        self._client = MilvusClient(uri=self.uri, token=self.token or "", timeout=self.timeout)
        return self._client

    @staticmethod
    def looks_like_connection_error(message: str) -> bool:
        """区分"连不上"与"连上了但用法有错"。

        两者的修复动作完全不同：前者去起容器，后者改代码。
        把 schema 冲突也报成"先把容器起来"，会让人白白折腾半天。
        """

        lowered = (message or "").lower()
        markers = (
            "fail connecting to server",
            "connection refused",
            "connection reset",
            "unavailable",
            "timed out",
            "timeout",
            "no route to host",
        )
        return any(marker in lowered for marker in markers)

    def ping(self) -> bool:
        """连通性探测。**失败只记录不抛**——调用方需要据此决定降级还是报错。"""

        try:
            client = self._ensure_client()
            client.list_collections()
            self.last_error = ""
            return True
        except Exception as exc:  # noqa: BLE001 - 探测失败不应该是异常路径
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    # —— 写入 ——

    def _create_collection(self, client: Any, name: str, dimension: int) -> None:
        """显式声明 schema——**不要依赖 create_collection 的默认字段类型**。

        ``MilvusClient.create_collection(dimension=...)`` 的便捷写法会把主键 ``id``
        默认建成 **int64**。我们的 chunk_id 是字符串（形如 ``doc-000#L3#12``），
        于是写入时报 ``DataNotMatchException: {id} field should be a int64``。

        **这个错误只有联机才会暴露**：离线单测用的是内存索引，根本走不到建表那一步。
        所以 schema 必须写死在这里，而不是交给默认值——
        默认值会随 pymilvus 版本变化，而这类变化的表现是"某天突然写不进去了"。
        """

        from pymilvus import DataType, MilvusClient  # noqa: PLC0415 - 可选依赖

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        # max_length 要够：chunk_id 是 "{document_id}#L{level}#{index}"，
        # 文档 id 里可能带较长的文件名派生串
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=int(dimension))
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=self.index_type,
            metric_type=self.metric_type,
            params={"M": 16, "efConstruction": 200},
        )
        client.create_collection(collection_name=name, schema=schema, index_params=index_params)

    def sync(
        self,
        *,
        corpus_key: str,
        embedder_name: str,
        dimension: int,
        items: Sequence[tuple[str, Sequence[float]]],
        recreate: bool = True,
    ) -> MilvusStatus:
        """全量同步。``recreate=True`` 时重建集合，保证与内存索引一致。

        选"重建"而不是"增量 upsert"，是因为评测场景要的是**幂等**：
        同一份语料跑两次必须得到完全一样的集合状态。
        增量 upsert 在删除场景下会残留孤儿向量，而孤儿向量会污染召回——
        它的表现是"召回了一些已经不存在的内容"，极难归因。
        """

        name = collection_name_for(self.prefix, corpus_key, embedder_name)
        try:
            client = self._ensure_client()
            if recreate and client.has_collection(name):
                client.drop_collection(name)
            if not client.has_collection(name):
                self._create_collection(client, name, int(dimension))
            if items:
                client.insert(
                    collection_name=name,
                    data=[{"id": identifier, "vector": list(vector)} for identifier, vector in items],
                )
                # flush 让刚写入的数据立刻可检索；不 flush 会出现
                # "刚同步完却搜不到"的间歇性失败，最难排查
                flush = getattr(client, "flush", None)
                if callable(flush):
                    flush(name)
            self._collection = name
            self._dim = int(dimension)
            self.last_error = ""
            return MilvusStatus(
                available=True,
                collection=name,
                entities=self.count(),
                dimension=int(dimension),
            )
        except Exception as exc:  # noqa: BLE001 - 统一转成状态而不是炸掉整条流水线
            self.last_error = f"{type(exc).__name__}: {exc}"
            return MilvusStatus(available=False, collection=name, error=self.last_error)

    def count(self) -> int:
        if not self._collection:
            return 0
        try:
            client = self._ensure_client()
            stats = client.get_collection_stats(self._collection) or {}
            return int(stats.get("row_count") or 0)
        except Exception:  # noqa: BLE001
            return 0

    # —— 检索 ——

    def search(self, vector: Sequence[float], limit: int) -> list[tuple[str, float]]:
        """返回 ``[(chunk_id, score)]``，按相似度降序。

        **分数要如实返回 Milvus 给的相似度**，不要在这里做归一化或换算——
        上层用它与稀疏通道做 RRF（基于排名，不依赖分数量纲），
        但在"只用稠密通道"的模式下分数会直接呈现给使用者。
        在这里偷偷改口径，会让两个后端的结果无法对比。
        """

        if not self._collection:
            raise ScoutError("Milvus 集合尚未同步", code=ErrorCode.PROVIDER_UNAVAILABLE)
        client = self._ensure_client()
        raw = client.search(
            collection_name=self._collection,
            data=[list(vector)],
            limit=max(1, int(limit)),
            output_fields=["id"],
            search_params={"params": {"ef": max(64, int(limit) * 4)}},
        )
        hits: list[tuple[str, float]] = []
        for group in raw or []:
            for hit in group:
                identifier = hit.get("id")
                if identifier is None:
                    entity = hit.get("entity") or {}
                    identifier = entity.get("id")
                distance = hit.get("distance", hit.get("score", 0.0))
                if identifier is not None:
                    hits.append((str(identifier), float(distance)))
        return hits

    def drop(self) -> bool:
        if not self._collection:
            return False
        try:
            client = self._ensure_client()
            if client.has_collection(self._collection):
                client.drop_collection(self._collection)
            return True
        except Exception:  # noqa: BLE001
            return False

    def status(self) -> MilvusStatus:
        ok = self.ping()
        return MilvusStatus(
            available=ok,
            collection=self._collection,
            entities=self.count() if ok else 0,
            dimension=self._dim,
            error=self.last_error,
        )


class MilvusHybridIndex(HybridIndex):
    """混合索引的 Milvus 版：**只把稠密通道换成 Milvus**。

    :param strict: ``True`` 时 Milvus 不可用即报错（不降级）。
        默认 ``False``：降级到内存并写入 ``degraded_code``。
        什么时候该用 ``True``？**当你需要证明"这套指标是跑在 Milvus 上的"**——
        比如出评测报告时。否则数字可能来自内存兜底，而报告上看不出来。
    """

    def __init__(
        self,
        *args: Any,
        milvus: MilvusSettings | None = None,
        corpus_key: str = "",
        strict: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.milvus_settings = milvus or MilvusSettings()
        self.corpus_key = corpus_key or f"anon-{time.time():.0f}"
        self.strict = self.milvus_settings.require_sync if strict is None else strict
        self.store = MilvusDenseStore(
            uri=self.milvus_settings.uri,
            token=self.milvus_settings.token,
            prefix=self.milvus_settings.collection_prefix,
            index_type=self.milvus_settings.index_type,
            metric_type=self.milvus_settings.metric_type,
            timeout=self.milvus_settings.timeout_seconds,
        )
        self._synced = False
        self._sync_status: MilvusStatus | None = None
        # 稠密通道的降级记录。search() 会把它写进 SearchResult.degraded_code，
        # 让"这次检索其实走的是内存"这件事有据可查。
        self.degraded_code = ""

    # —— 同步 ——

    def build(self) -> None:
        super().build()
        if self._synced:
            return
        status = self.store.sync(
            corpus_key=self.corpus_key,
            embedder_name=self.embedder.name,
            dimension=self.embedder.dim,
            items=[(chunk.chunk_id, self._vector_index[chunk.chunk_id]) for chunk in self.chunks],
        )
        self._sync_status = status
        if status.available:
            self._synced = True
            self.degraded_code = ""
            return
        self.degraded_code = "MILVUS_UNAVAILABLE_FALLBACK_MEMORY"
        if self.strict:
            raise ScoutError(
                f"Milvus 同步失败且 strict=True，拒绝降级：{status.error}",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                details={"uri": self.store.uri, "collection": status.collection},
            )

    def milvus_status(self) -> dict[str, Any]:
        if self._sync_status is None:
            return {"synced": False, "degraded_code": self.degraded_code}
        return {
            "synced": self._synced,
            "degraded_code": self.degraded_code,
            **self._sync_status.to_dict(),
        }

    # —— 唯一的检索覆写点 ——

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        mode: Any = None,
    ) -> SearchResult:
        """包一层，只为把"降级到内存"这件事写进结果。

        **不改编排逻辑**：融合、候选池、模式切换全部走父类。
        这里只做一个追加动作——把后端状态附加到 ``degraded_code``。
        没有这一步，报告里就看不出"这份指标其实跑在内存上"，
        而那是能让整份评测结论失效的信息。
        """

        kwargs: dict[str, Any] = {"top_k": top_k, "candidate_k": candidate_k}
        if mode is not None:
            kwargs["mode"] = mode
        result = super().search(query, **kwargs)
        if self.degraded_code:
            existing = result.degraded_code
            result.degraded_code = f"{existing}+{self.degraded_code}" if existing else self.degraded_code
            # 降级后本次结果的有效性要显式标注，避免被当成 Milvus 结果使用
            result.channel_sizes = {**result.channel_sizes, "dense_backend": 0}
        return result

    # —— 唯一的通道覆写点 ——

    def _dense_ranked(self, query: str, limit: int) -> list[tuple[str, float]]:
        """稠密召回：Milvus 优先，不可用则回退内存实现。"""

        self.build()
        if not self._synced:
            return super()._dense_ranked(query, limit)
        try:
            query_vector = self.embedder.embed([query])[0]
            hits = self.store.search(query_vector, limit)
        except Exception as exc:  # noqa: BLE001 - 检索期故障同样降级并留痕
            self.degraded_code = "MILVUS_SEARCH_FAILED_FALLBACK_MEMORY"
            self._synced = False
            self._sync_status = MilvusStatus(
                available=False, collection=self.store._collection,  # noqa: SLF001 - 同一模块
                error=f"{type(exc).__name__}: {exc}",
            )
            return super()._dense_ranked(query, limit)
        # 按 limit 截断：Milvus 已按相似度排序，但不同 metric 的排序方向一致，
        # 这里不再重排，保证"Milvus 的排序就是最终排序"，两端才可比。
        if not hits:
            # 集合为空时不要静默返回空结果——那会表现为"召回率突然归零"
            self.degraded_code = "MILVUS_EMPTY_RESULT_FALLBACK_MEMORY"
            return super()._dense_ranked(query, limit)
        return hits[:limit]


__all__ = [
    "MilvusDenseStore",
    "MilvusHybridIndex",
    "MilvusStatus",
    "collection_name_for",
]
