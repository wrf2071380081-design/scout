"""图片入库与 Milvus 后端测试（全部离线：不需要网络，也不需要 Milvus 服务）。

这一组测试的重点是**不需要真实服务也能验证的部分**：

- 图片抽取：请求体格式（写错不会报错、只会被静默忽略，所以必须断言）、
  体积上限、模型未配置时**拒绝**而不是把图片发给文本模型、空结果必须留警告；
- Milvus 后端：集合名派生、语料/向量器身份隔离、不可达时显式降级并留痕、
  ``strict`` 模式下拒绝降级；
- 后端等价性：Milvus 不可用时，结果必须与纯内存索引**完全一致**——
  这保证"降级"不会悄悄改变系统语义。

真实 Milvus 的联机等价性由 ``scripts/milvus_smoke.py`` 验证（需先起容器）。
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scout.config import MilvusSettings, get_settings
from scout.data import (
    GLMOCRExtractor,
    OllamaOCRExtractor,
    StubTextExtractor,
    VLMTextExtractor,
    build_extractor,
    extract_document,
    ingest,
    load_documents_with_extraction,
)
from scout.errors import ErrorCode, ProviderError, ScoutError
from scout.llm.scripted import HeuristicLLM
from scout.rag.embed import HashingEmbedder
from scout.rag.index import HybridIndex, RetrievalMode
from scout.rag.milvus_store import (
    MilvusDenseStore,
    MilvusHybridIndex,
    MilvusStatus,
    collection_name_for,
)
from scout.rag.pipeline import RAGPipeline, build_index

DOCS = [
    ("a.md", "云计算标准体系结构包括基础、技术、服务、应用、管理和安全六个部分。"),
    ("b.md", "低空经济标准体系重点围绕低空航空器、起降设施与运行服务展开。"),
]

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)

OCR_TEXT = "# 扫描件标题\n这是从图片里抽出来的正文，用于验证图片能走同一条入库流水线。"


# —— 图片抽取 ——


def _make_png(tmp_path: Path, name: str = "scan.png") -> Path:
    path = tmp_path / name
    path.write_bytes(PNG_1PX)
    return path


def test_vlm_payload_uses_openai_vision_shape(tmp_path: Path) -> None:
    """请求体必须符合 OpenAI 视觉消息格式。

    这类错误**不会报错**：网关收到格式不对的 content 只会当纯文本处理，
    于是模型根本没看到图片，返回一段凭空的描述。所以格式必须被断言。
    """

    path = _make_png(tmp_path)
    extractor = VLMTextExtractor(model="kimi-vl", base_url="https://example.com/v1")
    payload = extractor.build_payload(path)

    assert payload["model"] == "kimi-vl"
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # data URL 必须能还原回原字节
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == PNG_1PX


def test_vlm_refuses_when_model_missing(tmp_path: Path) -> None:
    """没有视觉模型就必须拒绝——发给文本模型不会失败，只会自信胡编。"""

    extractor = VLMTextExtractor(model="", base_url="https://example.com/v1")
    with pytest.raises(ProviderError) as excinfo:
        extractor.build_payload(_make_png(tmp_path))
    assert excinfo.value.code is ErrorCode.PROVIDER_UNAVAILABLE


def test_vlm_rejects_oversized_image(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    path.write_bytes(PNG_1PX + b"\x00" * 2_000_000)
    extractor = VLMTextExtractor(model="vl", base_url="https://x/v1", max_image_mb=1.0)
    with pytest.raises(ProviderError) as excinfo:
        extractor.build_payload(path)
    assert excinfo.value.code is ErrorCode.VALIDATION_FAILED


def test_vlm_parses_response_and_flags_unverified(tmp_path: Path) -> None:
    """抽取结果必须带"未经人工校验"的警告——否则会被当成可信语料直接入库。"""

    captured: dict[str, object] = {}

    def poster(url: str, headers: dict, payload: dict, timeout: float) -> dict:
        captured["url"] = url
        captured["headers"] = headers
        return {"choices": [{"message": {"content": OCR_TEXT}}]}

    extractor = VLMTextExtractor(
        model="vl", base_url="https://x/v1", api_key="k", poster=poster
    )
    result = extractor.extract(_make_png(tmp_path))

    assert result.text == OCR_TEXT
    assert captured["url"] == "https://x/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert any("未经人工校验" in w for w in result.warnings)


def test_vlm_warns_on_empty_result(tmp_path: Path) -> None:
    """空结果必须报警：它表现为"入库成功但什么都没进"。"""

    def poster(*_args: object) -> dict:
        return {"choices": [{"message": {"content": "   "}}]}

    extractor = VLMTextExtractor(model="vl", base_url="https://x/v1", poster=poster)
    result = extractor.extract(_make_png(tmp_path))
    assert any("空文本" in w for w in result.warnings)


def test_vlm_handles_chunked_content(tmp_path: Path) -> None:
    """有些网关把 content 返回成数组分片，必须能拼。"""

    def poster(*_args: object) -> dict:
        return {"choices": [{"message": {"content": [{"text": "第一段"}, {"text": "第二段"}]}}]}

    extractor = VLMTextExtractor(model="vl", base_url="https://x/v1", poster=poster)
    assert extractor.extract(_make_png(tmp_path)).text == "第一段第二段"


def test_vlm_distinguishes_reasoning_only_from_blind_model(tmp_path: Path) -> None:
    """**关键区分**：``content`` 为空时，是"预算被推理吃了"还是"没看到图"？

    这两种情况现象一样（都拿到空文本），修复动作却完全不同：
    前者调大 ``max_tokens``，后者要换模型。
    实测 kimi-k3 就是推理模型——第一次探测因为 ``max_tokens=32`` 被推理 token 吃光，
    ``content`` 为空，差点得出"它不支持图像"的相反结论。
    """

    def poster(*_args: object) -> dict:
        return {
            "choices": [
                {
                    "message": {"content": "", "reasoning_content": "图中写的是 7391，直接回答即可"},
                    "finish_reason": "length",
                }
            ]
        }

    extractor = VLMTextExtractor(model="vl", base_url="https://x/v1", poster=poster)
    with pytest.raises(ProviderError) as excinfo:
        extractor.extract(_make_png(tmp_path))
    message = str(excinfo.value)
    assert "只输出了推理内容" in message
    assert "max_tokens" in message
    # 必须可重试：这是配置问题，不是模型能力问题
    assert excinfo.value.retryable is True


def test_vlm_reasoning_is_never_used_as_extracted_text(tmp_path: Path) -> None:
    """思考过程绝不能当抽取结果——入库后会污染语料。

    ``reasoning_content`` 里混着"我应该逐字提取"这类元话语，
    把它当正文，等于往语料库塞了文档里没有的句子。
    """

    extractor = VLMTextExtractor(model="vl", base_url="https://x/v1")
    text = VLMTextExtractor.parse_response(
        {"choices": [{"message": {"content": "真正的正文", "reasoning_content": "元话语"}}]}
    )
    assert text == "真正的正文"
    assert "元话语" not in text


def test_vlm_payload_sets_max_tokens(tmp_path: Path) -> None:
    """必须显式给足 max_tokens：不给或给太小会被推理阶段吃光。"""

    extractor = VLMTextExtractor(model="vl", base_url="https://x/v1", max_tokens=2048)
    payload = extractor.build_payload(_make_png(tmp_path))
    assert payload["max_tokens"] == 2048


def test_image_without_extractor_is_rejected(tmp_path: Path) -> None:
    """图片路径必须显式失败，绝不能静默跳过。"""

    with pytest.raises(ProviderError) as excinfo:
        extract_document(_make_png(tmp_path), image_extractor=None)
    assert "未配置视觉抽取器" in str(excinfo.value)


def test_stub_extractor_enables_offline_image_ingest(tmp_path: Path) -> None:
    """离线替身让"图片入库"可被测试——否则这条链路没有回归保护。"""

    image = _make_png(tmp_path)
    outcome = load_documents_with_extraction([image], image_extractor=StubTextExtractor(
        mapping={"scan.png": OCR_TEXT}
    ))
    assert outcome.documents == [("scan.png", OCR_TEXT)]

    sections, report, _registry = ingest(outcome.documents)
    assert report.documents == 1
    assert report.kept >= 1
    assert any("扫描件标题" in section.text for section in sections)


def test_batch_extraction_records_failures_without_aborting(tmp_path: Path) -> None:
    """单篇失败不能拖垮整批，也不能静默跳过。"""

    good = tmp_path / "good.md"
    good.write_text("# 标题\n这是一段足够长的正常正文，用于验证批量抽取的容错行为。", encoding="utf-8")
    unsupported = tmp_path / "weird.xyz"
    unsupported.write_text("x", encoding="utf-8")

    outcome = load_documents_with_extraction([good, unsupported])
    assert len(outcome.documents) == 1
    assert len(outcome.skipped) == 1
    assert "weird.xyz" in outcome.skipped[0]


# —— GLM-OCR（专用文档解析模型）——


def test_glm_ocr_uses_layout_parsing_endpoint(tmp_path: Path) -> None:
    """GLM-OCR 走 /layout_parsing，不是 chat/completions。

    端点路径写错会得到 404 或"模型不存在"，容易被误判成"key 没权限"。
    """

    extractor = GLMOCRExtractor(api_key="k")
    assert extractor.endpoint() == "https://open.bigmodel.cn/api/paas/v4/layout_parsing"


def test_glm_ocr_payload_uses_data_uri(tmp_path: Path) -> None:
    """官方要求 file 字段是 URL 或 data URI——裸 base64 会被拒。"""

    extractor = GLMOCRExtractor(api_key="k")
    payload = extractor.build_payload(_make_png(tmp_path))
    assert payload["model"] == "glm-ocr"
    assert payload["file"].startswith("data:image/png;base64,")


def test_glm_ocr_parses_several_key_shapes() -> None:
    """字段名不做假设：md_results / markdown / 嵌套 data 都要能认。"""

    assert GLMOCRExtractor(api_key="k").parse_response({"md_results": "# 标题"}) == "# 标题"
    assert GLMOCRExtractor(api_key="k").parse_response({"markdown": "正文"}) == "正文"
    assert GLMOCRExtractor(api_key="k").parse_response({"data": {"md": "嵌套"}}) == "嵌套"
    assert GLMOCRExtractor(api_key="k").parse_response({"md": [{"text": "A"}, {"text": "B"}]}) == "A\nB"


def test_glm_ocr_reports_actual_keys_on_schema_miss() -> None:
    """**这是本类最有用的一条错误信息。**

    字段名猜错的表现是"抽取成功但内容为空"，而"内容为空"又会被上层当成
    "这篇文档没内容"，最终静默丢数据。所以必须把**真实键名**报出来——
    让这个问题从"要调试"变成"看一眼就改"。
    """

    extractor = GLMOCRExtractor(api_key="k")
    with pytest.raises(ProviderError) as excinfo:
        extractor.parse_response({"unexpected_field": "x", "another": "y"})
    message = str(excinfo.value)
    assert "找不到正文" in message
    assert "unexpected_field" in message
    assert excinfo.value.details.get("keys") == ["another", "unexpected_field"]


def test_glm_ocr_surfaces_business_error() -> None:
    """业务错误要如实抛出，不能当成"没有正文"。"""

    with pytest.raises(ProviderError) as excinfo:
        GLMOCRExtractor(api_key="k").parse_response({"error": {"code": "1210", "message": "参数错误"}})
    assert "业务错误" in str(excinfo.value)
    assert "1210" in str(excinfo.value)


def test_glm_ocr_end_to_end_with_injected_poster(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def poster(url: str, headers: dict, payload: dict, timeout: float) -> dict:
        captured["url"] = url
        captured["auth"] = headers.get("Authorization")
        return {"md_results": "| 容器 | 端口 |\n| --- | --- |\n| redis | 6379 |", "usage": {"total_tokens": 120}}

    extractor = GLMOCRExtractor(api_key="glm-key", poster=poster)
    result = extractor.extract(_make_png(tmp_path))
    assert "redis" in result.text
    assert captured["url"].endswith("/layout_parsing")
    assert captured["auth"] == "Bearer glm-key"
    assert extractor.usage_report()["total_tokens"] == 120


def test_build_extractor_selects_provider() -> None:
    """provider 决定用哪个抽取器；缺配置时返回 None 而不是偷偷换一个。"""

    vlm = build_extractor(vision_enabled=True, provider="vlm", model="kimi-k3")
    assert vlm is not None and isinstance(vlm, VLMTextExtractor)

    glm = build_extractor(vision_enabled=True, provider="glm-ocr", glm_api_key="k")
    assert isinstance(glm, GLMOCRExtractor)

    assert build_extractor(vision_enabled=True, provider="glm-ocr", glm_api_key="") is None

    local = build_extractor(
        vision_enabled=True, provider="ollama", model="glm-ocr", base_url="http://127.0.0.1:11434"
    )
    assert isinstance(local, OllamaOCRExtractor)


# —— 本地 Ollama 抽取 ——


def test_ollama_uses_native_generate_endpoint(tmp_path: Path) -> None:
    """本地走**原生** /api/generate，不是 OpenAI 兼容层。

    实测教训：Ollama 的 OpenAI 兼容端点在未显式指定上下文时会硬限在约 4096 token，
    长文档抽取的输出会在表格中间被截断、或撞上限后退化成复读。
    原生接口才能用 options.num_ctx / num_predict 控制。
    """

    extractor = OllamaOCRExtractor(host="http://127.0.0.1:11434")
    assert extractor.endpoint() == "http://127.0.0.1:11434/api/generate"
    # 误配成 OpenAI 风格地址时也要能纠正回来
    assert OllamaOCRExtractor(host="http://127.0.0.1:11434/v1").endpoint() == (
        "http://127.0.0.1:11434/api/generate"
    )


def test_ollama_payload_uses_raw_base64_and_explicit_context(tmp_path: Path) -> None:
    """原生接口要裸 base64（不带 data: 前缀），且必须显式给 num_ctx。"""

    extractor = OllamaOCRExtractor(model="glm-ocr", num_ctx=16384, num_predict=4096)
    payload = extractor.build_payload(_make_png(tmp_path))
    assert payload["model"] == "glm-ocr"
    assert isinstance(payload["images"], list) and len(payload["images"]) == 1
    assert not payload["images"][0].startswith("data:")  # 裸 base64
    assert payload["options"]["num_ctx"] == 16384
    assert payload["options"]["num_predict"] == 4096
    assert payload["stream"] is False


def test_ollama_trims_degenerate_repetition() -> None:
    """尾部复读必须裁掉。

    实测：本地 glm-ocr 在表格写完后不知道停，一路输出到 num_predict 上限，
    接着是几百行 ``` —**正文是对的，尾巴是垃圾**。
    不裁就会污染语料，检索时还会命中这些噪声。

    判据只看"尾部连续重复"，不假设文档结构，
    所以"表格后面还有正文"的文档不会被误伤。
    """

    good = "<table>\n<tr><td>redis</td><td>6379</td></tr>\n</table>"
    text = good + "\n" + "\n".join(["```"] * 200)
    trimmed, removed = OllamaOCRExtractor.trim_degenerate_tail(text, min_repeat=5)
    assert removed == 200
    assert trimmed == good
    assert "```" not in trimmed


def test_ollama_keeps_legitimate_content_after_table() -> None:
    """**不能误伤**：表格后面还有正文时，尾部不是复读就不裁。"""

    text = "<table>\n<tr><td>a</td></tr>\n</table>\n\n注意事项：本节自 2026 年起生效。"
    trimmed, removed = OllamaOCRExtractor.trim_degenerate_tail(text, min_repeat=5)
    assert removed == 0
    assert trimmed == text


def test_ollama_short_repeat_is_not_trimmed() -> None:
    """重复次数低于阈值不动手——正常的表格里也可能有连续相同单元格。"""

    text = "<table>\n" + "\n".join(["<tr><td>0%</td></tr>"] * 3) + "\n</table>"
    _trimmed, removed = OllamaOCRExtractor.trim_degenerate_tail(text, min_repeat=5)
    assert removed == 0


def test_ollama_warns_and_trims_on_context_exhaustion(tmp_path: Path) -> None:
    """撞上下文上限时要同时**留警告**和**裁退化**——只做一半都会掩盖问题。"""

    def poster(_url: str, _headers: dict, _payload: dict, _timeout: float) -> dict:
        return {
            "response": "<table><tr><td>redis</td></tr></table>\n" + "\n".join(["```"] * 50),
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 1340,
            "eval_count": 4096,
        }

    extractor = OllamaOCRExtractor(poster=poster)
    result = extractor.extract(_make_png(tmp_path))
    assert result.text.endswith("</table>")
    assert any("num_predict" in w for w in result.warnings)
    assert any("已裁掉尾部复读退化" in w for w in result.warnings)
    assert extractor.usage_report()["total_tokens"] == 5436
    # 本地推理零成本，这一点要在报告里说清楚
    assert "0" in extractor.usage_report()["cost"]


# —— Milvus 集合名与身份隔离 ——


def test_collection_name_isolates_corpus_and_embedder() -> None:
    """语料或向量器变了就换集合。

    复用集合并不会报错，只会返回"属于旧语料/旧语义空间"的结果——
    这是最难排查的一类脏数据事故。
    """

    a = collection_name_for("scout", "sha256:aaa", "bge-small-zh")
    b = collection_name_for("scout", "sha256:bbb", "bge-small-zh")
    c = collection_name_for("scout", "sha256:aaa", "hashing-256")
    assert len({a, b, c}) == 3


def test_collection_name_is_milvus_legal() -> None:
    """必须清洗 : 和 / ——否则报错信息指向"名字非法"，而真因是我们没处理冒号。"""

    name = collection_name_for("scout", "sha256:387ed85f42206bed", "BAAI/bge-small-zh-v1.5")
    assert ":" not in name and "/" not in name and "." not in name
    assert name[0].isalpha()
    assert len(name) <= 255


# —— Milvus 后端：降级与等价性 ——


def _settings_no_server() -> MilvusSettings:
    # 指向一个必然连不上的地址，用来验证降级路径（不依赖真实服务）
    return MilvusSettings(uri="http://127.0.0.1:1", timeout_seconds=0.5)


def _build_indices() -> tuple[HybridIndex, HybridIndex]:
    """两个后端必须由**同一个入口**构造。

    否则它们可能连分块方式都不一样，"两种后端的指标可对比"就成了空话。
    所以这里刻意都走 ``build_index``，只是 ``milvus`` 参数不同。
    """

    settings = replace(get_settings(), milvus=_settings_no_server())
    plain = build_index(DOCS, settings=settings, embedder=HashingEmbedder(dim=128), milvus=False)
    milvus_index = build_index(
        DOCS,
        settings=settings,
        embedder=HashingEmbedder(dim=128),
        milvus=True,
        corpus_key="test-corpus",
    )
    return plain, milvus_index


def test_milvus_unavailable_degrades_and_records_code() -> None:
    """不可达时必须显式降级并留痕，不能让人以为跑在 Milvus 上。"""

    _plain, milvus_index = _build_indices()
    status = milvus_index.milvus_status()
    assert status["synced"] is False
    assert status["degraded_code"] == "MILVUS_UNAVAILABLE_FALLBACK_MEMORY"
    assert status["error"], "降级必须带上失败原因"

    result = milvus_index.search("云计算标准体系包括哪几个部分", top_k=2)
    assert "MILVUS_UNAVAILABLE_FALLBACK_MEMORY" in result.degraded_code


def test_degraded_results_match_plain_memory_index() -> None:
    """降级后的结果必须与纯内存索引**完全一致**。

    这条断言是"换后端不改语义"的核心保障：把向量库接进来之后，
    不该出现"指标变了但说不清是后端还是别的原因"。
    """

    plain, milvus_index = _build_indices()
    for query in ("云计算标准体系包括哪几个部分", "低空经济标准体系的重点是什么"):
        for mode in (RetrievalMode.HYBRID, RetrievalMode.DENSE_ONLY, RetrievalMode.SPARSE_ONLY):
            expected = plain.search(query, top_k=3, mode=mode)
            actual = milvus_index.search(query, top_k=3, mode=mode)
            assert [h.chunk.chunk_id for h in actual.hits] == [
                h.chunk.chunk_id for h in expected.hits
            ], f"{mode.value} 模式下两个后端结果不一致"


def test_strict_mode_refuses_degradation() -> None:
    """出评测报告时必须用 strict：否则指标可能来自内存兜底而报告上看不出来。"""

    settings = replace(
        get_settings(), milvus=replace(_settings_no_server(), require_sync=True)
    )
    with pytest.raises(ScoutError) as excinfo:
        build_index(
            DOCS,
            settings=settings,
            embedder=HashingEmbedder(dim=64),
            milvus=True,
            corpus_key="strict-corpus",
            strict=True,
        )
    assert "拒绝降级" in str(excinfo.value)


def test_dense_store_reports_unavailable_status() -> None:
    store = MilvusDenseStore(uri="http://127.0.0.1:1", timeout=0.5)
    assert store.ping() is False
    assert store.last_error
    status = store.sync(
        corpus_key="x", embedder_name="e", dimension=4, items=[("c1", [0.1, 0.2, 0.3, 0.4])]
    )
    assert isinstance(status, MilvusStatus)
    assert status.available is False


def test_pipeline_works_on_milvus_backend_when_unavailable() -> None:
    """把 Milvus 版索引接到流水线上，降级后仍能完整跑通一次问答。"""

    _plain, milvus_index = _build_indices()
    pipeline = RAGPipeline(milvus_index, HeuristicLLM())
    result = pipeline.answer("云计算标准体系结构包括哪几个部分？")
    assert result.outcome in {"answered", "insufficient_evidence", "no_knowledge"}
    assert result.trace is not None
