"""图片/扫描件的文本抽取：让图片也能走同一条入库流水线。

**这一层的设计目标只有一句话：让"图片"和"文本文件"在入库流水线里长得一样。**
抽取器把图片变成文本之后，后面全部复用既有能力——
版面还原、结构切分、两段去重、版本登记，一行都不用改。
这也是为什么它放在 ``data/`` 而不是新建一个"多模态模块"：
**多模态不是一条平行的链路，而是入库入口多了一种输入形态。**

三个组件：

- :class:`PlainTextExtractor`：文本类文件直接读（Markdown / txt / csv）。
- :class:`VLMTextExtractor`：图片走视觉模型（OpenAI 兼容的 image_url 消息格式），
  把扫描件、截图、图表里的文字抽出来。
- :class:`StubTextExtractor`：离线替身，让"图片入库"这条链路能在无网络、
  无 API Key 的环境下被测试——与项目里 LLM/向量器/重排器的双轨设计一致。

**为什么用 VLM 而不是传统 OCR（tesseract 等）。**
两条理由，第二条更重要：

1. 传统 OCR 要装二进制依赖（tesseract + 语言包），破坏"克隆即可离线跑"的前提；
   VLM 走的是已有的 OpenAI 兼容网关，零新增系统依赖。
2. **传统 OCR 只输出字符流，保住不版面**——表格会被拉平成一行行文字，
   而这恰好是我们最需要保住的东西（项目的数据层专门在做跨页表格与双栏还原）。
   VLM 可以直接输出 Markdown 表格，**版面信息在抽取阶段就被保住了**。

代价是成本与不确定性（有幻觉风险、延迟高）。所以边界写清楚：
**图片抽取结果必须先经人工或规则校验，才能进主评测语料；**
它默认只进"待复核区"，这一点由调用方决定，但这里的警告字段会如实标注。
"""

from __future__ import annotations

import base64
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from ..errors import ErrorCode, ProviderError

# 图片类后缀：交给视觉模型
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"})
# 文本类后缀：直接读。与 evaluation.runner 的 SUPPORTED_SUFFIXES 保持相容
TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".text", ".csv", ".json"})


@dataclass(slots=True)
class ExtractedText:
    """一次抽取的产出。"""

    source: str
    text: str
    extractor: str
    duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "extractor": self.extractor,
            "chars": len(self.text),
            "duration_ms": round(self.duration_ms, 1),
            "warnings": list(self.warnings),
        }


class TextExtractor(Protocol):
    """抽取器协议：把任意文件变成文本。"""

    name: str

    def extract(self, path: Path) -> ExtractedText:  # pragma: no cover - 协议声明
        ...


class PlainTextExtractor:
    """文本文件直读。"""

    name = "plain"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in TEXT_SUFFIXES

    def extract(self, path: Path) -> ExtractedText:
        started = time.perf_counter()
        text = path.read_text(encoding="utf-8", errors="replace")
        return ExtractedText(
            source=path.name,
            text=text,
            extractor=self.name,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )


class VLMTextExtractor:
    """用视觉模型把图片转成文本（OpenAI 兼容的 image_url 消息格式）。

    :param model: 视觉模型名。**不给模型名就直接报错**——把图片发给纯文本模型
        不会失败，只会得到一段自信的胡编内容，那比报错危险得多。
    :param poster: 可注入的 HTTP 投递函数（测试用）。签名 ``(url, headers, payload, timeout) -> dict``。
        注入点放在这里，是为了让"请求体长什么样"成为**可断言**的东西——
    图片消息的格式（``type: image_url`` + data URL）写错了不会报错，只会被静默忽略。
    """

    name = "vlm"

    def __init__(
        self,
        *,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        prompt: str = "",
        max_image_mb: float = 8.0,
        max_output_chars: int = 20000,
        max_tokens: int = 16384,
        max_image_side: int = 0,
        max_attempts: int = 2,
        timeout: float = 600.0,
        poster: Any = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.prompt = prompt or DEFAULT_VISION_PROMPT
        self.max_image_mb = max_image_mb
        self.max_output_chars = max_output_chars
        self.max_tokens = max_tokens
        self.max_image_side = max_image_side
        self.max_attempts = max(max_attempts, 1)
        self.timeout = timeout
        self.poster = poster
        self.calls = 0
        self.usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.last_resize = ""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_SUFFIXES

    def _downscale(self, path: Path, mime: str) -> tuple[str, str, str]:
        """按最长边限制降采样后重新编码。

        **为什么值得做。** 图片 token 与像素面积大致成正比，
        而多数视觉编码器在超过 ~1600px 后**不会获得额外信息收益**——
        多出来的像素只是在推高账单。降采样既省钱又常能降低推理开销。

        **但要小心**：降太多会让小字号文字糊掉，抽取质量反而下降。
        所以这里只做"超长边"限制，且把实际缩放比例**记进警告**——
        比例太狠时人能立刻看出来该调参数，而不是等核对抽取结果才发现问题。
        """

        try:
            import io

            from PIL import Image  # noqa: PLC0415 - 可选依赖
        except ImportError:
            return base64.b64encode(path.read_bytes()).decode("ascii"), mime, ""
        with Image.open(path) as image:
            width, height = image.size
            longest = max(width, height)
            if longest <= self.max_image_side:
                return base64.b64encode(path.read_bytes()).decode("ascii"), mime, ""
            scale = self.max_image_side / longest
            resized = image.convert("RGB").resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.LANCZOS,
            )
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=88, optimize=True)
        return (
            base64.b64encode(buffer.getvalue()).decode("ascii"),
            "image/jpeg",
            f"{width}×{height} → {resized.width}×{resized.height}（{self.max_image_side}px 上限）",
        )

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_SUFFIXES

    # —— 请求体构造（独立出来是为了可测）——

    def build_payload(self, path: Path) -> dict[str, Any]:
        """构造 OpenAI 兼容的视觉请求体。"""

        if not self.model:
            raise ProviderError(
                "未配置视觉模型（SCOUT_VLM_MODEL），拒绝把图片发给文本模型",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                operation="vision_extract",
            )
        raw = path.read_bytes()
        size_mb = len(raw) / (1024 * 1024)
        if size_mb > self.max_image_mb:
            raise ProviderError(
                f"图片过大（{size_mb:.1f}MB > {self.max_image_mb}MB），请先压缩或分片",
                code=ErrorCode.VALIDATION_FAILED,
                operation="vision_extract",
                details={"size_mb": round(size_mb, 2)},
            )
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(raw).decode("ascii")
        if self.max_image_side > 0:
            encoded, mime, resized = self._downscale(path, mime)
            if resized:
                self.last_resize = resized
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            # **必须显式给足 max_tokens。** 推理模型（如本网关的 kimi-k3）会先花 token
            # 输出 reasoning_content，再输出正文；不给或给太小会出现
            # "响应 200、content 为空"——看起来像模型没看到图，实际是预算被吃光。
            "max_tokens": self.max_tokens,
        }

    def endpoint(self) -> str:
        """完整的 chat/completions 地址。

        独立成一个方法，是为了让注入的 ``poster`` 拿到**与真实调用完全一致的 URL**——
        否则测试断言的是"某个中间态"，而线上拼错路径不会被发现。
        """

        if not self.base_url:
            raise ProviderError(
                "未配置视觉模型 base_url（可用 SCOUT_VLM_BASE_URL 或复用 SCOUT_LLM_BASE_URL）",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                operation="vision_extract",
            )
        trimmed = self.base_url
        for suffix in ("/chat/completions", "/"):
            if trimmed.endswith(suffix):
                trimmed = trimmed[: -len(suffix)]
        return f"{trimmed}/chat/completions"

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.endpoint()
        if self.poster is not None:
            return self.poster(url, self._headers(), payload, self.timeout)
        import requests  # noqa: PLC0415 - 仅在真实调用时导入

        response = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        if not response.ok:
            raise ProviderError(
                f"视觉模型返回 {response.status_code}",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                retryable=response.status_code >= 500,
                operation="vision_extract",
                details={"body_preview": response.text[:300]},
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(
                "视觉模型返回非 JSON",
                code=ErrorCode.PROVIDER_INVALID_RESPONSE,
                retryable=True,
                operation="vision_extract",
            ) from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def parse_response(data: dict[str, Any]) -> str:
        """从响应里取出抽取结果。

        **只取 ``message.content``，不取 ``reasoning_content``。**
        很多模型（含本网关的 ``kimi-k3``）是**推理模型**：先输出思考过程到
        ``reasoning_content``，再输出答案到 ``content``。
        思考过程里混着"我应该逐字提取"这类元话语，**把它当抽取结果会污染语料**——
        入库后检索会命中这些本不属于文档的句子。

        但 ``reasoning_content`` 有一个不可替代的用途：**诊断**。
        ``content`` 为空而 ``reasoning_content`` 非空，说明输出预算被推理阶段吃光了
        （``finish_reason=length``），而不是模型没看到图。
        **这两种情况的现象一样（都拿到空文本），修复动作却完全不同**：
        前者调大 ``max_tokens``，后者要换模型。所以这里把区别体现在抛出的错误里。
        """

        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(
                "视觉模型返回中没有 choices",
                code=ErrorCode.PROVIDER_INVALID_RESPONSE,
                operation="vision_extract",
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            # 有些网关返回分片数组，拼接其中的 text 部分
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        text = str(content or "")
        if text.strip():
            return text

        reasoning = str(message.get("reasoning_content") or "")
        finish = str(choices[0].get("finish_reason") or "")
        if reasoning.strip():
            raise ProviderError(
                "视觉模型只输出了推理内容、正文为空"
                f"（finish_reason={finish or '未知'}）——大概率是输出预算被推理阶段耗尽，"
                "请调大 max_tokens；若反复出现则说明该模型不适合做抽取",
                code=ErrorCode.PROVIDER_INVALID_RESPONSE,
                retryable=True,
                operation="vision_extract",
                details={"finish_reason": finish, "reasoning_chars": len(reasoning)},
            )
        if finish == "length":
            raise ProviderError(
                "视觉模型输出被长度限制截断且没有正文——请调大 max_tokens",
                code=ErrorCode.PROVIDER_INVALID_RESPONSE,
                retryable=True,
                operation="vision_extract",
                details={"finish_reason": finish},
            )
        return ""

    def extract(self, path: Path) -> ExtractedText:
        started = time.perf_counter()
        payload = self.build_payload(path)
        warnings: list[str] = []
        text = ""
        last_error: ProviderError | None = None
        # 重试一次。**实测网关会偶发 `finish_reason=content_filter`**（同一张图、
        # 同样的请求，再跑一次就成功），不是确定性拒绝。
        # 不重试会把偶发故障当成"这张图不能抽"，进而丢掉一篇文档。
        for attempt in range(1, self.max_attempts + 1):
            self.calls += 1
            try:
                data = self._post(payload)
                self._accumulate_usage(data)
                text = self.parse_response(data)
                break
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.max_attempts:
                    raise
                time.sleep(0.8 * attempt)
        if last_error is not None and not text:
            raise last_error

        if self.last_resize:
            warnings.append(f"已降采样：{self.last_resize}")
        if not text.strip():
            # 空结果必须报警：它会让这篇文档"入库成功但什么都没进"，静默丢数据
            warnings.append("视觉模型返回空文本——该文件未被有效入库")
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars]
            warnings.append(f"抽取结果超过 {self.max_output_chars} 字符，已截断")
        warnings.append("图片抽取结果未经人工校验，不应直接并入主评测语料")
        return ExtractedText(
            source=path.name,
            text=text,
            extractor=self.name,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            warnings=warnings,
        )

    def _accumulate_usage(self, data: dict[str, Any]) -> None:
        """累计 token 用量。

        **图片抽取是"按张计费"里最容易被低估的一项**：单张图的视觉 token
        可能上千，一批扫描件跑下来成本不低。不把用量记下来，
        "图片入库要花多少钱"就只能靠猜——而这正好是岗位职责里
        "成本控制"要回答的问题。
        """

        usage = data.get("usage") or {}
        self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        self.usage["total_tokens"] += int(usage.get("total_tokens") or 0)

    def usage_report(self) -> dict[str, Any]:
        total = self.usage["total_tokens"]
        return {
            "calls": self.calls,
            **dict(self.usage),
            "avg_tokens_per_image": round(total / self.calls, 1) if self.calls else 0.0,
        }


class GLMOCRExtractor:
    """智谱 GLM-OCR：**专用文档解析模型**，与通用 VLM 是两种不同的工具。

    **为什么它在这个项目里更合适（不是"另一个可选模型"，是更对的工具）：**

    | | 通用推理 VLM（kimi-k3） | GLM-OCR |
    |---|---|---|
    | 单张成本 | 实测 3.5k~12k token，波动 3.4× | **0.2 元/百万 token，1 元≈2000 张 A4** |
    | 单张耗时 | 实测 67~270 秒 | **约 1.5 秒**（0.67 张/秒） |
    | token 去向 | **86~96% 花在推理上**（OCR 不需要推理） | 全部用于识别 |
    | 表格 | 输出 Markdown（准，但靠通用能力） | **针对合并单元格 / 多层表头优化，直出 HTML/Markdown** |
    | 榜单 | 无 OCR 专项成绩 | **OmniDocBench V1.5 第一（94.62）** |

    更关键的一点：它的**两阶段流水线（版面分析 → 并行区域识别）**，
    与本项目数据层做的"双栏还原 / 跨页表格表头补写"是**同一个问题的两层解法**——
    它做页面级版面分析，我们做文本级后处理。**这不是拼凑，是能力互补。**

    .. note::
       **响应字段名不做假设。** 官方文档只说明"返回 Markdown 与 JSON 版面信息"，
       没有给出精确的字段路径，而这类接口的字段名各版本会变。
       所以 :meth:`parse_response` 用一组候选键去探，探不到就**把真实键名报出来**——
       猜错字段名的表现是"抽取成功但内容为空"，那是最难查的一种静默失败。
    """

    name = "glm-ocr"

    #: 常见的返回字段候选。不同版本可能用其中之一。
    TEXT_KEYS = ("md_results", "markdown", "md", "text", "content", "result", "data")

    def __init__(
        self,
        *,
        model: str = "glm-ocr",
        base_url: str = "https://open.bigmodel.cn",
        api_key: str = "",
        max_image_mb: float = 10.0,
        max_output_chars: int = 20000,
        max_attempts: int = 2,
        timeout: float = 300.0,
        poster: Any = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or "https://open.bigmodel.cn").rstrip("/")
        self.api_key = api_key
        self.max_image_mb = max_image_mb
        self.max_output_chars = max_output_chars
        self.max_attempts = max(max_attempts, 1)
        self.timeout = timeout
        self.poster = poster
        self.calls = 0
        self.usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.last_keys: list[str] = []

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_SUFFIXES

    def endpoint(self) -> str:
        """``/layout_parsing`` 是专门的文档解析端点，不是 chat/completions。"""

        trimmed = self.base_url
        if trimmed.endswith("/v4"):
            return f"{trimmed}/layout_parsing"
        return f"{trimmed}/api/paas/v4/layout_parsing"

    def build_payload(self, path: Path) -> dict[str, Any]:
        raw = path.read_bytes()
        size_mb = len(raw) / (1024 * 1024)
        if size_mb > self.max_image_mb:
            raise ProviderError(
                f"图片过大（{size_mb:.1f}MB > {self.max_image_mb}MB）",
                code=ErrorCode.VALIDATION_FAILED,
                operation="glm_ocr",
                details={"size_mb": round(size_mb, 2)},
            )
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(raw).decode("ascii")
        # 官方说明：上游接受 URL 或 data URI；裸 base64 必须包成 data URI
        return {"model": self.model, "file": f"data:{mime};base64,{encoded}"}

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.endpoint()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.poster is not None:
            return self.poster(url, headers, payload, self.timeout)
        import requests  # noqa: PLC0415

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if not response.ok:
            raise ProviderError(
                f"GLM-OCR 返回 {response.status_code}",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                retryable=response.status_code >= 500,
                operation="glm_ocr",
                details={"body_preview": response.text[:300]},
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(
                "GLM-OCR 返回非 JSON",
                code=ErrorCode.PROVIDER_INVALID_RESPONSE,
                retryable=True,
                operation="glm_ocr",
            ) from exc

    def parse_response(self, data: dict[str, Any]) -> str:
        """在候选键里找正文。

        找不到时**把真实键名报出来**——猜错字段名会表现为"抽取成功但内容为空"，
        而"内容为空"又会被上层当成"这篇文档没内容"，最终静默丢数据。
        把键名报出来，这个问题就从"要调试"变成"看一眼就改"。
        """

        self.last_keys = sorted(data.keys())
        for key in self.TEXT_KEYS:
            value = data.get(key)
            text = self._coerce_text(value)
            if text.strip():
                return text
        # 再探一层：有些版本把结果包在 data/results 里
        for outer in ("data", "result", "results"):
            inner = data.get(outer)
            if isinstance(inner, dict):
                for key in self.TEXT_KEYS:
                    text = self._coerce_text(inner.get(key))
                    if text.strip():
                        return text
        if data.get("error"):
            raise ProviderError(
                f"GLM-OCR 业务错误：{str(data.get('error'))[:300]}",
                code=ErrorCode.PROVIDER_INVALID_RESPONSE,
                operation="glm_ocr",
            )
        raise ProviderError(
            "GLM-OCR 响应里找不到正文——请按实际字段名更新 TEXT_KEYS"
            f"（实际顶层键：{self.last_keys}）",
            code=ErrorCode.PROVIDER_INVALID_RESPONSE,
            operation="glm_ocr",
            details={"keys": self.last_keys},
        )

    @staticmethod
    def _coerce_text(value: Any) -> str:
        """把各种可能的结构摊成字符串（字符串 / 列表 / 带 markdown 字段的字典）。"""

        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [GLMOCRExtractor._coerce_text(item) for item in value]
            return "\n".join(part for part in parts if part.strip())
        if isinstance(value, dict):
            for key in ("markdown", "md", "text", "content"):
                if key in value:
                    inner = GLMOCRExtractor._coerce_text(value[key])
                    if inner.strip():
                        return inner
        return ""

    def extract(self, path: Path) -> ExtractedText:
        started = time.perf_counter()
        payload = self.build_payload(path)
        text = ""
        last_error: ProviderError | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.calls += 1
            try:
                data = self._post(payload)
                usage = data.get("usage") or {}
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    self.usage[key] += int(usage.get(key) or 0)
                text = self.parse_response(data)
                break
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.max_attempts:
                    raise
                time.sleep(0.8 * attempt)
        if last_error is not None and not text:
            raise last_error

        warnings: list[str] = []
        if not text.strip():
            warnings.append("GLM-OCR 返回空文本——该文件未被有效入库")
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars]
            warnings.append(f"抽取结果超过 {self.max_output_chars} 字符，已截断")
        return ExtractedText(
            source=path.name,
            text=text,
            extractor=self.name,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            warnings=warnings,
        )

    def usage_report(self) -> dict[str, Any]:
        total = self.usage["total_tokens"]
        return {
            "calls": self.calls,
            **dict(self.usage),
            "avg_tokens_per_image": round(total / self.calls, 1) if self.calls else 0.0,
        }


class StubTextExtractor:
    """离线替身：不联网，按文件名返回预设文本。

    它存在的意义与项目里其他替身一致——**让"图片入库"这条链路可被测试**。
    否则这条路径只能靠手工试，等于没有回归保护。
    """

    name = "stub"

    def __init__(self, mapping: dict[str, str] | None = None, default: str = "") -> None:
        self.mapping = dict(mapping or {})
        self.default = default
        self.calls = 0

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_SUFFIXES

    def extract(self, path: Path) -> ExtractedText:
        self.calls += 1
        text = self.mapping.get(path.name, self.default)
        return ExtractedText(
            source=path.name,
            text=text,
            extractor=self.name,
            warnings=["离线替身抽取，仅供测试"] if text else ["离线替身没有任何可用文本"],
        )


DEFAULT_VISION_PROMPT = (
    "请把这张图片中的全部文字逐字提取出来，保持原有的阅读顺序与段落结构。"
    "表格请用 Markdown 表格还原，并保留表头。"
    "只输出内容本身，不要添加任何解释、总结或前后缀。"
)


# —— 路由与批量入口 ——


@dataclass(slots=True)
class ExtractionOutcome:
    """批量抽取的结果与统计。"""

    documents: list[tuple[str, str]] = field(default_factory=list)
    extracted: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": len(self.documents),
            "extracted": list(self.extracted),
            "skipped": list(self.skipped),
            "warnings": list(self.warnings),
        }


def build_extractor(
    *,
    vision_enabled: bool = False,
    provider: str = "vlm",
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    prompt: str = "",
    max_tokens: int = 16384,
    max_image_side: int = 0,
    glm_base_url: str = "https://open.bigmodel.cn",
    glm_api_key: str = "",
    glm_model: str = "glm-ocr",
    allow_stub: bool = False,
) -> TextExtractor | None:
    """按配置挑图片抽取器。**挑不到就返回 None，而不是偷偷用一个假的。**

    调用方拿到 None 时应当明确报错或跳过图片，而不是"以为图片被处理了"。
    这类静默失效会让"入库了多少内容"这件事无法核对。
    """

    if vision_enabled and provider == "glm-ocr":
        # 专用文档解析模型：不走 chat/completions，走 /layout_parsing。
        # 缺 key 时返回 None 而不是"回退到通用 VLM"——**静默换工具会让
        # 成本与质量都变成另一回事**，而报表上看不出来。
        if glm_api_key:
            return GLMOCRExtractor(model=glm_model, base_url=glm_base_url, api_key=glm_api_key)
        return None
    if vision_enabled and model:
        return VLMTextExtractor(
            model=model,
            base_url=base_url,
            api_key=api_key,
            prompt=prompt,
            max_tokens=max_tokens,
            max_image_side=max_image_side,
        )
    if allow_stub:
        return StubTextExtractor(default="")
    return None


def extract_document(
    path: Path,
    *,
    image_extractor: TextExtractor | None = None,
) -> ExtractedText:
    """单文件抽取：文本直读，图片走抽取器。"""

    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return PlainTextExtractor().extract(path)
    if suffix in IMAGE_SUFFIXES:
        if image_extractor is None:
            raise ProviderError(
                f"{path.name} 是图片，但未配置视觉抽取器（SCOUT_VISION_ENABLED / SCOUT_VLM_MODEL）",
                code=ErrorCode.VALIDATION_FAILED,
                operation="extract",
            )
        return image_extractor.extract(path)
    raise ProviderError(
        f"不支持的文件类型：{suffix or '(无后缀)'}",
        code=ErrorCode.VALIDATION_FAILED,
        operation="extract",
        details={"path": str(path)},
    )


def load_documents_with_extraction(
    paths: Iterable[Path],
    *,
    image_extractor: TextExtractor | None = None,
) -> ExtractionOutcome:
    """批量抽取，产出可以直接喂给 :func:`scout.data.pipeline.ingest` 的 ``[(name, text)]``。

    单个文件失败**不中断整批**，但会记进 ``skipped``——
    批量入库里最糟的行为是"一篇坏了整批没了"，
    第二糟的是"坏的那篇被静默跳过而没人知道"。
    """

    outcome = ExtractionOutcome()
    for path in paths:
        try:
            extracted = extract_document(path, image_extractor=image_extractor)
        except Exception as exc:  # noqa: BLE001 - 单篇失败降级为记录
            outcome.skipped.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if not extracted.text.strip():
            outcome.skipped.append(f"{path.name}: 抽取结果为空")
            continue
        outcome.documents.append((extracted.source, extracted.text))
        outcome.extracted.append(extracted.to_dict())
        outcome.warnings.extend(f"{extracted.source}: {w}" for w in extracted.warnings)
    return outcome


__all__ = [
    "DEFAULT_VISION_PROMPT",
    "ExtractedText",
    "ExtractionOutcome",
    "IMAGE_SUFFIXES",
    "PlainTextExtractor",
    "StubTextExtractor",
    "TEXT_SUFFIXES",
    "TextExtractor",
    "VLMTextExtractor",
    "build_extractor",
    "extract_document",
    "load_documents_with_extraction",
]
