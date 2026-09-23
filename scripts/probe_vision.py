"""探测网关上的模型是否**真的支持图像输入**。

**为什么需要这个脚本，而不是直接看模型名。**
"多模态"这件事不能靠名字猜，而且失败有四种形态，其中两种最危险：

1. **报错**（400 / model not support image）→ 明确，好办；
2. **静默忽略**（模型没看到图，却基于提示词自信地描述一通）→
   返回 200、有内容、看起来正常，只有核对才会发现是编的；
3. **看到图但答不出**（``finish_reason: length``）→ 见下面这条；
4. **推理模型把答案放进 ``reasoning_content``**，``content`` 为空。

**第 3、4 种我亲自踩过，代价是差点得出相反的错误结论。**
第一次探测 ``kimi-k3``：HTTP 200、``content`` 为空、暗号没出现，
我据此写下"模型没看到图，绝对不能用"——
直到 dump 原始 JSON 才看到真相：

```
"content": "",
"reasoning_content": "... The image shows \"7391\".",
"finish_reason": "length"
```

**它看得见，而且读对了。** 但它是推理模型：答案先进 ``reasoning_content``，
而我给的 ``max_tokens=32`` 被 29 个推理 token 吃光，``content`` 就没位置了。

所以本脚本两条防呆：
- ``max_tokens`` 给足（默认 256），别让推理阶段吃掉全部预算；
- **同时在 ``content`` 与 ``reasoning_content`` 里找暗号**，
  并区分"没看到图"与"看到了但没输出到 content"——**这两件事的修复动作完全不同**。

成本：一次调用，约 **200~400 token**。
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402

from scout.config import get_settings  # noqa: E402

# 探测用的"暗号"：一个不容易被蒙对的四位数
SECRET = os.environ.get("PROBE_SECRET", "7391")
OUT = Path(__file__).resolve().parents[1] / "reports" / "vision_probe.md"


def make_probe_image(path: Path) -> tuple[int, int]:
    """生成一张白底黑字、含 SECRET 的小图。

    刻意做小（约 240×80）：视觉 token 与图像面积大致成正比，
    探测不需要大图，**省下的都是真金白银**。
    """

    from PIL import Image, ImageDraw, ImageFont

    width, height = 240, 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = None
    for candidate in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 56)
            break
    if font is None:  # pragma: no cover - 取决于机器
        font = ImageFont.load_default()
    draw.text((20, 8), SECRET, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return width, height


def list_models(base_url: str, api_key: str, timeout: float = 20.0) -> list[str]:
    try:
        response = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        if not response.ok:
            return [f"(HTTP {response.status_code})"]
        data = response.json()
        return [str(item.get("id")) for item in (data.get("data") or [])]
    except Exception as exc:  # noqa: BLE001 - 探测失败本身就是结论
        return [f"(失败：{type(exc).__name__}: {exc})"]


def probe_vision(
    base_url: str,
    api_key: str,
    model: str,
    image_path: Path,
    *,
    timeout: float = 90.0,
) -> dict[str, object]:
    """发一次带图的请求，返回结构化结论。"""

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这张图里写的数字是什么？只回答数字本身。"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        # 给足预算：推理模型会先花 token 在 reasoning 上，
        # 给太小（如 32）会出现 content 为空 + finish_reason=length
        "max_tokens": int(os.environ.get("PROBE_MAX_TOKENS", "256")),
    }
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "transport_error", "detail": f"{type(exc).__name__}: {exc}"}

    if not response.ok:
        return {
            "status": "http_error",
            "http": response.status_code,
            "detail": response.text[:400],
        }
    try:
        data = response.json()
    except ValueError:
        return {"status": "non_json", "detail": response.text[:300]}

    choices = data.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    text = str(content or "").strip()
    # 推理模型：答案可能只出现在 reasoning_content 里（content 因预算耗尽为空）
    reasoning = str(message.get("reasoning_content") or "")
    finish = (choices[0].get("finish_reason") if choices else "") or ""
    usage = data.get("usage") or {}
    return {
        "status": "ok",
        "answer": text,
        "reasoning": reasoning,
        "finish_reason": finish,
        # 两个字段都看：否则会把"看到了但没输出到 content"误判成"没看到图"
        "sees_image": SECRET in text or SECRET in reasoning,
        "answer_in_content": SECRET in text,
        "usage": usage,
    }


def main() -> int:
    settings = get_settings()
    base_url = (settings.llm.base_url or "").rstrip("/")
    api_key = settings.llm.api_key
    model = os.environ.get("PROBE_MODEL") or settings.vision.model or settings.llm.model

    lines: list[str] = ["# 视觉能力探测报告", ""]
    lines.append(f"- 网关：`{base_url}`")
    lines.append(f"- 密钥：{'已配置（len=%d）' % len(api_key) if api_key else '**未配置**'}")
    lines.append(f"- 探测模型：`{model}`")
    lines.append(f"- 暗号：`{SECRET}`（只有真的看见图才可能答对）")
    lines.append("")

    if not base_url or not api_key:
        lines.append("## 结论：无法探测")
        lines.append("")
        lines.append("缺少 `SCOUT_LLM_BASE_URL` 或 `SCOUT_LLM_API_KEY`。")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return 1

    # 1) 网关暴露了哪些模型
    models = list_models(base_url, api_key)
    lines.append(f"## 网关可用模型（{len(models)} 个）")
    lines.append("")
    if len(models) <= 60:
        for name in models:
            lines.append(f"- `{name}`")
    else:
        for name in models[:60]:
            lines.append(f"- `{name}`")
        lines.append(f"- …（共 {len(models)} 个，只列前 60）")
    lines.append("")

    # 2) 真的能不能看图
    image_path = Path(os.environ.get("PROBE_IMAGE", str(Path(os.environ.get("TEMP", ".")) / "vision_probe.png")))
    width, height = make_probe_image(image_path)
    lines.append(f"## 视觉探测（图片 {width}×{height}）")
    lines.append("")
    result = probe_vision(base_url, api_key, model, image_path)
    status = result.get("status")

    ok = False
    if status == "ok":
        usage = result.get("usage") or {}
        lines.append(f"- `content`：`{result.get('answer') or '(空)'}`")
        lines.append(f"- `finish_reason`：`{result.get('finish_reason') or '(无)'}`")
        if result.get("reasoning"):
            lines.append(f"- `reasoning_content`：`{str(result['reasoning'])[:200]}`")
        lines.append(f"- 是否看到图（暗号出现）：**{'✅ 是' if result.get('sees_image') else '❌ 否'}**")
        lines.append(
            f"- 用量：prompt {usage.get('prompt_tokens')} ｜ completion {usage.get('completion_tokens')} "
            f"｜ total **{usage.get('total_tokens')}**"
        )
        lines.append("")
        if result.get("sees_image") and result.get("answer_in_content"):
            ok = True
            lines.append(f"## 结论：`{model}` **支持图像输入**，可直接用于图片抽取。")
        elif result.get("sees_image"):
            ok = True
            lines.append(
                f"## 结论：`{model}` **支持图像输入**——但答案是推理模型式的"
            )
            lines.append("")
            lines.append(
                "暗号出现在 `reasoning_content` 而 `content` 为空"
                f"（`finish_reason={result.get('finish_reason')}`）。"
                "**这不是「看不见图」，是「输出预算被推理阶段吃掉了」**——"
                "调大 `max_tokens` 即可。"
                "另外注意：解析响应时**不能只看 `message.content`**，"
                "否则会把这类模型的正常输出当成空结果。"
            )
        else:
            lines.append("## 结论：调用成功，但**未发现暗号**")
            lines.append("")
            lines.append(
                "可能是：① 真没看到图（静默忽略）；② 图太小/字体渲染问题导致读不出；"
                "③ 输出被截断。**先用 `PROBE_MAX_TOKENS=512` 重试一次**再下结论——"
                "本脚本初版就是因为 `max_tokens=32` 差点得出相反的错误结论。"
            )
    elif status == "http_error":
        code = result.get("http")
        lines.append(f"- HTTP {code}")
        lines.append("")
        lines.append("```")
        lines.append(str(result.get("detail")))
        lines.append("```")
        lines.append("")
        lines.append("## 结论：`%s` 不接受图像输入（或网关不支持该字段）" % model)
        lines.append("")
        lines.append("**这是好消息**：它明确报错了，比静默忽略安全得多。")
        lines.append("请从上面的模型清单里挑一个视觉模型，用 `PROBE_MODEL=<名字>` 重跑本脚本。")
    else:
        lines.append(f"- 状态：{status}")
        lines.append(f"- 详情：{result.get('detail')}")
        lines.append("")
        lines.append("## 结论：探测未完成（网络或网关问题）")

    lines.append("")
    lines.append("## 关于\"要不要纯视觉模型\"")
    lines.append("")
    lines.append(
        "**不需要。** 需要的是**能接受图像输入的多模态 LLM**（VLM）——"
        "它既看得懂图，又能输出结构化文本（我们要的是 Markdown 表格）。"
        "纯视觉模型（检测/分类/分割那类）反而不能直接产出可入库的文本。"
    )
    lines.append("")
    lines.append(
        "判断标准只有一条：**能否正确读取图中的可验证内容**。"
        "名字里带不带 vl / vision 都不算证据。"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告 -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
