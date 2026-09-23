"""图片抽取的「成本 vs 质量」对照实验。

**为什么这个实验值得做。**
图片抽取是**按张计费**的，而"降采样能省多少 token、会不会伤抽取质量"
不能靠感觉。但只看 token 数也不够——便宜但抽错的抽取，比贵而准确的更贵：
错的内容进了语料库，后面所有检索都会命中它。

所以本实验同时测两件事，并**用 ground truth 客观打分**：

1. **成本**：prompt / reasoning / content 三段 token 分别多少；
2. **质量**：抽取结果命中了多少个预定义"关键事实"（容器名、镜像名、端口）。

**为什么要拆出 reasoning token。**
推理模型（如本网关的 kimi-k3）会先把思考过程写进 ``reasoning_content``，
再输出正文。做 OCR 这类"看到什么写什么"的任务时，
**大量推理 token 其实是纯开销**——把它们单独统计出来，
才能量化"这个模型做抽取的经济性到底如何"。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scout.config import get_settings  # noqa: E402
from scout.data.extract import VLMTextExtractor  # noqa: E402

# ground truth：Docker 截图里应当被抽出来的关键事实
KEY_FACTS = [
    "supermew",
    "milvus-attu",
    "milvus-standalone",
    "postgres",
    "redis",
    "milvus-etcd",
    "milvus-minio",
    "zilliz/attu",
    "milvusdb/milvus",
    "postgres:15",
    "redis:7-alpine",
    "coreos/etcd",
    "minio/minio",
    "8080",
    "19530",
    "5432",
    "6379",
    "9000",
]


@dataclass(slots=True)
class Arm:
    """一个实验臂（一组参数）。

    每个臂自带 ``base_url`` / ``model`` / ``provider``——
    因为本实验的核心是**横向比较不同工具**：云端通用推理 VLM 与本地专用 OCR
    根本不是同一个东西（成本模型、延迟、质量都不同），
    共用一个网关配置会让对照失去意义。
    """

    name: str
    max_image_side: int
    max_tokens: int
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    provider: str = "vlm"
    text: str = ""
    error: str = ""
    duration_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_tokens: int = 0
    hits: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        total = len(self.hits) + len(self.misses)
        return len(self.hits) / total if total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "max_image_side": self.max_image_side,
            "max_tokens": self.max_tokens,
            "chars": len(self.text),
            "fact_recall": round(self.score, 3),
            "hits": len(self.hits),
            "misses": self.misses,
            "usage": dict(self.usage),
            "reasoning_tokens": self.reasoning_tokens,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error,
        }


def base_url_model(arm: "Arm", settings: object) -> tuple[str, str]:
    """臂未指定时回落到全局配置。"""

    base = arm.base_url or settings.vision.base_url or settings.llm.base_url
    model = arm.model or settings.vision.model
    return base, model


def run_arm(
    image: Path,
    *,
    arm: Arm,
    settings: object,
) -> Arm:
    base, model = base_url_model(arm, settings)
    extractor = VLMTextExtractor(
        model=model,
        base_url=base,
        api_key=arm.api_key or settings.vision.api_key or settings.llm.api_key,
        prompt=settings.vision.prompt,
        max_tokens=arm.max_tokens,
        max_image_side=arm.max_image_side,
        timeout=900.0,  # 本地 CPU 推理可能很慢
    )
    # 直接调 _post 拿原始响应，才能统计 reasoning token——
    # 这是本实验的核心指标之一，走 extract() 会丢掉它。
    started = time.perf_counter()
    try:
        payload = extractor.build_payload(image)
        data = extractor._post(payload)  # noqa: SLF001 - 实验脚本，需要原始响应
        arm.duration_ms = (time.perf_counter() - started) * 1000.0
        usage = data.get("usage") or {}
        arm.usage = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }
        details = (usage.get("completion_tokens_details") or {})
        arm.reasoning_tokens = int(details.get("reasoning_tokens") or 0)
        arm.text = extractor.parse_response(data)
        if extractor.last_resize:
            arm.error = f"(已降采样 {extractor.last_resize})"
    except Exception as exc:  # noqa: BLE001 - 失败也是一种结果，必须记录
        arm.duration_ms = (time.perf_counter() - started) * 1000.0
        arm.error = f"{type(exc).__name__}: {str(exc)[:300]}"
        arm.usage = dict(extractor.usage)

    lowered = arm.text.lower()
    arm.hits = [fact for fact in KEY_FACTS if fact.lower() in lowered]
    arm.misses = [fact for fact in KEY_FACTS if fact.lower() not in lowered]
    return arm


def main() -> int:
    settings = get_settings()
    image = Path(os.environ["VISION_TEST_IMAGE"])
    out_dir = Path(__file__).resolve().parents[1] / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_arms = [
        Arm("A 原图", max_image_side=0, max_tokens=16384),
        Arm("B 降采样1600px", max_image_side=1600, max_tokens=16384),
    ]
    # VISION_ARMS=A / B / AB —— 便于单独复现某一个臂，避免重复烧钱
    wanted = os.environ.get("VISION_ARMS", "AB").upper()
    arms = [arm for arm in all_arms if arm.name.split()[0] in wanted] or all_arms

    lines = ["# 图片抽取：成本 vs 质量对照", ""]
    lines.append(f"- 图片：`{image.name}`")
    from PIL import Image

    with Image.open(image) as im:
        lines.append(f"- 原始尺寸：{im.width}×{im.height}（{image.stat().st_size / 1024:.0f} KB）")
    lines.append(f"- 模型：`{settings.vision.model}`")
    lines.append(f"- ground truth 关键事实：**{len(KEY_FACTS)}** 项（容器名 / 镜像名 / 端口）")
    lines.append("")

    results: list[dict[str, object]] = []
    for arm in arms:
        print(
            f"跑 {arm.name} …（模型 {base_url_model(arm, settings)[1]}"
            f" @ {base_url_model(arm, settings)[0]}，max_tokens={arm.max_tokens}，"
            f"降采样={arm.max_image_side or '关'}）"
        )
        done = run_arm(image, arm=arm, settings=settings)
        results.append(done.to_dict())
        print(
            f"  → 事实命中 {len(done.hits)}/{len(KEY_FACTS)}｜"
            f"total {done.usage.get('total_tokens', 0)} token"
            f"（reasoning {done.reasoning_tokens}）｜{done.duration_ms / 1000:.1f}s"
            + (f"\n  ! {done.error}" if done.error else "")
        )
        (out_dir / f"vision_arm_{done.name.split()[0]}.txt").write_text(done.text, encoding="utf-8")

    lines.append("## 对照结果")
    lines.append("")
    lines.append("| 臂 | 降采样 | prompt | completion | 其中 reasoning | total | 事实命中 | 耗时 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for item in results:
        usage = item["usage"]  # type: ignore[index]
        lines.append(
            f"| {item['name']} | {item['max_image_side'] or '关'} "
            f"| {usage.get('prompt_tokens', 0)} | {usage.get('completion_tokens', 0)} "
            f"| {item['reasoning_tokens']} | **{usage.get('total_tokens', 0)}** "
            f"| **{item['hits']}/{len(KEY_FACTS)}**（{item['fact_recall']:.0%}） "
            f"| {item['duration_ms'] / 1000:.1f}s |"
        )
    lines.append("")

    if len(results) < 2:
        lines.append("## 结论（单臂运行）")
        lines.append("")
        only = results[0]
        u = only["usage"]  # type: ignore[index]
        lines.append(
            f"- 事实命中 **{only['hits']}/{len(KEY_FACTS)}**（{only['fact_recall']:.0%}）"
            f"｜total **{u.get('total_tokens', 0)}** token"
            f"（其中 reasoning {only['reasoning_tokens']}）"
            f"｜{only['duration_ms'] / 1000:.1f}s"
        )
        if only["error"]:
            lines.append(f"- ⚠️ 该臂未完成：`{only['error']}`")
        completion = int(u.get("completion_tokens", 0) or 0)
        if only["reasoning_tokens"] and completion:
            lines.append(
                f"- **completion 里有 {only['reasoning_tokens'] / completion:.0%} 花在 reasoning 上**"
                f"（{only['reasoning_tokens']} / {completion}）——"
                "OCR 是'看到什么写什么'的任务，这些推理 token 基本是纯开销。"
            )
        lines.append("")
        lines.append("## 未命中的事实")
        lines.append("")
        lines.append(f"- {only['misses'] if only['misses'] else '（全部命中）'}")
        lines.append("")
        report = out_dir / "vision_cost_quality.md"
        report.write_text("\n".join(lines), encoding="utf-8")
        (out_dir / "vision_cost_quality.json").write_text(
            json.dumps({"image": str(image), "facts": KEY_FACTS, "arms": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print()
        print("\n".join(lines))
        print(f"\n报告 -> {report}")
        return 0

    a, b = results[0], results[1]
    a_usage, b_usage = a["usage"], b["usage"]  # type: ignore[index]
    a_tok = a_usage.get("total_tokens", 0)
    b_tok = b_usage.get("total_tokens", 0)
    a_done = not a["error"] or str(a["error"]).startswith("(已降采样")
    lines.append("## 结论")
    lines.append("")
    if b_tok and a_tok:
        saved = 1 - b_tok / a_tok
        lines.append(
            f"- **降采样省了 {saved:.0%} 的 token**（{a_tok} → {b_tok}）"
        )
    lines.append(
        f"- 质量：原图命中 {a['hits']}/{len(KEY_FACTS)}"
        + ("（该臂调用失败，命中数不具可比性）" if not a_done else "")
        + f"，降采样命中 {b['hits']}/{len(KEY_FACTS)}"
    )
    if not a_done:
        lines.append(f"- ⚠️ 原图那臂**未能完成**：`{a['error']}`")
    if a["hits"] == b["hits"]:
        lines.append("- **质量没有损失** → 降采样是纯赚，应当默认开启。")
    elif b["hits"] < a["hits"]:
        lines.append(
            f"- **质量有损失**（少命中 {a['hits'] - b['hits']} 项）→ "
            "降采样不能无条件开，要看文档字号；小字号文档应保留原图。"
        )
    else:
        lines.append("- 降采样反而命中更多——说明原图太大时模型反而抓不住细节。")
    lines.append("")
    # 失败臂可能没有 completion_tokens；用 .get 保护，别让报告因为一个缺失字段崩掉
    a_completion = int(a_usage.get("completion_tokens", 0) or 0)
    if a["reasoning_tokens"] and a_completion:
        share = a["reasoning_tokens"] / max(a_completion, 1)
        lines.append(
            f"- ⚠️ **原图那次的 completion 里有 {share:.0%} 花在 reasoning 上**"
            f"（{a['reasoning_tokens']} / {a['completion_tokens']}）。"
            "OCR 是'看到什么写什么'的任务，这些推理 token 基本是纯开销——"
            "**换一个非推理模型做抽取，成本可能直接降到零头。**"
        )
    lines.append("")
    lines.append("## 未命中的事实")
    lines.append("")
    for item in results:
        misses = item["misses"]  # type: ignore[index]
        lines.append(f"- **{item['name']}**：{misses if misses else '（全部命中）'}")
    lines.append("")

    report = out_dir / "vision_cost_quality.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "vision_cost_quality.json").write_text(
        json.dumps({"image": str(image), "facts": KEY_FACTS, "arms": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print("\n".join(lines))
    print(f"\n报告 -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
