"""上下文压缩的测试。"""

from __future__ import annotations

import pytest

from scout.context import EvidenceCompressor, ObservationCompressor
from scout.rag.chunking import Chunk
from scout.rag.merge import EvidenceUnit


def _unit(chunk_id: str, text: str, score: float = 1.0) -> EvidenceUnit:
    return EvidenceUnit(
        chunk=Chunk(
            document_id="d",
            document_version="v1",
            filename="f.md",
            level=3,
            index=0,
            text=text,
            parent_chunk_id=None,
            chunk_id=chunk_id,
        ),
        score=score,
    )


REDIS_TEXT = (
    "第一段是无关的公司介绍。"
    "Redis 的高性能来自三点。"
    "第一，数据放在内存中，避免了磁盘 IO 的开销。"
    "第二，采用单线程模型处理命令，省去了上下文切换和锁竞争。"
    "第三，使用 IO 多路复用处理大量并发连接。"
)


def test_compressor_reduces_and_reports() -> None:
    """压缩要真减字符，并且账目完整——这是"成本工程"成立的前提。"""

    units = [_unit("v1::redis.md::l3::0", REDIS_TEXT)]
    compressed, report = EvidenceCompressor().compress("Redis 为什么这么快？", units)

    assert report.chars_after < report.chars_before
    assert 0 < report.ratio <= 1.0
    assert report.savings >= 0.0
    meta = report.to_meta()
    assert "compression_ratio" in meta


def test_compressor_keeps_relevant_and_digit_sentences() -> None:
    """压缩要保住与问题相关的句子，尤其是含数字的承重句。"""

    units = [_unit("v1::redis.md::l3::0", REDIS_TEXT)]
    compressed, _report = EvidenceCompressor(max_sentences_per_unit=2).compress(
        "Redis 单线程 为什么快", units
    )
    text = compressed[0].context_text
    # 含"单线程"的承重句应该被保住
    assert "单线程" in text
    # 纯无关句应该被丢掉
    assert "公司介绍" not in text


def test_compressor_preserves_citation() -> None:
    """出处不能丢：压缩后的文本必须还带 chunk_id，否则归因门控没法工作。"""

    units = [_unit("v1::redis.md::l3::0", REDIS_TEXT)]
    compressed, _report = EvidenceCompressor().compress("Redis", units)
    assert "v1::redis.md::l3::0" in compressed[0].context_text


def test_compressor_keeps_min_sentences_even_if_irrelevant() -> None:
    """哪怕是唯一的证据，也至少保住它的最好一句——

    否则充分性门控会因为"证据凭空消失"而误报。
    """

    units = [_unit("v1::x.md::l3::0", "这句话和问题完全无关。")]
    compressed, _report = EvidenceCompressor(min_sentences_per_unit=1).compress(
        "一个完全不相干的问题", units
    )
    assert len(compressed) == 1
    assert compressed[0].context_text.strip()


def test_observation_compressor_compresses_old_keeps_recent() -> None:
    """长运行时：旧的工具观测压成摘要，最近几条保留原文（成本与召回的取舍）。"""

    messages = [
        {"role": "system", "content": "你是助手"},
        *[
            {"role": "tool", "content": f"第{i}段很长的工具输出。" * 20}
            for i in range(1, 7)
        ],
    ]
    compressor = ObservationCompressor(keep_recent=2)
    compressed, report = compressor.compress_messages(messages)

    assert report.compressed_observations == 4  # 6 条工具输出里保留最近 2 条，压缩 4 条
    assert report.chars_after < report.chars_before
    # 最近两条保留原文
    tool_contents = [m["content"] for m in compressed if m["role"] == "tool"]
    assert "第6段" in tool_contents[-1]
    assert "第5段" in tool_contents[-2]
    # 被压缩的带摘要前缀
    assert any("已压缩" in content for content in tool_contents[:2])


def test_observation_compressor_no_tool_messages_is_noop() -> None:
    messages = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好"}]
    compressed, report = ObservationCompressor().compress_messages(messages)
    assert report.compressed_observations == 0
    assert compressed == messages
