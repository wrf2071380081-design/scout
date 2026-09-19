"""修复 fastembed 在 Windows 上的快照目录问题（可复用）。

背景：fastembed / huggingface_hub 在 Windows 上默认用符号链接
从 ``blobs/`` 生成 ``snapshots/``。没有开发者模式时链接创建失败，
于是留下"blobs 齐全、snapshots 缺失/不全"的状态，
ONNX 加载即报 NO_SUCHFILE（`Load model ... failed. File doesn't exist`）。

本脚本按 ``trees/<revision>.json`` 的 ``files`` 映射，
把 ``blobs/`` 复制（或从镜像补齐）成 ``snapshots/`` 下的正确文件名结构，
同时处理 LFS 文件（其 blob 文件名是 ``lfs_sha256`` 而不是 ``blob_id``）。

用法：
    python scripts/fix_fastembed_windows.py                # 修复缓存里所有模型
    python scripts/fix_fastembed_windows.py BAAI/bge-reranker-base  # 只修某一个

这是个一次性工具，但它值得进仓库——同一条坑在这个环境里已经踩过两次，
之后任何装 fastembed 的人都会在此受益。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

CACHE_ROOT = Path(os.path.expandvars(r"%TEMP%\fastembed_cache"))
MIRROR_BASE = "https://hf-mirror.com"


def resolve_blobs(tree_file: Path) -> list[tuple[str, dict]]:
    raw = json.loads(tree_file.read_text(encoding="utf-8"))
    files = raw.get("files") or {}
    return list(files.items())


def build_snapshot(repo_dir: Path, tree_file: Path, opener) -> tuple[int, list[str]]:
    """按 tree 映射把 blobs 复制成 snapshots 目录。返回写入文件数与日志行。"""

    revision = tree_file.stem
    snapshot_dir = repo_dir / "snapshots" / revision
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    count = 0
    repo_label = repo_dir.name.replace("models--", "").replace("--", "/", 1)

    for name, meta in resolve_blobs(tree_file):
        dest = snapshot_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        candidates = [meta.get("lfs_sha256"), meta.get("blob_id")]
        copied = False
        for candidate in candidates:
            if not candidate:
                continue
            source = repo_dir / "blobs" / candidate
            if source.exists() and source.stat().st_size > 0:
                shutil.copyfile(source, dest)
                count += 1
                copied = True
                logs.append(f"  [复制] {name}")
                break
        if copied:
            continue
        url = f"{MIRROR_BASE}/{repo_label}/resolve/main/{name}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "scout"})
            with opener.open(req, timeout=600) as response:
                dest.write_bytes(response.read())
            count += 1
            logs.append(f"  [下载] {name} ({dest.stat().st_size} bytes)")
        except Exception as exc:  # noqa: BLE001
            logs.append(f"  [跳过] {name}: {type(exc).__name__}: {str(exc)[:80]}")
    return count, logs


def repair(model_filter: str = "") -> list[str]:
    """修复缓存里的快照目录。``model_filter`` 为空时遍历全部模型。"""

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    lines: list[str] = [f"缓存根目录: {CACHE_ROOT}  存在={CACHE_ROOT.exists()}"]
    if not CACHE_ROOT.exists():
        lines.append("缓存不存在，无需修复。")
        return lines

    for repo_dir in sorted(CACHE_ROOT.glob("models--*")):
        label = repo_dir.name.replace("models--", "").replace("--", "/", 1)
        if model_filter and model_filter.lower() not in label.lower():
            continue
        trees = sorted((repo_dir / "trees").glob("*.json"))
        if not trees:
            lines.append(f"- {label}: 无 tree 文件，跳过")
            continue
        lines.append(f"- {label}")
        for tree in trees:
            count, logs = build_snapshot(repo_dir, tree, opener)
            lines.extend(logs)
            lines.append(f"  写入 {count} 个文件")
    return lines


def main() -> int:
    model_filter = sys.argv[1] if len(sys.argv) > 1 else ""
    lines = [f"修复 fastembed 快照（过滤: {model_filter or '全部'}）"]
    lines.extend(repair(model_filter))
    output = "\n".join(lines)
    print(output)
    out_path = Path(__file__).resolve().parents[1] / "reports" / "fastembed_fix.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(f"\n已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
