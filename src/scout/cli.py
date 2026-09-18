"""命令行入口。

设计原则：**每条命令都必须在无 API Key、无外部服务的环境下可运行**。
这不是为了省事，而是为了让"可复现"成立——一个需要先申请密钥才能跑的评测脚本，
别人（以及未来的你）不会去跑第二遍，于是它就不再是回归测试，只是一次性脚本。

用法::

    scout info
    scout demo
    scout eval validate --dataset evals/longdoc_v1.json
    scout eval run --dataset evals/longdoc_v1.json --corpus datasets/longdoc-gold
    scout eval run --mode agent --out reports/agent.json --markdown reports/agent.md
    scout eval ablation --dataset evals/longdoc_v1.json --corpus datasets/longdoc-gold --outdir reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import scout

from .agent.loop import ToolAgent
from .config import get_settings
from .evaluation.dataset import load_dataset
from .evaluation.report import (
    render_ablation_table,
    render_failure_taxonomy,
    render_report,
)
from .evaluation.runner import (
    RunConfig,
    default_ablation_configs,
    load_corpus,
    run_ablation,
    run_evaluation,
)
from .hitl import (
    HumanDecision,
    ResumableAgent,
    RunStatus,
    TimeoutPolicy,
    build_default_policy,
)
from .llm.base import LLMResponse, ToolCall
from .llm.scripted import HeuristicLLM, ScriptedLLM, default_client
from .memory.store import LayeredMemory
from .orchestrator.healing import SelfHealingOrchestrator
from .rag.pipeline import RAGPipeline, build_index
from .tools.actions import build_action_tools
from .tools.builtin import build_default_tools
from .tools.knowledge import KnowledgeSearchTool
from .tools.registry import ToolRegistry

DEFAULT_DATASET = "evals/longdoc_v1.json"
DEFAULT_CORPUS = "datasets/longdoc-gold"


# —— info ——


def cmd_info(_args: argparse.Namespace) -> int:
    settings = get_settings()
    client = default_client()
    print(f"scout {scout.__version__}")
    print(f"  LLM 客户端      : {client.model_name}"
          f"{'（离线启发式，未配置 SCOUT_LLM_BASE_URL）' if not settings.llm.configured else ''}")
    print(f"  Agent 预算      : {settings.agent.max_steps} 步 / "
          f"{settings.agent.max_tool_calls} 次工具调用 / {settings.agent.deadline_seconds:.0f}s")
    print(f"  检索            : top_k={settings.retrieval.top_k}, "
          f"RRF k={settings.retrieval.rrf_k}, "
          f"auto_merge={'on' if settings.retrieval.auto_merge_enabled else 'off'}")
    print(f"  证据预算        : {settings.retrieval.evidence_budget_chars} 字符")
    print(f"  拒答门控        : 覆盖阈值 {settings.verify.sufficiency_min_coverage}")
    print("  可用工具        : " + ", ".join(spec.name for spec in build_default_tools()) + ", knowledge_search")
    return 0


# —— demo ——


def cmd_demo(args: argparse.Namespace) -> int:
    """不读数据集，用内置小语料跑通一次 Agent 问答。"""

    documents = [
        (
            "redis.md",
            "Redis 的高性能来自三点：数据放在内存中，避免磁盘 IO；"
            "单线程处理命令，省去上下文切换与锁竞争；"
            "使用 IO 多路复用（epoll）在单线程内处理大量并发连接。",
        ),
        (
            "golang.md",
            "Go 的 GMP 调度模型由 G、M、P 组成。G 是 goroutine，M 是操作系统线程，"
            "P 是处理器上下文。调度器通过 work stealing 提升并行效率。",
        ),
    ]
    index = build_index(documents)
    pipeline = RAGPipeline(index, default_client())
    registry = ToolRegistry()
    registry.register(KnowledgeSearchTool(pipeline, trace_factory=lambda: None).spec())
    for spec in build_default_tools():
        registry.register(spec)
    agent = ToolAgent(default_client(), registry, orchestrator=SelfHealingOrchestrator())

    question = args.question or "Redis 为什么这么快？"
    result = agent.run(question)
    print(f"问题：{question}")
    print(f"结果：{result.outcome}（{result.step_count} 步 / {result.tool_call_count} 次工具调用）")
    print(f"回答：{result.answer}")
    print(f"归因：{result.grounding.verdict.value}，支撑率 {result.grounding.support_rate:.2f}")
    print("耗时分布：" + json.dumps(result.trace.kind_durations() if result.trace else {}, ensure_ascii=False))
    return 0


# —— eval ——


def cmd_eval_validate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    print(f"数据集 {dataset.name} 校验通过：{len(dataset)} 条样本")
    for tag, count in dataset.tag_counts().items():
        print(f"  {tag:<20} {count}")
    print(f"指纹：{dataset.fingerprint()}")
    return 0


def cmd_eval_run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    documents = load_corpus(args.corpus, max_files=args.max_files)
    if not documents:
        print(f"语料目录为空：{args.corpus}", file=sys.stderr)
        return 2

    config = RunConfig(
        mode=args.mode,
        label=args.label or "",
        limit=args.limit,
    )
    report = run_evaluation(dataset, documents, run_config=config)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"报告已写入：{path}")

    markdown = render_report(report)
    if args.markdown:
        path = Path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown + "\n" + render_failure_taxonomy(report), encoding="utf-8")
        print(f"Markdown 已写入：{path}")
    else:
        print(markdown)

    if args.fail_under is not None:
        score = report.metric("recall_at_5") or 0.0
        if score < args.fail_under:
            print(
                f"门禁未通过：Recall@5 = {score:.3f} < {args.fail_under:.3f}",
                file=sys.stderr,
            )
            return 1
    return 0


def cmd_eval_ablation(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    documents = load_corpus(args.corpus, max_files=args.max_files)
    if not documents:
        print(f"语料目录为空：{args.corpus}", file=sys.stderr)
        return 2

    configs = default_ablation_configs()
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        configs = [item for item in configs if item.label in wanted]
        if not configs:
            print(f"没有匹配的配置：{args.only}", file=sys.stderr)
            return 2

    reports = run_ablation(dataset, documents, configs=configs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for report in reports:
        safe = report.config_label.replace("/", "_").replace(":", "_")
        (outdir / f"{safe}.json").write_text(
            json.dumps(report.to_dict(include_observations=False), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (outdir / f"{safe}.md").write_text(render_report(report), encoding="utf-8")

    table = render_ablation_table(reports)
    (outdir / "ablation.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"\n消融结果已写入：{outdir}")
    return 0


# —— hitl demo ——


def cmd_hitl_demo(args: argparse.Namespace) -> int:
    """不读数据集，演示中断 → 审批 → 恢复 → 时间旅行的完整闭环。"""

    registry = ToolRegistry([*build_default_tools(), *build_action_tools()])

    send_mail = LLMResponse(
        tool_calls=[ToolCall(name="send_email", arguments={"to": "boss@example.com", "subject": "周报", "body": "…"})]
    )
    agent = ResumableAgent(
        ScriptedLLM(
            [
                send_mail,
                LLMResponse(content="报告已发出。"),
                send_mail,  # 分叉后的第二遍会再次触发同一个调用，用于演示幂等回放
                LLMResponse(content="报告已发出。"),
            ]
        ),
        registry,
        timeout_policy=TimeoutPolicy.REJECT,
    )

    first = agent.start("把本周周报发给老板")
    print(f"1. 启动 → 状态：{first.status.value}")
    if first.needs_human:
        request = first.request.to_dict()
        print(f"2. 触发审批：{request['tool_name']} 风险={request['risk_level']}")
        print(f"   原因：{request['risk_reason']}")

        decision = HumanDecision(
            request_id=first.request.request_id, approved=True, decided_by="reviewer"
        )
        resumed = agent.resume(first.run_id, decision)
        print(f"3. 审批通过后恢复 → 状态：{resumed.status.value}  回答：{resumed.answer}")

        # 时间旅行：回到发邮件之前分叉
        history = agent.history(first.run_id)
        earliest = min(checkpoint.step for checkpoint in history)
        forked = agent.run_fork(
            first.run_id,
            at_step=earliest,
            new_run_id=None,
        )
        print(f"4. 时间旅行分叉（step {earliest}）→ 新 run：{forked.run_id}")
        print(
            "   关键保障：分叉不能重复副作用（效果数 "
            f"{forked.meta['effects']}，幂等回放 {forked.meta['effects_replayed']} 次）"
        )
    return 0


# —— mcp ——


def cmd_mcp(args: argparse.Namespace) -> int:
    """以 MCP Server 运行。"""

    from .mcp import run_mcp

    corpus = args.corpus or str(Path(__file__).resolve().parents[2] / "datasets" / "longdoc-gold")
    run_mcp(corpus)
    return 0


# —— serve ——


def cmd_serve(args: argparse.Namespace) -> int:
    """启动 Web 控制台。"""

    from .server import serve

    corpus = args.corpus or str(Path(__file__).resolve().parents[2] / "datasets" / "longdoc-gold")
    serve(corpus, host=args.host, port=args.port, embed_backend=getattr(args, "embed", "auto"))
    return 0


# —— doctor ——


def cmd_doctor(_args: argparse.Namespace) -> int:
    """体检：报告当前哪些部件是"真实模型"，哪些还是离线替身。

    这个命令存在的理由：**"跑通了"和"跑的是真东西"是两件事。**
    离线实现让项目可复现，但它不是能力本身。如果不把"现在生效的是哪一种"
    明确打印出来，很容易把离线基线的分数当成真实水平——那是最坏的结果。
    """

    from urllib.parse import urlparse

    from .rag.embed import embedder_status

    settings = get_settings()
    client = default_client()
    status = embedder_status(settings)
    ok = True

    print(f"scout {scout.__version__} 环境体检")
    print("=" * 56)

    # —— LLM ——
    llm_real = settings.llm.configured
    print(f"[LLM]      {'✅ 真实模型' if llm_real else '⚠️  离线启发式'}")
    print(f"           客户端 : {client.model_name}")
    if llm_real:
        host = urlparse(settings.llm.base_url).netloc or settings.llm.base_url
        print(f"           网关   : {host}")
        print(f"           密钥   : {'已配置' if settings.llm.api_key else '未配置（部分网关必需）'}")
        print(f"           可达性 : {'可达' if _tcp_ok(host) else '不可达（检查网络/代理）'}")
    else:
        print("           未配置 SCOUT_LLM_BASE_URL → 回答由词法启发式生成，质量不代表真实水平")
        ok = False

    # —— 向量器 ——
    semantic = bool(status["semantic"])
    print(f"[向量器]   {'✅ 语义向量' if semantic else '⚠️  离线哈希（词法，非语义）'}")
    print(f"           后端   : {status['configured_backend']} → 生效 {status['resolved_backend']}")
    print(f"           模型   : {status['local_model']}")
    print(f"           本地能力: {status['local_detail']}")
    if not semantic:
        print("           启用方式: pip install fastembed   然后 set SCOUT_EMBED_BACKEND=local")
        ok = False

    # —— 语料与数据 ——
    root = Path(__file__).resolve().parents[2]
    corpus = root / "datasets" / "longdoc-gold"
    musique = root / "datasets" / "musique_ans_dev.jsonl"
    print(f"[语料]     {'✅' if corpus.exists() else '❌'} 长文档语料 {corpus}")
    print(f"[公开基准] {'✅' if musique.exists() else '⚠️  未下载'} MuSiQue-Ans {musique.name}")
    if not musique.exists():
        print("           下载: 见 scripts/bench_musique.py 顶部说明")

    # —— 可选依赖 ——
    print("[依赖]")
    for module, hint in [
        ("requests", "LLM/向量 HTTP 调用"),
        ("pydantic", "结构化输出解析"),
        ("fastembed", "本地语义向量（可选）"),
        ("pytest", "测试（可选）"),
    ]:
        try:
            __import__(module)
            print(f"           ✅ {module:<12} {hint}")
        except ImportError:
            print(f"           ⚪ {module:<12} {hint} — 未安装")

    print("=" * 56)
    if ok:
        print("结论：LLM 与向量器都已是真实模型，评测结果可直接对外汇报。")
    else:
        print("结论：**当前仍在部分离线模式**。上面的 ⚠️ 项决定结果能否代表真实水平。")
    return 0 if ok else 1


def _tcp_ok(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    import socket

    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# —— 参数解析 ——


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scout", description="scout — 长文档 Agentic RAG 框架")
    parser.add_argument("--version", action="version", version=f"scout {scout.__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="打印当前配置与可用工具").set_defaults(func=cmd_info)

    sub.add_parser("doctor", help="体检：哪些部件是真实模型，哪些还是离线替身").set_defaults(func=cmd_doctor)

    demo = sub.add_parser("demo", help="用内置小语料跑通一次 Agent 问答（无需外部服务）")
    demo.add_argument("--question", default="", help="自定义问题")
    demo.add_argument(
        "--embed",
        choices=("auto", "local", "hashing", "openai"),
        default=argparse.SUPPRESS,
        help="向量器后端（默认读取 SCOUT_EMBED_BACKEND，auto=有 fastembed 就用本地语义模型）",
    )
    demo.set_defaults(func=cmd_demo)

    hitl = sub.add_parser("hitl", help="HITL 中断/审批/恢复/时间旅行演示")
    hitl.add_argument("--mode", choices=("demo",), default="demo", help="演示模式")
    hitl.set_defaults(func=cmd_hitl_demo)

    mcp = sub.add_parser("mcp", help="以 MCP Server 运行（stdio，供 Agent 客户端接入）")
    mcp.add_argument("--corpus", default="", help="语料目录，默认 datasets/longdoc-gold")
    mcp.set_defaults(func=cmd_mcp)

    serve = sub.add_parser("serve", help="启动本地 Web 控制台（离线，零新增依赖）")
    serve.add_argument("--corpus", default="", help="语料目录，默认 datasets/longdoc-gold")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument(
        "--embed",
        choices=("auto", "local", "hashing", "openai"),
        default="auto",
        help="向量器后端；local=本地语义向量（需 fastembed）",
    )
    serve.set_defaults(func=cmd_serve)

    evaluation = sub.add_parser("eval", help="评测相关命令")
    eval_sub = evaluation.add_subparsers(dest="eval_command")

    validate = eval_sub.add_parser("validate", help="校验数据集")
    validate.add_argument("--dataset", default=DEFAULT_DATASET)
    validate.set_defaults(func=cmd_eval_validate)

    run = eval_sub.add_parser("run", help="运行一次评测")
    run.add_argument("--dataset", default=DEFAULT_DATASET)
    run.add_argument("--corpus", default=DEFAULT_CORPUS)
    run.add_argument("--mode", choices=("pipeline", "agent", "multiagent"), default="pipeline")
    run.add_argument("--label", default="")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--max-files", type=int, default=None)
    run.add_argument("--out", default="")
    run.add_argument("--markdown", default="")
    run.add_argument("--fail-under", type=float, default=None, help="Recall@5 低于该值时返回非零退出码")
    run.set_defaults(func=cmd_eval_run)

    ablation = eval_sub.add_parser("ablation", help="运行模块级消融")
    ablation.add_argument("--dataset", default=DEFAULT_DATASET)
    ablation.add_argument("--corpus", default=DEFAULT_CORPUS)
    ablation.add_argument("--outdir", default="reports/ablation")
    ablation.add_argument("--max-files", type=int, default=None)
    ablation.add_argument("--only", default="", help="只跑指定配置，逗号分隔")
    ablation.set_defaults(func=cmd_eval_ablation)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
