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

    # 状态存储：默认内存（演示够用），给了 --store 就换成外部后端。
    # 这里刻意"给了 URL 就必须成功"——不做静默降级，
    # 因为"以为在用 Redis 实际在用内存"会让跨进程恢复这件事悄悄不成立。
    store = None
    store_label = "内存"
    store_url = getattr(args, "store", "") or ""
    if store_url:
        from .hitl.checkpoints import RedisCheckpointStore

        try:
            store = RedisCheckpointStore(store_url)
            store_label = f"Redis({store_url})"
        except Exception as exc:  # noqa: BLE001
            print(f"无法连接状态存储 {store_url}：{type(exc).__name__}: {exc}")
            print("提示：Redis 需要 pip install redis 且服务可达；也可先用内存实现跑演示。")
            return 1
    print(f"状态存储：{store_label}")

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
        store=store,
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


def cmd_intent(args: argparse.Namespace) -> int:
    """跑一次意图三层漏斗，并说明"停在哪一层、为什么"。

    这个命令的价值不在分类结果本身，而在**可解释性**：
    同一个问题为什么不走大模型？为什么被要求澄清？
    这些在生产里都是会被反复追问的问题，答案应该能直接打印出来。
    """

    settings = get_settings()
    from .rag.embed import default_embedder
    from .runtime.intent import build_knowledge_funnel

    client = default_client() if args.llm else None
    try:
        embedder = default_embedder(settings.embedding) if args.embed else None
    except Exception as exc:  # noqa: BLE001 - 没有向量器时退化为"只有规则层"，如实说明
        print(f"（向量器不可用，第二层语义判定跳过：{type(exc).__name__}）")
        embedder = None

    funnel = build_knowledge_funnel(
        embedder=embedder,
        llm=client,
        semantic_threshold=settings.intent.semantic_threshold,
        margin_threshold=settings.intent.margin_threshold,
    )

    queries = [args.query] if args.query else [
        "云计算标准体系包括哪几个部分？",
        "帮我给客户发一封邮件",
        "今天天气怎么样",
        "帮我写一首诗",
        "我那笔订单想退一下",
    ]
    print(f"意图漏斗（向量器：{'语义' if embedder else '未启用（仅规则层+LLM）'}）")
    print("-" * 72)
    for query in queries:
        result = funnel.classify(query)
        print(f"输入：{query}")
        print(f"  判定：{result.label or '（未判定）'}  置信度 {result.confidence:.3f}  "
              f"分差 {result.margin:.3f}")
        print(f"  停在：{result.tier.value} 层    理由：{result.reason}")
        options = funnel.clarify_options(result)
        if options:
            print(f"  澄清候选：{' / '.join(options)}")
        print()
    print("各层命中分布：", funnel.distribution())
    if not settings.intent.enabled:
        print()
        print("提示：流水线默认不启用意图前筛。要启用：set SCOUT_INTENT_ENABLED=1")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """跑一遍入库流水线（版面还原 → 切分 → 去重 → 版本），只打报告不写库。"""

    from .data import ingest

    directory = Path(args.corpus)
    if not directory.exists():
        print(f"语料目录不存在：{directory}")
        return 1
    documents = load_corpus(directory, max_files=args.limit or None)
    sections, report, registry = ingest(documents)

    print(f"入库流水线：{directory}")
    print("-" * 72)
    print(f"  文档        : {report.documents}")
    print(f"  切分片段    : {report.sections}")
    print(f"  保留        : {report.kept}")
    print(f"  精确重复丢弃: {report.dropped_exact}")
    print(f"  近似重复丢弃: {report.dropped_near}")
    print(f"  版本变更    : {report.versions_created}")
    layout = report.layout
    print(f"  版面        : 双栏={layout.get('columns_detected')} "
          f"重排行数={layout.get('lines_reordered')} "
          f"跨页表格合并={layout.get('tables_merged')} "
          f"补写表头={layout.get('headers_repeated')}")
    scores = sorted(report.reading_order.values())
    if scores:
        undecided = sum(1 for value in report.reading_order.values() if value >= 1.0)
        judgeable = [value for value in scores if value < 1.0]
        summary = f"  阅读顺序    : 可判断 {len(judgeable)} 篇 / 无法判断 {undecided} 篇"
        if judgeable:
            summary += f"｜最低 {judgeable[0]:.2f} 中位 {judgeable[len(judgeable) // 2]:.2f}"
        print(summary)
        print("              （表格/列表为主的文档散文行不足，判为无法判断而不是不合格；")
        print("                在 40 篇真实语料上量过，可判断的那些没有一篇低于 0.5）")
    if report.low_quality:
        print(f"  质量门禁拒绝: {len(report.low_quality)} 篇 → {report.low_quality[:5]}")
    else:
        print("  质量门禁    : 未启用拒收（默认只记录；需要显式设阈值才会拦）")
    print()
    print(f"版本登记：{len(list(registry.active()))} 个活跃文档")
    if args.out:
        Path(args.out).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写入 {args.out}")
    return 0


def cmd_stack(args: argparse.Namespace) -> int:
    """展示运行时链的装配顺序，并跑一次最小缓存/预算/路由验证。"""

    from .llm.base import ChatMessage, LLMRequest
    from .rag.embed import default_embedder
    from .runtime.factory import build_runtime_stack

    settings = get_settings()
    # 演示时把三层都打开，否则"什么都没有"的栈看不出顺序
    demo = settings
    try:
        from dataclasses import replace

        demo = replace(
            settings,
            runtime=replace(
                settings.runtime,
                cache_enabled=True,
                routing_enabled=True,
                budget_tokens=args.budget,
            ),
        )
        embedder = default_embedder(settings.embedding) if args.embed else None
    except Exception:  # noqa: BLE001
        embedder = None

    stack = build_runtime_stack(default_client(), demo, embedder=embedder)
    print("运行时链装配（由内到外）：", " → ".join(stack.order))
    print("-" * 72)
    print("  顺序不是随意的：缓存最外层（命中不花钱，必须最先短路）→")
    print("  预算中间层（要按真正选中的模型价判额度）→ 路由最内层（决定用哪个模型）。")
    print()

    request = LLMRequest(
        messages=[ChatMessage(role="user", content=args.query)],
        task="grade",
        context={"evidence_digest": "demo-scope"},
    )
    for attempt in (1, 2):
        response = stack.client.complete(request)
        print(f"  第 {attempt} 次调用：{len(response.content)} 字符，"
              f"用量 {response.usage.total} token")
    report = stack.report()
    print()
    print("各层数据：")
    print(f"  缓存：{report.get('cache')}")
    print(f"  预算：{report.get('budget')}")
    print(f"  路由：{report.get('router')}")
    print()
    print("注意第 2 次调用的 token 用量为 0 —— 它是缓存命中，本来就不该花钱；")
    print("而缓存层在预算层之外，所以这次命中不会触发任何预算检查。")
    return 0


def cmd_judge_compare(args: argparse.Namespace) -> int:
    """LLM 裁判 vs 词法归因校验：同一批答案上两种评分的对照。"""

    from .evolution.compare import compare_judges, demo_cases

    settings = get_settings()
    client = default_client()
    cases = demo_cases()
    if not settings.llm.configured and not args.allow_offline:
        print("未配置真实 LLM，裁判层的分数没有意义。")
        print("用 --allow-offline 强行运行（会用离线启发式当裁判，仅供看流程）。")
        return 1

    result = compare_judges(client, cases)
    print(f"对照样本：{len(cases)} 条")
    print("-" * 72)
    print(f"{'问题':<28}{'归因判定':<12}{'裁判均分':<10}{'一致'}")
    for row in result["rows"]:
        agree = "✓" if row["agree"] else "✗"
        print(f"{row['question'][:26]:<28}{row['verdict']:<12}{row['judge_overall']:<10.2f}{agree}")
    print()
    print(f"一致率：{result['agreement']:.0%}（{result['n']} 条）")
    print("裁判校准：" + json.dumps(result["calibration"], ensure_ascii=False))
    print()
    print("为什么要把两者放在一起看：")
    print("  归因校验是**词法**的（能不能在证据里找到支撑），便宜、确定、可复现；")
    print("  裁判是**语义**的（答得切不切题、有没有冗余），贵、有偏见、但能覆盖词法盲区。")
    print("  两者不一致的样本最有价值——它不是'谁对谁错'，而是提示 rubric 或阈值需要校准。")
    return 0


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
    hitl.add_argument("--store", default="", help="状态存储：留空=内存，redis://host:port/db 用 Redis")
    hitl.set_defaults(func=cmd_hitl_demo)

    intent = sub.add_parser("intent", help="跑一次意图三层漏斗，看它停在哪一层、为什么")
    intent.add_argument("query", nargs="?", default="", help="要判定的问题；省略则跑内置样例")
    intent.add_argument("--embed", action="store_true", help="启用第二层语义判定（需要向量器）")
    intent.add_argument("--llm", action="store_true", help="启用第三层 LLM 兜底")
    intent.set_defaults(func=cmd_intent)

    ingest = sub.add_parser("ingest", help="跑一遍入库流水线（版面还原/切分/去重/版本），只出报告")
    ingest.add_argument("--corpus", default=DEFAULT_CORPUS, help="语料目录")
    ingest.add_argument("--limit", type=int, default=0, help="最多读多少个文件（0=全部）")
    ingest.add_argument("--out", default="", help="把报告写成 JSON")
    ingest.set_defaults(func=cmd_ingest)

    stack = sub.add_parser("stack", help="展示运行时链装配顺序（缓存/预算/路由）并做最小验证")
    stack.add_argument("--query", default="判一下这段证据够不够", help="用于演示的请求内容")
    stack.add_argument("--budget", type=int, default=20000, help="演示用的 token 预算")
    stack.add_argument("--embed", action="store_true", help="启用语义缓存（需要向量器）")
    stack.set_defaults(func=cmd_stack)

    judge = sub.add_parser("judge-compare", help="LLM 裁判 vs 词法归因校验的对照")
    judge.add_argument("--allow-offline", action="store_true", help="未配置 LLM 时也强行跑（仅供看流程）")
    judge.set_defaults(func=cmd_judge_compare)

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
