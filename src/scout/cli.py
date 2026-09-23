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
import os
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
        skip=args.skip,
    )
    report = run_evaluation(dataset, documents, run_config=config)

    if getattr(args, "observations_out", ""):
        from .evolution.observations import ObservationLog

        log = ObservationLog(args.observations_out)
        written = log.extend(report.observations)
        print(f"观测已写入：{args.observations_out}（{written} 条）")

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
    """跑一遍入库流水线（版面还原 → 切分 → 去重 → 版本），只打报告不写库。

    支持图片/扫描件：``--images`` 让目录里的图片走视觉模型抽取文本，
    抽取结果与文本文件走**同一条**入库流水线——
    多模态不是旁路，只是入库入口多了一种输入形态。
    """

    from .data import ingest, load_documents_with_extraction
    from .data.extract import IMAGE_SUFFIXES, TEXT_SUFFIXES, build_extractor

    directory = Path(args.corpus)
    if not directory.exists():
        print(f"语料目录不存在：{directory}")
        return 1

    settings = get_settings()
    documents: list[tuple[str, str]] = []
    extraction_report: dict[str, object] | None = None

    if args.images:
        extractor = build_extractor(
            vision_enabled=settings.vision.enabled,
            model=settings.vision.model,
            base_url=settings.vision.base_url or settings.llm.base_url,
            api_key=settings.vision.api_key or settings.llm.api_key,
            prompt=settings.vision.prompt,
            max_tokens=settings.vision.max_tokens,
            max_image_side=settings.vision.max_image_side,
        )
        if extractor is None:
            print("已指定 --images，但没有可用的视觉抽取器。")
            print("请设置：set SCOUT_VISION_ENABLED=1 与 set SCOUT_VLM_MODEL=<视觉模型名>")
            print("（也可以用 SCOUT_VLM_BASE_URL / SCOUT_VLM_API_KEY 指定独立网关）")
            return 1
        suffixes = TEXT_SUFFIXES | IMAGE_SUFFIXES
        files = sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes
        )
        if args.limit:
            files = files[: args.limit]
        print(f"抽取器：{extractor.name}（模型 {settings.vision.model}）")
        outcome = load_documents_with_extraction(files, image_extractor=extractor)
        documents = outcome.documents
        extraction_report = outcome.to_dict()
        print(f"抽取完成：成功 {len(documents)} 篇｜失败 {len(outcome.skipped)} 篇")
        for item in outcome.extracted[:5]:
            print(f"  - {item['source']}（{item['extractor']}，{item['chars']} 字，{item['duration_ms']}ms）")
        for skipped in outcome.skipped[:5]:
            print(f"  ✗ {skipped}")
        for warning in outcome.warnings[:5]:
            print(f"  ! {warning}")
        # token 成本必须可见：图片抽取按张计费，一批扫描件的成本不该靠猜
        usage_report = getattr(extractor, "usage_report", None)
        if callable(usage_report):
            usage = usage_report()
            if usage.get("calls"):
                print(
                    f"  token   : 共 {usage['total_tokens']}（prompt {usage['prompt_tokens']} + "
                    f"completion {usage['completion_tokens']}）｜ "
                    f"平均 {usage['avg_tokens_per_image']}/张"
                )
        print()
    else:
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
        payload: dict[str, object] = report.to_dict()
        if extraction_report is not None:
            payload["extraction"] = extraction_report
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写入 {args.out}")
    return 0


def cmd_milvus_smoke(args: argparse.Namespace) -> int:
    """对运行中的 Milvus 做一次联机冒烟：连通性 + 与内存精确检索的重合度。"""

    import subprocess

    script = Path(__file__).resolve().parents[2] / "scripts" / "milvus_smoke.py"
    if not script.exists():
        print(f"未找到冒烟脚本：{script}")
        return 1
    env = dict(os.environ)
    if args.uri:
        env["SCOUT_MILVUS_URI"] = args.uri
    env["SMOKE_DOCS"] = str(args.docs)
    env["SMOKE_TOP_K"] = str(args.top_k)
    print(f"运行 {script.name}（Milvus {args.uri or '默认 127.0.0.1:19530'}）…")
    completed = subprocess.run(  # noqa: S603 - 本仓库自带脚本
        [sys.executable, str(script)], env=env, check=False
    )
    return int(completed.returncode)


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


def cmd_flywheel(args: argparse.Namespace) -> int:
    """把数据飞轮转一圈：观测 → 挖掘 → 复核 → 评测集增量。

    **这一步的产物是"候选"，不是"新评测集"。**
    挖掘是自动的，入库必须有人签字——模型判错的样本里混着
    "这题本来就不该答"。直接把它当"应该答对"的样本加进评测集，
    会系统性地惩罚一个行为正确的系统：门控被越调越松，最后变成自信胡说。

    所以本命令产出两样东西：
    - 复核队列文件（JSONL，带证据，供人逐条补 gold）
    - 数据集增量（``--out``），每条都带"由飞轮挖出、待补标"的诚实标记
    """

    from .evolution import FailureMiner, ReviewQueue
    from .evolution.observations import FlywheelReport, ObservationLog, mine_report
    from .evaluation.dataset import EvalCase, EvalDataset, load_dataset, save_dataset

    log = ObservationLog(args.observations)
    records = log.read()
    if not records:
        print(f"观测文件为空或不存在：{args.observations}")
        print("先用 `scout eval run --observations-out <path>` 产出观测。")
        return 1

    summary = mine_report(records)
    print(f"读取观测 {summary['total']} 条（去重后 {summary['unique_questions']} 个问题）")
    print(f"  其中：拒答 {summary['abstained']} ｜ 低支撑 {summary['low_support']} ｜ 空检索 {summary['empty_retrieval']}")
    print("-" * 72)

    miner = FailureMiner()
    candidates = miner.mine(records)
    queue = ReviewQueue()
    added = queue.enqueue(candidates)

    print(f"挖出候选 {len(candidates)} 条（新入队 {added} 条）")
    print("来源分布：" + (", ".join(f"{k}={v}" for k, v in sorted(miner.stats.by_reason.items())) or "无"))
    print()

    if args.list_only:
        for case in queue.sample_pending(limit=args.limit):
            print(f"  [{case.reason.value:<18}] {case.question[:60]}")
            print(f"      证据 {case.retrieved} 条 ｜ 支撑率 {case.support_rate:.2f} ｜ 结果 {case.outcome}")
        print()
        print("加 --approve N 复核通过前 N 条，或 --approve-all 全部通过并写出增量。")
        return 0

    approve_count = len(queue.pending) if args.approve_all else max(0, args.approve)
    approved = []
    for case in queue.sample_pending(limit=approve_count or len(queue.pending)):
        decided = queue.approve(case.case_id)
        if decided is not None:
            approved.append(decided)
    for case in list(queue.pending.values()):
        queue.reject(case.case_id, "本轮未复核（留待下一轮）")

    print(f"复核：通过 {len(approved)} 条，拒绝/待定 {len(queue.rejected)} 条")

    dataset = load_dataset(args.dataset)
    before = len(dataset.cases)
    existing = {case.question for case in dataset.cases}
    increment: list[EvalCase] = []
    skipped_existing = 0
    for case in approved:
        if case.question in existing:
            skipped_existing += 1
            continue
        increment.append(
            EvalCase(
                case_id=case.case_id,
                question=case.question,
                tags=list(_guess_tags(case)),
                # gold 留空：我们不知道正确答案，只有人补标之后才有资格进主评测集。
                # 这正是"候选"与"样本"的区别。
                gold_snippets=[],
                allow_unknown=True,
                notes=f"飞轮挖出：{case.reason.value}｜观测结果 {case.outcome}｜支撑率 {case.support_rate:.2f}｜待补 gold",
            )
        )

    if skipped_existing:
        # 把"被跳过"显式说出来。**从自己的评测集里挖，是挖不出新东西的**——
        # 这是飞轮的常见误解：它的增量来自"评测集之外的真实流量"，
        # 拿评测集自己的观测来喂它，结果一定是零增长，而且看起来像 bug。
        print(f"跳过 {skipped_existing} 条：问题已在评测集中（挖掘用观测应当来自评测集之外）")

    merged = list(dataset.cases) + increment
    if args.out:
        save_dataset(
            EvalDataset(
                name=dataset.name + "-flywheel",
                cases=merged,
                description=dataset.description,
                corpus_fingerprint=dataset.corpus_fingerprint,
            ),
            args.out,
        )

    report = FlywheelReport(
        observations=summary["total"],
        mined=len(candidates),
        approved=len(approved),
        rejected=len(queue.rejected),
        dataset_before=before,
        dataset_after=len(merged),
        by_reason=dict(miner.stats.by_reason),
        output_path=args.out or "",
        skipped_existing=skipped_existing,
        new_cases=len(increment),
    )
    print()
    print(report.render())

    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# 数据飞轮一轮\n\n"
            "## 观测摘要\n\n"
            f"- 观测 {summary['total']} 条（去重后 {summary['unique_questions']} 个问题）\n"
            f"- 拒答 {summary['abstained']} ｜ 低支撑 {summary['low_support']} ｜ 空检索 {summary['empty_retrieval']}\n\n"
            "## 飞轮结果\n\n" + report.render() +
            "\n## 候选清单\n\n" + "\n".join(
                f"- `{case.reason.value}` {case.question[:70]}（证据 {case.retrieved}，支撑率 {case.support_rate:.2f}）"
                for case in approved
            ) + "\n",
            encoding="utf-8",
        )
        print(f"报告已写入 {args.report}")

    print()
    print("注意：新增样本的 gold 留空 —— 它们只有经过人工补标才有资格进主评测集。")
    print("自动挖掘可以扩大候选池，但不能自己决定'什么算答对'。")
    return 0


def _guess_tags(case: Any) -> list[Any]:
    """从失败画像猜查询类型，只作为给复核人的初始建议。

    **刻意只猜"类型"，不猜"答案"。** 类型可以从证据数量与检索行为推出来，
    而"正确答案是什么"推不出来——那必须由人给。
    """

    from .evaluation.taxonomy import QueryTag

    tags: list[QueryTag] = []
    if case.retrieved >= 4:
        tags.append(QueryTag.MULTI_HOP)
    else:
        tags.append(QueryTag.SINGLE_FACT)
    if case.reason.value == "abstained":
        tags.append(QueryTag.NO_KNOWLEDGE)
    return tags


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

    # —— 向量库后端（内存 / Milvus）——
    if settings.milvus.enabled:
        from .rag.milvus_store import MilvusDenseStore

        store = MilvusDenseStore(
            uri=settings.milvus.uri,
            token=settings.milvus.token,
            prefix=settings.milvus.collection_prefix,
            timeout=min(settings.milvus.timeout_seconds, 3.0),
        )
        reachable = store.ping()
        print(f"[向量库]   {'✅ Milvus' if reachable else '❌ Milvus 不可达 → 会降级回内存'}")
        print(f"           地址   : {settings.milvus.uri}（{settings.milvus.index_type} / {settings.milvus.metric_type}）")
        if reachable:
            print(f"           集合前缀: {settings.milvus.collection_prefix}_<语料指纹>")
            print(f"           拒降级 : {'是（strict，指标必然来自 Milvus）' if settings.milvus.require_sync else '否（不可用时回退内存并留痕）'}")
        else:
            print(f"           失败原因: {store.last_error or '未知'}")
            print("           启动方式: docker start milvus-etcd milvus-minio milvus-standalone")
    else:
        print("[向量库]   ⚪ 内存索引（精确暴力检索，评测可离线复现）")
        print("           启用 Milvus: set SCOUT_MILVUS_ENABLED=1  然后 scout milvus-smoke")

    # —— 多模态抽取 ——
    vision_ready = bool(settings.vision.enabled and settings.vision.model)
    print(f"[多模态]   {'✅ 图片抽取（视觉模型）' if vision_ready else '⚪ 未启用（只处理文本文件）'}")
    if vision_ready:
        print(f"           模型   : {settings.vision.model}")
        print(f"           网关   : {urlparse(settings.vision.base_url or settings.llm.base_url).netloc or '(未配置)'}")
    else:
        print("           启用方式: set SCOUT_VISION_ENABLED=1 与 set SCOUT_VLM_MODEL=<视觉模型名>")

    # —— 可选依赖 ——
    print("[依赖]")
    for module, hint in [
        ("requests", "LLM/向量 HTTP 调用"),
        ("pydantic", "结构化输出解析"),
        ("fastembed", "本地语义向量（可选）"),
        ("pymilvus", "Milvus 向量库后端（可选）"),
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
    ingest.add_argument(
        "--images",
        action="store_true",
        help="把图片/扫描件也纳入（走视觉模型抽取后进同一条流水线）",
    )
    ingest.set_defaults(func=cmd_ingest)

    milvus = sub.add_parser("milvus-smoke", help="Milvus 联机冒烟：连通性 + 与内存精确检索的重合度")
    milvus.add_argument("--uri", default="", help="覆盖 SCOUT_MILVUS_URI")
    milvus.add_argument("--docs", type=int, default=8, help="语料篇数")
    milvus.add_argument("--top-k", type=int, default=10, help="检索深度")
    milvus.set_defaults(func=cmd_milvus_smoke)

    stack = sub.add_parser("stack", help="展示运行时链装配顺序（缓存/预算/路由）并做最小验证")
    stack.add_argument("--query", default="判一下这段证据够不够", help="用于演示的请求内容")
    stack.add_argument("--budget", type=int, default=20000, help="演示用的 token 预算")
    stack.add_argument("--embed", action="store_true", help="启用语义缓存（需要向量器）")
    stack.set_defaults(func=cmd_stack)

    judge = sub.add_parser("judge-compare", help="LLM 裁判 vs 词法归因校验的对照")
    judge.add_argument("--allow-offline", action="store_true", help="未配置 LLM 时也强行跑（仅供看流程）")
    judge.set_defaults(func=cmd_judge_compare)

    flywheel = sub.add_parser("flywheel", help="数据飞轮：观测 → 挖掘 → 复核 → 评测集增量")
    flywheel.add_argument("--observations", default="reports/observations.jsonl", help="观测日志（JSONL）")
    flywheel.add_argument("--dataset", default=DEFAULT_DATASET, help="当前评测集")
    flywheel.add_argument("--out", default="", help="写出增量后的评测集")
    flywheel.add_argument("--report", default="", help="写出 Markdown 报告")
    flywheel.add_argument("--approve", type=int, default=0, help="复核通过前 N 条")
    flywheel.add_argument("--approve-all", action="store_true", help="全部通过（谨慎）")
    flywheel.add_argument("--list-only", action="store_true", help="只看候选，不写任何东西")
    flywheel.add_argument("--limit", type=int, default=20, help="展示候选条数上限")
    flywheel.set_defaults(func=cmd_flywheel)

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
    run.add_argument("--skip", type=int, default=0, help="跳过前 N 条（用于只跑评测集之外的部分）")
    run.add_argument("--max-files", type=int, default=None)
    run.add_argument("--out", default="")
    run.add_argument("--markdown", default="")
    run.add_argument("--fail-under", type=float, default=None, help="Recall@5 低于该值时返回非零退出码")
    run.add_argument(
        "--observations-out",
        default="",
        help="把本次评测的结构化观测追加到 JSONL（供数据飞轮挖掘）",
    )
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
