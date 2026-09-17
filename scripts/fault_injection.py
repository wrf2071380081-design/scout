"""故障注入实验：量化自愈编排的恢复率与降级路径。

核心问题：**系统在各种"倒霉"情况下，是崩溃给一个异常栈，还是败得可控、可分类、可降级？**

注入方式：用 scripted 的故障客户端（包一层的 ``HeuristicLLM`` 或注册表）在
**可预期的位置**引入失败——provider 超时、工具参数校验失败、重复调用死循环、
断点状态损坏……然后统计：

- **类型化失败率**：失败结束时是不是稳定的错误码（而不是裸异常）
- **预算内恢复率**：重试 ≤ 预算后实现正常完成或优雅降级
- **暴露率**：失败有没有以可读的方式告诉调用方（而不是"说不能"）

这个脚本不依赖任何外部服务，任何环境都可以复现：

```bash
python scripts/fault_injection.py
```
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# 允许直接运行此脚本时不预先安装 scout
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scout.errors import ProviderError, ScoutError, ErrorCode  # noqa: E402
from scout.hitl import HumanDecision, InMemoryCheckpointStore, ResumableAgent, TimeoutPolicy  # noqa: E402
from scout.llm.base import LLMRequest, LLMResponse, ToolCall  # noqa: E402
from scout.llm.scripted import HeuristicLLM, ScriptedLLM  # noqa: E402
from scout.agent.loop import ToolAgent  # noqa: E402
from scout.config import Settings  # noqa: E402
from scout.hitl.checkpoints import AgentState, Checkpoint, FileCheckpointStore  # noqa: E402
from scout.tools.builtin import build_default_tools  # noqa: E402
from scout.tools.registry import ToolRegistry  # noqa: E402
from scout.orchestrator.healing import SelfHealingOrchestrator  # noqa: E402


# —— 故障注入客户端 ——


class FaultInjectingLLM:
    """包装一个真 LLMClient，在指定任务的第 N 次调用时抛出指定故障。"""

    def __init__(self, inner, *, fail_task: str, times: int = 1, exc: Exception | None = None) -> None:
        self.inner = inner
        self.fail_task = fail_task
        self.times_left = times
        self.exc = exc or ProviderError("client_error: provider_timeout (注入的模拟故障)", retryable=True)
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return "fault-injecting"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request.task)
        if request.task == self.fail_task and self.times_left > 0:
            self.times_left -= 1
            raise self.exc
        return self.inner.complete(request)

    def fail_last_n(self, n: int) -> None:
        self.times_left = n


@dataclass(slots=True)
class Scenario:
    name: str
    description: str
    injected_fault: str
    expected_behavior: str
    recovered: bool = False
    typed_failure: bool = True
    outcome: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    crashed: bool = False
    exception: str = ""


# —— 场景 ——


def build_agent(llm) -> tuple[ToolAgent, SelfHealingOrchestrator]:
    registry = ToolRegistry([*build_default_tools()])
    orchestrator = SelfHealingOrchestrator()
    agent = ToolAgent(llm, registry, settings=Settings(), orchestrator=orchestrator)
    return agent, orchestrator


def scenario_provider_timeout_once() -> Scenario:
    """Provider 超时一次，然后恢复。自愈编排应该重试并完成。"""

    scenario = Scenario(
        name="provider_timeout_once",
        description="decide 阶段 provider 超时一次，第二次恢复",
        injected_fault="ProviderError(retryable=True) at task=decide ×1",
        expected_behavior="重试一次后正常完成，恢复统计里可见这次重试",
    )
    llm = FaultInjectingLLM(
        ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": "1+1"})]),
            LLMResponse(content="完成。"),
        ]),
        fail_task="decide",
        times=1,
    )
    agent, _orch = build_agent(llm)
    result = agent.run("算一下 1 加 1")
    scenario.outcome = result.outcome
    scenario.details["recovery_stats"] = result.meta.get("recovery_stats")
    scenario.details["recovery_attempts"] = result.meta.get("recovery_attempts")
    scenario.recovered = result.outcome in {"answered", "insufficient_evidence"}
    return scenario


def scenario_provider_persistent_failure() -> Scenario:
    """Provider 持续不可用。系统不能崩溃，也不能给一个"不知道"就完事——

    必须是类型化的失败。"""

    scenario = Scenario(
        name="provider_persistent_failure",
        description="provider 持续超时直到预算耗尽",
        injected_fault="ProviderError(retryable=True) 永久",
        expected_behavior="预算耗尽后类型化失败（provider_timeout），不崩溃不静默",
    )
    llm = FaultInjectingLLM(
        ScriptedLLM([]),
        fail_task="decide",
        times=99,
    )
    agent, _orch = build_agent(llm)
    result = agent.run("算一下 1 加 1")
    scenario.outcome = result.outcome
    scenario.details["recovery_stats"] = result.meta.get("recovery_stats")
    scenario.details["recovery_attempts"] = result.meta.get("recovery_attempts")
    scenario.recovered = result.outcome in {"provider_timeout", "insufficient_evidence"}
    return scenario


def scenario_malformed_tool_args() -> Scenario:
    """模型给了非法参数。注册表的 Schema 校验必须把它挡下，并把错误反馈给模型。"""

    scenario = Scenario(
        name="malformed_tool_args",
        description="模型发出非法参数的工具调用",
        injected_fault="calculator(expression='1+(')",
        expected_behavior="Schema 校验失败 → 错误回灌 → 模型修正或诚实停止",
    )
    llm = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": "1+("})]),
        LLMResponse(content="表达式写错了，中止。"),
    ])
    agent, _orch = build_agent(llm)
    result = agent.run("算一下")
    scenario.outcome = result.outcome
    scenario.details["recovery_stats"] = result.meta.get("recovery_stats")
    scenario.recovered = result.outcome in {"answered", "insufficient_evidence"}
    return scenario


def scenario_repeat_call_loop() -> Scenario:
    """同一工具、同一参数反复调用。重复调用抑制必须生效，循环不能无限。"""

    scenario = Scenario(
        name="repeat_call_loop",
        description="模型反复用同一参数调用同一工具",
        injected_fault="同一 calculator 调用 ×6",
        expected_behavior="重复调用被抑制，预算终止或最终作答，不崩溃不烧穿预算",
    )
    repeated = ToolCall(name="calculator", arguments={"expression": "1+2"})
    llm = ScriptedLLM([
        *[LLMResponse(tool_calls=[repeated]) for _ in range(6)],
        LLMResponse(content="做不下去了。"),
    ])
    agent, _orch = build_agent(llm)
    result = agent.run("算一下 1 加 2")
    scenario.outcome = result.outcome
    scenario.details["repeated_call_count"] = result.repeated_call_count
    scenario.details["stop_reason"] = result.meta.get("stop_reason")
    scenario.recovered = (result.repeated_call_count >= 1) and (result.outcome in {
        "answered", "insufficient_evidence", "budget_exceeded"
    })
    return scenario


def scenario_unknown_tool() -> Scenario:
    """模型调用了不存在的工具。必须是类型化失败，不能让异常穿透到主循环。"""

    scenario = Scenario(
        name="unknown_tool",
        description="模型调用了注册表里不存在的工具",
        injected_fault="ToolCall(name='nonexistent_tool')",
        expected_behavior="类型化错误（tool_not_found），Agent 继续或诚实停止",
    )
    llm = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall(name="nonexistent_tool", arguments={})]),
        LLMResponse(content="工具不存在，我不能继续。"),
    ])
    agent, _orch = build_agent(llm)
    result = agent.run("调用一个不存在的工具")
    scenario.outcome = result.outcome
    scenario.details["recovery_stats"] = result.meta.get("recovery_stats")
    scenario.recovered = result.outcome in {"answered", "insufficient_evidence"}
    return scenario


def scenario_corrupted_checkpoint() -> Scenario:
    """断点状态损坏。恢复必须显式失败（ValidationError），不能反序列化出一个半对的状态。"""

    scenario = Scenario(
        name="corrupted_checkpoint",
        description="HITL 断点状态被篡改/损坏",
        injected_fault="从外部写入与 schema 不符的检查点",
        expected_behavior="ValidationError 显式失败，不渲染半对状态",
    )
    store = InMemoryCheckpointStore()
    state = AgentState(run_id="run-corrupt", question="test")
    store.save(Checkpoint(run_id="run-corrupt", step=0, state=state.to_dict()))

    # 直接污染存储里的那条记录（绕过 from_dict 校验，模拟落盘后又被篡改）
    bad_record = Checkpoint(run_id="run-corrupt", step=0, state={}, schema_version=99)
    store._by_run["run-corrupt"] = [bad_record]

    agent = ResumableAgent(
        ScriptedLLM([]),
        ToolRegistry([]),
        store=store,
    )
    try:
        agent.resume("run-corrupt", None)
        scenario.crashed = False
        scenario.recovered = False
        scenario.details["note"] = "损坏版本被静默加载了——这是要被修复的行为"
    except ScoutError as error:
        scenario.recovered = error.code is ErrorCode.VALIDATION_FAILED
        scenario.details["error_code"] = error.code.value
    return scenario


def scenario_crash_mid_jsonl() -> Scenario:
    """JSONL 最后一行崩溃损坏：FileCheckpointStore 应该跳过损坏行、沿用前面的成功检查点。"""

    import tempfile
    scenario = Scenario(
        name="crash_mid_jsonl",
        description="JSONL 检查点最后一行写坏",
        injected_fault="最后一个 JSONL 行截断",
        expected_behavior="跳过损坏行、从上一份完好的检查点恢复，不丢全部状态",
    )
    with tempfile.TemporaryDirectory() as tmp:
        store = FileCheckpointStore(tmp)
        state = AgentState(run_id="run-crash", question="短任务")
        store.save(Checkpoint(run_id="run-crash", step=0, state=state.to_dict(), label="start"))
        path = Path(tmp) / "run-crash.jsonl"
        # 手动写入一行 "半截" JSON
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"schema_version": 1, "run_id": "run-crash", "step": 1, ')

        latest = store.latest("run-crash")
        scenario.recovered = latest is not None and latest.step == 0
        scenario.details["recovered_step"] = latest.step if latest else None
    return scenario


def scenario_hitl_timeout_reject() -> Scenario:
    """审批超时且没人处理：默认 fail-closed，自动拒绝并继续，绝不自动放行。"""

    from scout.tools.actions import build_action_tools
    scenario = Scenario(
        name="hitl_timeout_reject",
        description="HITL 审批超时",
        injected_fault="timeout_policy=REJECT + approval_deadline=0",
        expected_behavior="自动拒绝该调用并继续，副作用不发生",
    )
    registry = ToolRegistry([*build_default_tools(), *build_action_tools()])
    mail = LLMResponse(tool_calls=[ToolCall(name="send_email", arguments={"to": "a@b.com", "subject": "周报", "body": "…"})])
    agent = ResumableAgent(
        ScriptedLLM([mail, LLMResponse(content="那我先不发邮件，改用其他方式。")]),
        registry,
        timeout_policy=TimeoutPolicy.REJECT,
        approval_deadline_seconds=0.0,
    )
    first = agent.start("发周报")
    resumed = agent.resume(first.run_id, None)  # 没有决策，且已超时
    scenario.outcome = resumed.status.value
    scenario.details["reject_source"] = resumed.state.meta["approval_log"][-1]["source"] if resumed.state.meta.get("approval_log") else None
    scenario.recovered = (resumed.status.value == "completed") and (resumed.state.meta.get("decisions_rejected") == 1)
    return scenario


# —— 主流程 ——


SCENARIOS: list[Callable[[], Scenario]] = [
    scenario_provider_timeout_once,
    scenario_provider_persistent_failure,
    scenario_malformed_tool_args,
    scenario_repeat_call_loop,
    scenario_unknown_tool,
    scenario_corrupted_checkpoint,
    scenario_crash_mid_jsonl,
    scenario_hitl_timeout_reject,
]


def main() -> int:
    results: list[Scenario] = []
    for runner in SCENARIOS:
        try:
            scenario = runner()
        except Exception as exc:  # noqa: BLE001 - 这就是我们要量化的"裸异常"
            scenario = Scenario(
                name=runner.__name__,
                description="",
                injected_fault="",
                expected_behavior="",
                recovered=False,
                crashed=True,
                exception=repr(exc)[:300],
            )
            scenario.typed_failure = False
        results.append(scenario)

    # 汇总
    total = len(results)
    typed = sum(1 for item in results if item.typed_failure and not item.crashed)
    recovered = sum(1 for item in results if item.recovered and not item.crashed)
    crashed = sum(1 for item in results if item.crashed)

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_scenarios": total,
        "typed_failure_rate": round(typed / total, 4),
        "recovery_rate_within_budget": round(recovered / total, 4),
        "uncaught_crash_count": crashed,
        "scenarios": [
            {
                "name": item.name,
                "description": item.description,
                "injected_fault": item.injected_fault,
                "expected_behavior": item.expected_behavior,
                "outcome": item.outcome or None,
                "recovered": item.recovered,
                "crashed": item.crashed or bool(item.exception),
                "exception": item.exception or None,
                "details": item.details or None,
            }
            for item in results
        ],
    }

    out_dir = Path(__file__).resolve().parents[1] / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fault_injection.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_dir / "fault_injection.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"已生成 {json_path}")
    print(f"已生成 {md_path}")
    print(f"类型化失败率：{typed}/{total} = {report['typed_failure_rate']:.1%}")
    print(f"预算内恢复率：{recovered}/{total} = {report['recovery_rate_within_budget']:.1%}")
    print(f"未捕获裸异常：{crashed}")
    return 0 if crashed == 0 and typed == total else 1


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 故障注入报告",
        "",
        "> 用脚本化故障在可预期的位置注入失败，验证自愈编排的行为。",
        "> 数据由 `scripts/fault_injection.py` 生成，可离线复现。",
        "",
        "## 汇总",
        "",
        "| 指标 | 值 | 解读 |",
        "|---|---|---|",
        f"| 场景数 | {report['total_scenarios']} | 注入的故障类型数 |",
        f"| 类型化失败率 | {report['typed_failure_rate']:.1%} | 失败结束时是否带稳定错误码 |",
        f"| 预算内恢复率 | {report['recovery_rate_within_budget']:.1%} | 重试 ≤ 预算后完成或优雅降级 |",
        f"| 未捕获裸异常 | {report['uncaught_crash_count']} | 崩溃逃逸（应为 0） |",
        "",
        "## 场景明细",
        "",
        "| 场景 | 注入的故障 | 期望行为 | 实际结果 | 恢复 |",
        "|---|---|---|---|---|",
    ]
    for item in report["scenarios"]:
        recovered = "✓" if item["recovered"] else ("✗" if not item["recovered"] and not item["exception"] else "✗✗")
        outcome = item["outcome"] or (item["exception"] or "—")
        lines.append(
            f"| `{item['name']}` | {item['injected_fault']} | {item['expected_behavior']} | {outcome} | {recovered} |"
        )
    lines += [
        "",
        "> 这不是“系统不会出错”的证明——是“系统出错时**败得可控、败得可解释**”的证明。",
        "> 任何一行 “✗” 都表示那一条机制的分支尚未实现或还有 bug，应该直接被提 issue。",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
