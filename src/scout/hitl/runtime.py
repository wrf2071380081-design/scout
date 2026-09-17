"""可中断、可恢复、可时间旅行的 Agent 运行时。

**它和普通 Agent 循环的区别只有一个，但影响一切：状态必须能离开内存。**

普通 Agent 循环把状态放在栈上——messages 列表、当前步数、已执行的调用——
进程活着就一切正常。一旦要求"人可以中途介入，几分钟后再继续"，
这些栈上的东西就必须搬到可以持久化的地方，于是三个问题接踵而来：

1. **状态放哪。** 见 :mod:`scout.hitl.checkpoints`：状态是纯数据，
   append-only 落盘，恢复靠重放而不是靠反序列化对象图。
2. **什么时候停。** 见 :mod:`scout.hitl.interrupts.InterruptPolicy`：
   按风险分级决定，而不是一刀切。
3. **恢复时怎么不错。** 见 :class:`~scout.hitl.interrupts.SideEffectLedger`：
   以调用槽位为幂等键，已产生副作用的槽位只回放、不重放。

::: 三个必须说清的语义

``suspend`` 而不是 ``block``
    中断时运行**返回**而不是阻塞等待。挂起后进程可以退出，状态已经落盘；
    审批可能来自几分钟后的另一个进程、另一台机器。用 ``input()`` 阻塞式确认
    在设计上就是错的——它把"人的响应时间"和"进程生命周期"绑在了一起。

``resume`` 而不是 ``rerun``
    恢复是从断点继续，不是从头重跑。区别体现在副作用上：重跑会再发一次邮件。
    这也是为什么幂等键建在"槽位"（第几步第几次调用）而不是"参数"上。

``fork`` 而不是 ``rollback``
    从历史检查点分叉出一个新的 run，而不是把已发生的运行改回去。
    已经发生的事改不回去（邮件已经发出去了），所以正确的操作是**新建一条时间线**，
    而不是伪装历史不存在。审计上这一点很重要。

::: 一个容易做错的细节：恢复点是"那次调用"，不是"那一步"

中断往往发生在某一步的调用循环中间：例如模型在这一步要调用三个工具，
第一个发邮件（已批准并执行）、第二个删除文件（待审批，停在这）、
第三个更新记录（还没跑到）。恢复时必须**从第二个调用继续**，而不是
把这一步的三个调用整体重跑一遍（那会把邮件再发一次），也不是跳过到下一步。

所以状态里存了 ``pending_step = {step, tool_calls, next_index, request}``——
"进行中的那一步"的完整快照，以及应该从哪里继续。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ..config import AgentSettings, Settings, get_settings
from ..errors import ErrorCode, ScoutError
from ..llm.base import ChatMessage, LLMClient, LLMRequest, ToolCall
from ..prompts import AGENT_SYSTEM_PROMPT
from ..trace import StepKind, Trace
from ..tools.registry import ToolRegistry, ToolResult, ToolSpec
from .checkpoints import (
    AgentState,
    Checkpoint,
    CheckpointStore,
    InMemoryCheckpointStore,
    message_to_dict,
    messages_from_dicts,
    messages_to_dicts,
    tool_call_from_dict,
    tool_call_to_dict,
)
from .interrupts import (
    HumanDecision,
    InterruptPolicy,
    InterruptRequest,
    RiskAssessment,
    RiskLevel,
    SideEffectLedger,
    TimeoutPolicy,
    ToolRiskProfile,
)


class RunStatus(str, Enum):
    """一次运行的状态。"""

    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    BUDGET_EXCEEDED = "budget_exceeded"
    FAILED = "failed"


@dataclass(slots=True)
class RunOutcome:
    """运行结果。要么跑完了，要么停在某处等人。"""

    run_id: str
    status: RunStatus
    answer: str = ""
    state: AgentState | None = None
    request: InterruptRequest | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_human(self) -> bool:
        return self.status is RunStatus.AWAITING_APPROVAL

    def to_dict(self, *, include_messages: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "status": self.status.value,
            "answer": self.answer,
            "needs_human": self.needs_human,
            "request": self.request.to_dict() if self.request else None,
            "meta": self.meta,
        }
        if include_messages and self.state is not None:
            payload["messages"] = self.state.messages
        return payload


def build_default_policy(registry: ToolRegistry) -> InterruptPolicy:
    """从工具声明里推导默认审批策略。

    工具的副作用属性写在工具定义里（``ToolSpec.side_effects`` / ``compensator``），
    策略在这里统一翻译成风险等级——这样"新增一个工具要不要审批"这件事
    跟着工具一起定义，而不用去改策略代码。
    """

    profiles: dict[str, ToolRiskProfile] = {}
    for name in registry.names():
        spec = registry.get(name)
        if spec is None:
            continue
        if not spec.side_effects:
            profiles[name] = ToolRiskProfile(
                level=RiskLevel.SAFE, note=f"{name} 是只读工具", auto_approve=True
            )
        elif spec.compensator:
            profiles[name] = ToolRiskProfile(
                level=RiskLevel.CONFIRM,
                note=f"{name} 有副作用但可补偿（{spec.compensator}）",
            )
        else:
            profiles[name] = ToolRiskProfile(
                level=RiskLevel.REQUIRED, note=f"{name} 有副作用且不可补偿"
            )
    return InterruptPolicy(profiles)


class ResumableAgent:
    """可中断、可恢复的 Agent 运行时。

    :param store: 检查点存储。默认内存实现；生产应换成
        :class:`~scout.hitl.checkpoints.FileCheckpointStore`，
        否则进程一退状态就没了，"跨进程恢复"这件事不成立。
    :param timeout_policy: 审批超时的默认动作。默认 REJECT（fail-closed）。
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        policy: InterruptPolicy | None = None,
        store: CheckpointStore | None = None,
        budget: AgentSettings | None = None,
        settings: Settings | None = None,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        timeout_policy: TimeoutPolicy = TimeoutPolicy.REJECT,
        approval_deadline_seconds: float = 900.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.budget = budget or self.settings.agent
        self.llm = llm
        self.registry = registry
        self.policy = policy or build_default_policy(registry)
        self.store: CheckpointStore = store or InMemoryCheckpointStore()
        self.system_prompt = system_prompt
        self.timeout_policy = timeout_policy
        self.approval_deadline_seconds = approval_deadline_seconds

    # —— 入口：开始 ——

    def start(
        self,
        question: str,
        *,
        run_id: str | None = None,
        history: Sequence[ChatMessage] | None = None,
    ) -> RunOutcome:
        """开始一次新的运行。"""

        identifier = run_id or f"run-{uuid.uuid4().hex[:12]}"
        state = AgentState(run_id=identifier, question=question)
        state.messages = messages_to_dicts(
            [ChatMessage(role="system", content=self.system_prompt), *(history or []),
             ChatMessage(role="user", content=question)]
        )
        state.meta["budget"] = {
            "max_steps": self.budget.max_steps,
            "max_tool_calls": self.budget.max_tool_calls,
        }
        # lineage（血缘）= 这条时间线最早的 run_id。分叉会换新 run_id，
        # 但**血缘不变**——副作用账本的幂等键建在它上面，所以
        # "从过去分叉出一个新 run"不会把已经发出去的副作用再发一次。
        state.meta["lineage"] = identifier
        self._init_meta(state)
        self._checkpoint(state, label="start")
        return self._drive(state)

    # —— 入口：恢复 ——

    def resume(
        self,
        run_id: str,
        decision: HumanDecision | None = None,
        *,
        now: float | None = None,
    ) -> RunOutcome:
        """从断点继续。

        ``decision`` 为 ``None`` 时，仅当待审批请求已经超时才允许继续
        （按 :class:`TimeoutPolicy` 自动处理）。**不能在没有审批结果时静默继续**——
        那等于把"等待审批"变成了"默认放行"。
        """

        state = self.load_state(run_id)
        if state is None:
            raise ScoutError(f"找不到运行 {run_id}", code=ErrorCode.VALIDATION_FAILED)

        if state.status not in {"awaiting_approval", "interrupted"}:
            raise ScoutError(
                f"运行 {run_id} 当前状态为 {state.status}，无需恢复",
                code=ErrorCode.VALIDATION_FAILED,
            )

        pending = state.pending_step
        if not pending or not pending.get("request"):
            raise ScoutError(
                f"运行 {run_id} 没有待处理的审批请求", code=ErrorCode.VALIDATION_FAILED
            )
        request = self._request_from_payload(pending["request"])

        if decision is not None:
            if decision.request_id != request.request_id:
                # 决策与请求对不上时必须显式失败：错配的审批会批准一件没被审过的事。
                raise ScoutError(
                    f"审批结果 {decision.request_id} 与待处理请求 {request.request_id} 不匹配",
                    code=ErrorCode.VALIDATION_FAILED,
                )
        else:
            if not request.expired:
                raise ScoutError(
                    f"运行 {run_id} 仍在等待审批（请求 {request.request_id}），且未超时",
                    code=ErrorCode.VALIDATION_FAILED,
                )
            decision = self.timeout_policy.to_decision(request, now=now)
            state.meta["last_timeout_policy"] = self.timeout_policy.value

        state.decisions.append(decision.to_dict())
        state.status = "running"
        self._record_approval(state, decision, request)
        self._checkpoint(state, label=f"resume_step_{state.step}")
        return self._drive(state)

    # —— 查询 ——

    def load_state(self, run_id: str) -> AgentState | None:
        checkpoint = self.store.latest(run_id)
        return AgentState.from_dict(checkpoint.state) if checkpoint else None

    def history(self, run_id: str) -> list[Checkpoint]:
        """完整检查点历史。**时间旅行的原料。**"""

        return self.store.history(run_id)

    def pending_request(self, run_id: str) -> InterruptRequest | None:
        state = self.load_state(run_id)
        if state is None:
            return None
        request = (state.pending_step or {}).get("request")
        return self._request_from_payload(request) if request else None

    def effects(self, run_id: str) -> SideEffectLedger:
        state = self.load_state(run_id)
        return SideEffectLedger.from_dicts(state.effects if state else [])

    def rollback_plan(self, run_id: str) -> list[dict[str, Any]]:
        """补偿计划：逆序撤销已执行的副作用。"""

        return self.effects(run_id).compensation_plan(run_id=run_id)

    def fork(self, run_id: str, at_step: int, *, new_run_id: str | None = None) -> str:
        """从历史检查点分叉出一条新运行线。

        注意 ``fork`` 不会撤销已发生的副作用——那不可能。它只是给出一条
        从过去某个状态重新演化的路径。为了**分叉不会把已经发出的副作用再发一次**：

        1. 分叉继承**最新的副作用账本**（不是所选检查点里的旧账本），
           因为"副作用是否已发生"是关于世界的客观事实，与历史状态无关；
        2. 分叉保留**血缘 run_id**，副作用的幂等键建在它上面，所以新 run
           遇到同一个调用槽位时只会回放、不会重放。

        要撤销副作用请用 :meth:`rollback_plan`——回滚是一个**补偿**动作，
        不是把历史删掉。
        """

        checkpoint = self.store.at(run_id, at_step)
        if checkpoint is None:
            raise ScoutError(
                f"运行 {run_id} 不存在 step={at_step} 的检查点", code=ErrorCode.VALIDATION_FAILED
            )
        state = AgentState.from_dict(checkpoint.state)
        latest = self.store.latest(run_id)
        if latest is not None:
            # 副作用账本继承自最新检查点：世界是客观发生的，不随历史状态回退。
            latest_state = AgentState.from_dict(latest.state)
            state.effects = list(latest_state.effects)
        forked_id = new_run_id or f"{run_id}-fork{at_step}-{uuid.uuid4().hex[:6]}"
        state.run_id = forked_id
        state.status = "running"
        state.answer = ""
        state.pending_step = {}
        state.meta = dict(state.meta)
        # 血缘保留原 run_id：新时间线的副作用幂等键与原时间线一致。
        state.meta["lineage"] = state.meta.get("lineage") or run_id
        state.meta["forked_from"] = {"run_id": run_id, "step": at_step}
        self._checkpoint(state, label=f"fork_from_step_{at_step}")
        return forked_id

    def run_fork(self, run_id: str, at_step: int, *, new_run_id: str | None = None) -> RunOutcome:
        """分叉并继续跑。"""

        forked_id = self.fork(run_id, at_step, new_run_id=new_run_id)
        state = self.load_state(forked_id)
        assert state is not None
        return self._drive(state)

    # —— 主循环 ——

    def _drive(self, state: AgentState) -> RunOutcome:
        """把状态推进到"完成"或"停在待审批处"。"""

        trace = Trace(question=state.question)
        ledger = SideEffectLedger.from_dicts(state.effects)
        deadline = time.monotonic() + self.budget.deadline_seconds

        # 断点续跑：如果上次停在某一步的中间，先把那一步跑完，再进入主循环。
        # 跳过这一步会让"上一步的后半段"整个被吃掉，是 resume 最常见也最隐蔽的错误。
        if state.pending_step:
            outcome = self._finish_pending_step(state, ledger, trace)
            if outcome is not None:
                return outcome

        while state.step < self.budget.max_steps:
            if time.monotonic() > deadline:
                state.status = "budget_exceeded"
                state.meta["stop_reason"] = "deadline_exceeded"
                self._checkpoint(state, label="deadline_exceeded")
                return self._outcome(state, RunStatus.BUDGET_EXCEEDED, trace)

            state.step += 1
            self._checkpoint(state, label=f"step_{state.step}_planned")

            messages = messages_from_dicts(state.messages)
            with trace.step("decide", StepKind.PLAN, index=state.step) as step:
                response = self.llm.complete(
                    LLMRequest(
                        messages=messages,
                        tools=self.registry.schemas(),
                        task="decide",
                        context={"question": state.question, "step": state.step},
                        deadline=deadline,
                    )
                )
                state.input_tokens += response.usage.input_tokens
                state.output_tokens += response.usage.output_tokens
                step.outputs = {"tool_calls": len(response.tool_calls), "chars": len(response.content)}

            if not response.has_tool_calls:
                state.answer = response.content.strip()
                state.status = "completed"
                state.meta["stop_reason"] = "model_final_answer"
                state.pending_step = {}
                self._checkpoint(state, label="completed")
                return self._outcome(state, RunStatus.COMPLETED, trace)

            state.messages.append(
                message_to_dict(
                    ChatMessage(
                        role="assistant",
                        content=response.content,
                        tool_calls=list(response.tool_calls),
                    )
                )
            )

            outcome = self._drive_calls(state, ledger, response.tool_calls, 0, trace)
            if outcome is not None:
                return outcome

            self._checkpoint(state, label=f"step_{state.step}_done")

        state.status = "budget_exceeded"
        state.meta["stop_reason"] = "max_steps_exhausted"
        self._checkpoint(state, label="max_steps_exhausted")
        return self._outcome(state, RunStatus.BUDGET_EXCEEDED, trace)

    def _drive_calls(
        self,
        state: AgentState,
        ledger: SideEffectLedger,
        tool_calls: Sequence[ToolCall],
        start_index: int,
        trace: Trace,
    ) -> RunOutcome | None:
        """处理本步的所有调用。任何一次被审批门拦下都会返回中断结果。"""

        for index in range(start_index, len(tool_calls)):
            state.pending_step = {
                "step": state.step,
                "tool_calls": [tool_call_to_dict(call) for call in tool_calls],
                "next_index": index,
                "request": (state.pending_step or {}).get("request", {}),
            }
            outcome = self._handle_call(state, ledger, tool_calls[index], index, trace)
            if outcome is not None:
                # 待审批状态必须在返回前落盘，否则"进程在这里被 kill 掉"之后
                # 恢复时看到的是上一份没中断的状态——审批就永远等不到结果。
                self._checkpoint(
                    state,
                    label=f"awaiting_{outcome.request.tool_name if outcome.request else 'approval'}",
                )
                return outcome
            self._enforce_tool_budget(state, ledger)
            if state.status == "budget_exceeded":
                state.pending_step = {}
                self._checkpoint(state, label="max_tool_calls_exhausted")
                return self._outcome(state, RunStatus.BUDGET_EXCEEDED, trace)

        state.pending_step = {}
        return None

    def _finish_pending_step(
        self, state: AgentState, ledger: SideEffectLedger, trace: Trace
    ) -> RunOutcome | None:
        """把中断的那一步跑完：先处理被审批的调用，再继续该步剩余调用。"""

        pending = state.pending_step
        tool_calls = [tool_call_from_dict(item) for item in pending.get("tool_calls", [])]
        index = int(pending.get("next_index", 0))
        request = self._request_from_payload(pending["request"])
        decision = self._decision_for(state, request.request_id)

        if index >= len(tool_calls):
            # 防御：pending_step 损坏时宁可显式失败，不要静默吞掉。
            state.pending_step = {}
            raise ScoutError(
                f"运行 {state.run_id} 的断点状态不完整（索引越界）",
                code=ErrorCode.VALIDATION_FAILED,
            )

        # 先处理被审批的那一次调用
        outcome = self._handle_call(state, ledger, tool_calls[index], index, trace, decision=decision)
        if outcome is not None:
            return outcome

        # 继续该步剩余的调用
        outcome = self._drive_calls(state, ledger, tool_calls, index + 1, trace)
        if outcome is not None:
            return outcome
        state.pending_step = {}
        self._checkpoint(state, label=f"step_{state.step}_done")
        return None

    def _handle_call(
        self,
        state: AgentState,
        ledger: SideEffectLedger,
        call: ToolCall,
        index: int,
        trace: Trace,
        *,
        decision: HumanDecision | None = None,
    ) -> RunOutcome | None:
        """处理一次工具调用。返回非 None 表示需要挂起等审批。

        顺序是刻意的（不能换）：

        1. **先查幂等账本**：这个槽位已经产生过副作用 → 它只是"只读回放"，
           不需要审批、也不需要执行——因为它不会改变任何东西。
           把它放到审批门之前，时间旅行分叉才不会因为一个"只会回放"的调用
           被反复要求重新审批。
        2. **再查审批策略**：真正要执行的调用才需要审批。
        """

        slot = SideEffectLedger.slot_of(self._lineage(state), state.step, index)

        # —— 幂等闸门：这个槽位已经产生过副作用 → 只回放，不重放 ——
        if ledger.already_executed(slot):
            record = ledger.by_slot(slot)
            assert record is not None
            state.meta["effects_replayed"] += 1
            state.messages.append(
                message_to_dict(
                    ChatMessage(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=record.observation,
                    )
                )
            )
            state.call_log.append(
                {
                    "step": state.step,
                    "index": index,
                    "tool": call.name,
                    "replayed": True,
                    "effect_key": record.key,
                }
            )
            return None

        assessment: RiskAssessment = self.policy.assess(call.name, call.arguments)

        if assessment.needs_human and decision is None:
            request = self._make_request(state, call, index, assessment)
            state.pending_step["request"] = request.to_dict()
            state.status = "awaiting_approval"
            state.meta["approvals_required"] += 1
            with trace.step("interrupt", StepKind.HUMAN, name=call.name) as step:
                step.outputs = request.to_dict()
            return RunOutcome(
                run_id=state.run_id,
                status=RunStatus.AWAITING_APPROVAL,
                state=state,
                request=request,
                meta={
                    "risk_level": assessment.level.value,
                    "risk_reason": assessment.reason,
                    "step": state.step,
                    "call_index": index,
                },
            )

        effective_arguments = dict(call.arguments)
        if decision is not None:
            if not decision.approved:
                state.meta["decisions_rejected"] += 1
                state.messages.append(
                    message_to_dict(
                        ChatMessage(
                            role="tool",
                            name=call.name,
                            tool_call_id=call.id,
                            content=(
                                f"[审批被拒绝] 调用 {call.name} 未获批准。"
                                f"审批人留言：{decision.comment or '（无）'}。"
                                "请改用不需要审批的方式完成任务，或如实说明无法完成。"
                            ),
                        )
                    )
                )
                state.call_log.append(
                    {
                        "step": state.step,
                        "index": index,
                        "tool": call.name,
                        "approved": False,
                        "decided_by": decision.decided_by,
                    }
                )
                return None
            if decision.edited_arguments:
                effective_arguments = dict(decision.edited_arguments)
                state.meta.setdefault("argument_edits", []).append(
                    {
                        "step": state.step,
                        "index": index,
                        "tool": call.name,
                        "original": dict(call.arguments),
                        "edited": dict(effective_arguments),
                        "by": decision.decided_by,
                    }
                )

        spec = self.registry.get(call.name)
        has_side_effect = self._has_side_effect(spec, assessment)

        with trace.step("tool", StepKind.TOOL, name=call.name, arguments=effective_arguments) as step:
            result: ToolResult = self.registry.call(call.name, effective_arguments)
            step.outputs = {"ok": result.ok, "error_code": result.error_code or None}
            step.metrics = {"side_effect": has_side_effect}

        observation = result.to_observation()
        state.messages.append(
            message_to_dict(
                ChatMessage(role="tool", name=call.name, tool_call_id=call.id, content=observation)
            )
        )
        state.call_log.append(
            {
                "step": state.step,
                "index": index,
                "tool": call.name,
                "ok": result.ok,
                "side_effect": has_side_effect,
            }
        )

        if has_side_effect and result.ok:
            record = ledger.record(
                run_id=state.run_id,
                step=state.step,
                call_index=index,
                tool_name=call.name,
                arguments=effective_arguments,
                compensator=(spec.compensator if spec else "") or "",
                compensator_payload=self._compensator_payload(call.name, effective_arguments, result),
                observation=observation,
            )
            # 副作用落账后立刻持久化：如果进程在下一步之前就崩了，
            # 恢复时的幂等闸门仍然能认出它已经执行过。这是"崩溃即中断"的保障。
            state.effects = ledger.to_dicts()
            self._checkpoint(state, label=f"effect_{state.step}_{index}")
            state.meta["last_effect_key"] = record.key
        return None

    # —— 辅助 ——

    @staticmethod
    def _lineage(state: AgentState) -> str:
        """副作用幂等键建在这条时间线的**血缘**上，而不是当前 run_id 上。

        分叉会换 run_id 但保留血缘，所以新时间线遇到同一个调用槽位时
        只会回放、不会重放已发生的副作用。
        """

        return str(state.meta.get("lineage") or state.run_id)

    @staticmethod
    def _has_side_effect(spec: ToolSpec | None, assessment: RiskAssessment) -> bool:
        """副作用判定：工具显式声明，或被风险评估认为"不是 SAFE"。"""

        if spec is not None and spec.side_effects:
            return True
        return assessment.level is not RiskLevel.SAFE

    @staticmethod
    def _compensator_payload(
        tool_name: str, arguments: dict[str, Any], result: ToolResult
    ) -> dict[str, Any]:
        """构造补偿动作所需的载荷。

        优先用工具自己返回的撤销凭据（比如新建记录的 id），
        没有就退回原始参数。撤销往往需要"执行时才产生的标识"——
        这是补偿事务与简单回滚最容易踩坑的地方。
        """

        payload: dict[str, Any] = {"arguments": dict(arguments)}
        for key in ("undo_token", "record_id", "message_id", "ticket_id", "action_id", "id"):
            if key in result.data:
                payload["undo_token"] = result.data[key]
                break
        return payload

    def _decision_for(self, state: AgentState, request_id: str) -> HumanDecision | None:
        for item in reversed(state.decisions):
            if str(item.get("request_id")) == request_id:
                return HumanDecision.from_dict(item)
        return None

    def _make_request(
        self,
        state: AgentState,
        call: ToolCall,
        index: int,
        assessment: RiskAssessment,
    ) -> InterruptRequest:
        return InterruptRequest(
            request_id=f"req-{uuid.uuid4().hex[:10]}",
            run_id=state.run_id,
            step=state.step,
            call_index=index,
            tool_name=call.name,
            arguments=dict(call.arguments),
            assessment=assessment,
            deadline=time.time() + self.approval_deadline_seconds,
        )

    @staticmethod
    def _request_from_payload(payload: dict[str, Any]) -> InterruptRequest:
        assessment = RiskAssessment(
            level=RiskLevel(str(payload.get("risk_level", RiskLevel.CONFIRM.value))),
            reason=str(payload.get("risk_reason", "")),
        )
        return InterruptRequest(
            request_id=str(payload.get("request_id", "")),
            run_id=str(payload.get("run_id", "")),
            step=int(payload.get("step", 0)),
            call_index=int(payload.get("call_index", 0)),
            tool_name=str(payload.get("tool_name", "")),
            arguments=dict(payload.get("arguments") or {}),
            assessment=assessment,
            created_at=float(payload.get("created_at", 0.0)),
            deadline=payload.get("deadline") if payload.get("deadline") is not None else None,
        )

    def _record_approval(
        self, state: AgentState, decision: HumanDecision, request: InterruptRequest
    ) -> None:
        """把审批结果同步进状态（保留审批人信息，供审计）。"""

        state.meta.setdefault("approval_log", []).append(
            {
                "request_id": request.request_id,
                "tool": request.tool_name,
                "risk_level": request.assessment.level.value,
                "approved": decision.approved,
                "decided_by": decision.decided_by,
                "source": decision.source,
                "comment": decision.comment,
                "edited": bool(decision.edited_arguments),
            }
        )

    def _enforce_tool_budget(self, state: AgentState, ledger: SideEffectLedger) -> None:
        if len(state.call_log) > self.budget.max_tool_calls:
            state.status = "budget_exceeded"
            state.meta["stop_reason"] = "max_tool_calls_exhausted"

    @staticmethod
    def _init_meta(state: AgentState) -> None:
        state.meta.setdefault("decisions_rejected", 0)
        state.meta.setdefault("effects_replayed", 0)
        state.meta.setdefault("approvals_required", 0)

    def _checkpoint(self, state: AgentState, *, label: str) -> None:
        self.store.save(
            Checkpoint(run_id=state.run_id, step=state.step, state=state.to_dict(), label=label)
        )

    def _outcome(self, state: AgentState, status: RunStatus, trace: Trace) -> RunOutcome:
        return RunOutcome(
            run_id=state.run_id,
            status=status,
            answer=state.answer,
            state=state,
            meta={
                "steps": state.step,
                "tool_calls": len(state.call_log),
                "effects": len(state.effects),
                "effects_replayed": state.meta.get("effects_replayed", 0),
                "approvals_required": state.meta.get("approvals_required", 0),
                "decisions_rejected": state.meta.get("decisions_rejected", 0),
                "argument_edits": len(state.meta.get("argument_edits", [])),
                "stop_reason": state.meta.get("stop_reason", ""),
                "trace_durations_ms": trace.kind_durations(),
            },
        )


__all__ = ["ResumableAgent", "RunOutcome", "RunStatus", "build_default_policy"]
