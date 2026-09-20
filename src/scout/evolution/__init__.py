"""评测与自进化层：让系统能"判断自己好不好"，并据此改进。

**这一层与评测层的区别**（容易混，但职责完全不同）：

- ``scout.evaluation``（已有）：**跑分**。给定数据集与配置，算出指标、出报告。
  它是"测量仪器"。
- ``scout.evolution``（本层）：**用测出来的东西改进系统**。三个部件：

  - :class:`~scout.evolution.judge.LLMJudge`：当词法指标不够用时，
    用模型做语义评分（并处理它自身的位置/长度偏见）。
  - :class:`~scout.evolution.flywheel.FailureMiner` /
    :class:`~scout.evolution.flywheel.ReviewQueue`：把线上失败变成评测集增量。
  - :class:`~scout.evolution.tuner.ParamTuner`：把指标结构翻译成参数建议。

**这一层唯一不可妥协的纪律：所有自动变更都要有留痕与回滚路径。**
裁判分数要经过校准（:class:`~scout.evolution.judge.JudgeCalibration`）、
样本入库要经过复核（:class:`~scout.evolution.flywheel.ReviewQueue`）、
参数建议要带样本量与置信度（:class:`~scout.evolution.tuner.ParamSuggestion`）。
理由很朴素：**自进化系统一旦开始自我合理化，它坏得比手动系统更彻底**——
每次错误的自动变更都会被"新数据"重新论证一遍。
"""

from __future__ import annotations

from .flywheel import (
    CandidateCase,
    FailureMiner,
    FailureReason,
    FlywheelLedger,
    FlywheelTurn,
    MineStats,
    ReviewQueue,
    candidate_id,
    export_preference_pairs,
)
from .judge import JudgeCalibration, JudgeVerdict, LLMJudge
from .tuner import ParamSuggestion, ParamTuner

__all__ = [
    "CandidateCase",
    "FailureMiner",
    "FailureReason",
    "FlywheelLedger",
    "FlywheelTurn",
    "JudgeCalibration",
    "JudgeVerdict",
    "LLMJudge",
    "MineStats",
    "ParamSuggestion",
    "ParamTuner",
    "ReviewQueue",
    "candidate_id",
    "export_preference_pairs",
]
