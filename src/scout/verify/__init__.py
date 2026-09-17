"""验证与防护：证据接地校验、拒答校准、检索侧注入防护。"""

from __future__ import annotations

from .grounding import (
    ABSTENTION_MARKERS,
    ABSTENTION_OUTCOMES,
    Claim,
    GroundingReport,
    SufficiencyEstimate,
    Verdict,
    abstained_from,
    check_grounding,
    estimate_sufficiency,
    extract_claims,
    is_abstention,
    verify_answer,
)
from .sanitize import (
    SanitizeReport,
    attribution_gate,
    detect_injection,
    neutralize,
    sanitize_text,
)

__all__ = [
    "ABSTENTION_MARKERS",
    "ABSTENTION_OUTCOMES",
    "Claim",
    "GroundingReport",
    "SanitizeReport",
    "SufficiencyEstimate",
    "Verdict",
    "abstained_from",
    "attribution_gate",
    "check_grounding",
    "detect_injection",
    "estimate_sufficiency",
    "extract_claims",
    "is_abstention",
    "neutralize",
    "sanitize_text",
    "verify_answer",
]
