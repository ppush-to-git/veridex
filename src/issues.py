"""Canonical Issue and Severity types used across all modules.

Every module reports findings as a list of Issue objects. The risk
aggregator combines them into a final decision. Because every module
appends to the same list, multiple issues from the same image show up
together instead of being suppressed by an early return.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Dict


class Severity(str, Enum):
    """How serious an issue is when it fires.

    INFO   - observational, no risk contribution
    LOW    - minor anomaly, small nudge to risk
    MEDIUM - notable issue, contributes meaningfully
    HIGH   - strong evidence of a problem
    CRITICAL - hard-fail signal (e.g. MRZ checksum invalid)
    """
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Numerical weight each severity contributes to the risk score.
SEVERITY_WEIGHT: Dict[Severity, float] = {
    Severity.INFO: 0.0,
    Severity.LOW: 5.0,
    Severity.MEDIUM: 15.0,
    Severity.HIGH: 30.0,
    Severity.CRITICAL: 60.0,
}


@dataclass
class Issue:
    """One finding produced by a module.

    Attributes
    ----------
    code : short machine-readable identifier, e.g. "MRZ_CHECKSUM_FAIL"
    module : which module raised it (ocr, validator, tampering, face)
    severity : Severity enum
    message : human-readable explanation shown to the officer
    evidence : optional dict of supporting data (scores, bbox, etc.)
    """
    code: str
    module: str
    severity: Severity
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def weight(self) -> float:
        return SEVERITY_WEIGHT[self.severity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "module": self.module,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": self.evidence,
        }
