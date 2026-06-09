"""Validated records used throughout EntailForge."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class LogicExample:
    """One binary logical entailment example."""

    id: str
    premises: List[str]
    hypothesis: str
    label: int
    rule: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Example id cannot be empty.")
        if len(self.premises) < 2:
            raise ValueError("Each example requires at least two premises.")
        if any(not premise.strip() for premise in self.premises):
            raise ValueError("Premises cannot be empty.")
        if not self.hypothesis.strip():
            raise ValueError("Hypothesis cannot be empty.")
        if self.label not in (0, 1):
            raise ValueError("Label must be 0 or 1.")
        if not self.rule.strip():
            raise ValueError("Rule cannot be empty.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LogicExample":
        if not isinstance(payload, dict):
            raise ValueError("Example must be a JSON object.")
        return cls(
            id=str(payload.get("id", "")),
            premises=list(payload.get("premises", [])),
            hypothesis=str(payload.get("hypothesis", "")),
            label=int(payload.get("label", -1)),
            rule=str(payload.get("rule", "")),
        )
