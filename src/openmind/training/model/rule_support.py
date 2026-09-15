from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class RuleSupport:
    """A value rule's term and how strongly each signal supports it: by signal name, the term's weight in the fit to that
    signal's targets, on standardized values so strengths compare between terms."""

    term: PythonRule
    strengths: tuple[tuple[str, float], ...]
