from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class HypothesisTest:
    """A hypothesis and how it fared on validation data it wasn't discovered on. Effects are mean within-state
    differences in advantage between matching and other actions, on the discovery and on the validation states
    (validation_effect is None without any validation state). p_value comes from a one-sided sign-flip permutation test
    in the discovered direction, q_value is its Benjamini-Hochberg adjustment over every hypothesis tested together, and
    validated means q_value is within the false discovery rate with the effect in the discovered direction."""

    action: str
    conditions: tuple[PythonRule, ...]
    direction: int
    discovery_effect: float
    discovery_states: int
    validation_effect: float | None
    validation_states: int
    p_value: float
    q_value: float
    validated: bool
