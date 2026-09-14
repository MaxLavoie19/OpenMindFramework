from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A hypothesis about an action, as discovered: within the scope of its parent conditions, the actions matching all
    of its conditions have a higher (direction 1) or lower (direction -1) advantage than the other actions of the same
    state, by effect on average over that many states. visits and expected_value are the matching actions'; score ranks
    hypotheses of the same size; priority marks a payoff near the best or worst seen."""

    action: str
    conditions: tuple[PythonRule, ...]
    parent: tuple[PythonRule, ...]
    direction: int
    effect: float
    states: int
    visits: int
    expected_value: float
    priority: bool
    score: float
