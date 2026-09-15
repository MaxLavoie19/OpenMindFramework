from dataclasses import dataclass

from openmind.rule.model.rule import Rule


@dataclass(frozen=True, slots=True)
class Branch:
    """One possible outcome of an action: its probability and its effects, a Python script whose assignments to state
    variables make the new state, or a function of the state and the action's parameters giving the new state."""

    probability: float
    effects: Rule
