from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class Observation:
    """What each player sees of a domain's states, as value rules reading a state and the parameter `player`: `hidden`
    gives the names of the variables that player can't see, and `completions`, reading what they see, gives the possible
    values of the hidden variables together, as a list of (values by name, probability). Both see `definitions`."""

    hidden: PythonRule
    completions: PythonRule
    definitions: PythonRule | None = None
