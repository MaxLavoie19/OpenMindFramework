from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule
from openmind.rule.model.rule import Rule


@dataclass(frozen=True, slots=True)
class Observation:
    """What each player sees of a domain's states, as rules reading a state and the parameter `player`: `hidden` gives
    the names of the variables that player can't see, and `completions`, reading what they see, gives the possible values
    of the hidden variables together, as a list of (values by name, probability). Both are Python source, seeing
    `definitions`, or the project's own functions, called with the state and the player."""

    hidden: Rule
    completions: Rule
    definitions: PythonRule | None = None
