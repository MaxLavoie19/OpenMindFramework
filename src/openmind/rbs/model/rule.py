from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

from openmind.rbs.model.python_rule import PythonRule
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value


class ConstraintRule(Protocol):
    """Whether an action with these parameter values is legal in the state."""

    def __call__(self, state: State, **parameters: Value) -> bool: ...


class ValuesRule(Protocol):
    """The values a parameter can take in the state, in order; a value given twice counts once."""

    def __call__(self, state: State) -> Iterable[Value]: ...


class EffectsRule(Protocol):
    """The state an action with these parameter values leads to."""

    def __call__(self, state: State, **parameters: Value) -> State: ...


class EndingRule(Protocol):
    """Why a finished game ended, or None while it goes on."""

    def __call__(self, state: State) -> str | None: ...


class RecordRule(Protocol):
    """A game's record, from the initial state and the actions played, or None."""

    def __call__(self, state: State, actions: Sequence[Action]) -> str | None: ...


#: A game's rule: Python source OMF compiles, or a function the project wrote and gives OMF, of one of the shapes
#: above. A function takes the state first and the parameters it needs by name; an effects function gives the next state.
type Rule = PythonRule | Callable[..., object]
