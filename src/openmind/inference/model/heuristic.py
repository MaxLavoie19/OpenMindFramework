from dataclasses import dataclass

from openmind.inference.model.derivation import Derivation
from openmind.inference.model.parameter import Parameter
from openmind.statement.model.clause import Clause
from openmind.structure.model.value import Value

#: What a heuristic is trying to do. Optimal play maximises the win; predictive play says what a player will
#: actually do. Both are valuing a position or a move, and they are not the same valuation.
OPTIMAL, PREDICTIVE = "optimal", "predictive"


@dataclass(frozen=True, slots=True)
class Heuristic:
    """A way of valuing a position or a move, reasoned out of the rules rather than guessed at.

    `clause` is the logic: what has to hold for this to bear on a position. That is the engine's real
    contribution — *which rules apply*, in a form that can be read, resolved against and argued with.

    `parameters` are the numbers it could reason out, each with where it came from. They are starting points handed
    to the search, never answers: what a rule is finally worth in a context is fitted there.

    `aim` decides what it is fitted against, and that is the whole of the difference between the two. Optimal play
    is fitted on what the game paid; predictive play on what a player actually did. Fitting one against the other's
    target gives a model that is neither. `holder` is whose play it predicts, empty for optimal play.

    `derivation` is why it was proposed. A heuristic that plays badly can then be read back to the reasoning that
    suggested it, and the reasoning argued with — which is the thing a tuned number can never be."""

    about: Value
    clause: Clause
    kind: str
    derivation: Derivation
    aim: str = OPTIMAL
    holder: Value | None = None
    parameters: tuple[Parameter, ...] = ()

    @property
    def value(self) -> float:
        """What it starts the thing it is about at."""
        return self.parameters[0].initial if self.parameters else 0.0

    @property
    def readable(self) -> str:
        return f"{self.about} is worth {self.value:g}"
