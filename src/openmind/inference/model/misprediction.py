from dataclasses import dataclass

from openmind.inference.model.chance import Chance
from openmind.rule.model.literal import Literal


@dataclass(frozen=True, slots=True)
class Misprediction:
    """An outcome that did not match what was expected, and which kind of not-matching it was.

    Two things go wrong here and they want opposite repairs, so running them together is how a game with chance in
    it comes to look like a game whose rules keep breaking.

    An outcome **nothing predicted at all** is a clause missing. Something happens that the model has no account
    of, and no amount of adjusting what is already there will produce it: something has to be learned.

    An outcome predicted **at the wrong rate** is nothing broken. The clause is right about what can happen and
    wrong about how often, and the repair is counting — which needs no discovery, no new evidence beyond what is
    already arriving, and no attention.

    `expected` is what the clauses gave it, None where they gave it nothing. `held` is how often it actually came
    about. `unforeseen` marks the first kind."""

    outcome: Literal
    expected: Chance | None
    held: Chance
    unforeseen: bool = False
    where: object | None = None

    @property
    def readable(self) -> str:
        if self.unforeseen:
            return f"{self.outcome.predicate}: nothing predicted it, and it held {self.held.readable}"
        expected = self.expected.readable if self.expected is not None else "nothing"
        return f"{self.outcome.predicate}: expected {expected}, held {self.held.readable}"
