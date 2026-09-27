from dataclasses import dataclass

from openmind.inference.model.chance import Chance
from openmind.inference.model.derivation import Derivation
from openmind.statement.model.clause import Clause

#: What the engine can say about a question.
#:
#: There are three, not two, and the third is the important one. A prover that must answer yes or no will answer
#: one of them when it knows neither, and a guess dressed as a conclusion is worse than no conclusion: it is acted
#: on. Saying *unknown*, and saying why, leaves the question open and points at what would settle it.
PROVED, DISPROVED, UNKNOWN = "proved", "disproved", "unknown"


@dataclass(frozen=True, slots=True)
class Answer:
    """What asking a question came to.

    `derivations` holds **every** way found of reaching it within the budget, not the first. A chance cannot be
    worked out from one proof, and neither can whether a conclusion has one reason behind it or several — and a
    conclusion with two independent reasons is worth more than the same conclusion with one, which is a thing
    worth being able to see.

    `reason` accompanies an unknown and says which kind it is: the budget ran out, or nothing in the clauses bears
    on the question. Those call for different things — more time, or more to reason from."""

    goal: Clause
    status: str
    derivations: tuple[Derivation, ...] = ()
    chance: Chance | None = None
    seconds: float = 0.0
    reason: str | None = None

    @property
    def proved(self) -> bool:
        return self.status == PROVED

    @property
    def settled(self) -> bool:
        """Whether it came to anything at all."""
        return self.status in (PROVED, DISPROVED)

    @property
    def readable(self) -> str:
        said = f"{self.goal.readable}: {self.status}"
        if self.reason:
            said += f", {self.reason}"
        if self.chance is not None:
            said += f", chance {self.chance.readable}"
        return said
