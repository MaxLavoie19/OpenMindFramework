from dataclasses import dataclass

from openmind.inference.model.derivation_step import DerivationStep
from openmind.statement.model.clause import Clause


@dataclass(frozen=True, slots=True)
class Derivation:
    """One way a conclusion was reached, readable back to the clauses it started from.

    A conclusion without this is a number somebody has to take on trust. With it, whoever doubts the conclusion can
    be shown the steps, and what they disagree with is a step — which is something that can be argued about, and
    something the engine can be corrected on.

    `chances` names the clauses it used that hold only some of the time. This is what makes it more than a record:
    two derivations of the same thing are two independent reasons only if they lean on different ones, and a
    conclusion supported twice over by the same doubtful clause is supported once."""

    conclusion: Clause
    steps: tuple[DerivationStep, ...] = ()
    chances: tuple[str, ...] = ()

    @property
    def rests_on(self) -> tuple[str, ...]:
        """The names of the clauses it was given, in the order they were first used."""
        found: dict[str, None] = {}
        for step in self.steps:
            if step.rule == "given" and step.clause.name:
                found.setdefault(step.clause.name, None)
        return tuple(found)

    @property
    def readable(self) -> str:
        return "\n".join(step.readable for step in self.steps)
