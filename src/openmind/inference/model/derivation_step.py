from dataclasses import dataclass

from openmind.inference.model.substitution import Substitution
from openmind.rule.model.clause import Clause

#: What a step did, as the engine names it.
GIVEN = "given"
RESOLVED = "resolved"
FACTORED = "factored"
EVALUATED = "evaluated"


@dataclass(frozen=True, slots=True)
class DerivationStep:
    """One step of reasoning: what it did, what it used, what it concluded, and what it had to take things to mean.

    The substitution is kept because without it a step cannot be checked. A rule about anything, used of one thing,
    concludes about that thing only because the step agreed it would — and a reader who cannot see the agreement
    cannot tell a sound step from a sleight of hand."""

    number: int
    rule: str
    premises: tuple[int, ...]
    clause: Clause
    substitution: Substitution = Substitution()

    @property
    def readable(self) -> str:
        used = f" from {', '.join(str(one) for one in self.premises)}" if self.premises else ""
        return f"{self.number}. {self.rule}{used}: {self.clause.readable}"
