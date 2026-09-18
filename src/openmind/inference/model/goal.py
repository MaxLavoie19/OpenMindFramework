from dataclasses import dataclass

from openmind.inference.model.formula import Formula
from openmind.inference.model.theory import Axiom, Theory


@dataclass(frozen=True, slots=True)
class Goal:
    """What to prove: a conclusion, from premises and the definitions of theories."""

    conclusion: Formula
    premises: tuple[Axiom, ...] = ()
    theories: tuple[Theory, ...] = ()
