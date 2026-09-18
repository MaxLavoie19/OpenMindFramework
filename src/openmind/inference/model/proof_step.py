from dataclasses import dataclass

from openmind.inference.model.formula import Formula


@dataclass(frozen=True, slots=True)
class ProofStep:
    """One step of a proof: its number, the rule it applies as the prover names it (`asserted`, `mp`, `unit-resolution`,
    `quant-inst`, `induction`, ...), the numbers of the steps it uses, and the formula it concludes."""

    number: int
    rule: str
    premises: tuple[int, ...]
    formula: Formula
