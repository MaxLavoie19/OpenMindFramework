from dataclasses import dataclass

from openmind.inference.model.goal import Goal
from openmind.inference.model.proof_step import ProofStep


@dataclass(frozen=True, slots=True)
class Proof:
    """What proving a goal found: `proved`, `disproved`, `independent` or `unknown`; with `proved`, the steps deriving a
    contradiction from the premises, the theories and the conclusion denied, the last step concluding false, and with
    `disproved` the same from the conclusion itself; with `independent`, a counterexample, a model of the premises where
    the conclusion is false, as the prover writes it; how long it took; and why the prover couldn't tell, with `unknown`."""

    goal: Goal
    status: str
    steps: tuple[ProofStep, ...] = ()
    seconds: float = 0.0
    counterexample: str | None = None
    reason: str | None = None
