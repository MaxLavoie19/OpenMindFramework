from collections.abc import Sequence
from typing import Protocol

from openmind.inference.model.answer import Answer
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.rule.model.clause import Clause


class Prover(Protocol):
    """Answering a question from a set of clauses, within a budget.

    A port, because there is more than one way to answer and no single way is best at everything. Resolution
    answers questions about what follows from rules, and is the only one that can also hand back a new rule. A
    solver answers questions that turn on arithmetic, which resolution is poor at. A stateless service fills it,
    given the clauses it runs."""

    def ask(self, clauses: Sequence[Clause], goal: Clause, budget: InferenceBudget) -> Answer: ...
