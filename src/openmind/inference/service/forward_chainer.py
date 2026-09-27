import logging
import time
from collections.abc import Callable, Sequence

from openmind.inference.model.derivation import Derivation
from openmind.inference.model.derivation_step import FACTORED, RESOLVED
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.derivation_builder import DerivationBuilder
from openmind.inference.service.resolver import Resolver
from openmind.inference.service.subsumer import Subsumer
from openmind.statement.model.clause import Clause

logger = logging.getLogger(__name__)

#: Why it stopped.
SETTLED = "nothing further follows"
OUT_OF_TIME = "the time ran out"
OUT_OF_STEPS = "the steps ran out"


class ForwardChainer:
    """Everything that follows from a set of clauses, until nothing further does.

    This is what replaces the stand-in's chaining, and the difference is not that it derives more. It is where the
    reasoning lives. The stand-in had eight kinds of conclusion written out as branches of Python, so the only way
    to conclude a ninth kind of thing was for somebody to edit the engine. Here a kind of conclusion is a clause
    like any other: OMF concludes something new by being given a rule or by learning one, and the engine is not
    touched.

    Two things keep it from running away. A conclusion something already held makes pointless is not kept — which
    includes the case nothing structural can see, where a bound already had is tighter than the one just derived,
    and without which the same thing is rederived a little weaker for ever. And a clause whose computed conditions
    have failed is idle and goes no further.

    It stops when nothing new follows, or when the budget the caller set runs out. It sets no limit of its own:
    what follows from a set of rules is endless, and how much of it is worth having is not the engine's to judge.

    It keeps nothing: built once, it is given the clauses on every call."""

    def __init__(
        self,
        resolver: Resolver | None = None,
        subsumer: Subsumer | None = None,
        derivation_builder: DerivationBuilder | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._resolver = Resolver() if resolver is None else resolver
        self._subsumer = Subsumer() if subsumer is None else subsumer
        self._derivations = DerivationBuilder() if derivation_builder is None else derivation_builder
        self._clock = clock

    def chain(self, clauses: Sequence[Clause], budget: InferenceBudget) -> tuple[Derivation, ...]:
        """Every conclusion that follows, each carrying the steps that reached it.

        The clauses given come back among them, as conclusions resting on themselves, so what was assumed and what
        was worked out can be told apart by reading the steps rather than by remembering."""
        deadline = self._clock() + budget.seconds
        held: list[tuple[Clause, Derivation]] = []
        for clause in clauses:
            settled = self._resolver.settled(clause)
            if settled is not None and not self._subsumer.redundant(settled, [one for one, _ in held]):
                held.append((settled, self._derivations.given(settled)))
        given, steps, apart, waiting = len(held), 0, 0, 0
        stopped = SETTLED
        while waiting < len(held):
            if self._clock() >= deadline:
                stopped = OUT_OF_TIME
                break
            if budget.steps is not None and steps >= budget.steps:
                stopped = OUT_OF_STEPS
                break
            mine, its = held[waiting]
            waiting += 1
            for theirs, their_derivation in list(held):
                apart += 1
                steps += 1
                for made, agreed in self._resolver.resolve(mine, theirs, apart):
                    self._keep(held, made, self._derivations.joined(made, RESOLVED, agreed, its, their_derivation))
                if budget.steps is not None and steps >= budget.steps:
                    break
            for made, agreed in self._resolver.factors(mine):
                self._keep(held, made, self._derivations.joined(made, FACTORED, agreed, its))
        found = tuple(derivation for _, derivation in held)
        logger.info(
            "Chained %d conclusions from %d clauses in %d steps: %s",
            len(found) - given,
            given,
            steps,
            stopped,
        )
        for clause, _ in held[given:]:
            logger.debug("Concluded %s", clause.readable)
        return found

    def _keep(self, held: list[tuple[Clause, Derivation]], made: Clause, derivation: Derivation) -> None:
        """Keep the conclusion where it adds something to what is already held.

        Nothing held is ever taken away. Dropping a conclusion that a later, stronger one makes pointless would
        keep the set smaller, and it would also pull the ground out from under the loop walking that same list.
        What matters for stopping is that nothing pointless goes *in*, and that is what this does."""
        settled = self._resolver.settled(made)
        if settled is None or self._subsumer.redundant(settled, [one for one, _ in held]):
            return
        held.append((settled, Derivation(settled, derivation.steps, derivation.chances)))
