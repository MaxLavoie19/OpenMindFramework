import logging
import time
from collections.abc import Callable, Sequence

from openmind.inference.model.answer import DISPROVED, PROVED, UNKNOWN, Answer
from openmind.inference.model.derivation import Derivation
from openmind.inference.model.derivation_step import RESOLVED
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.derivation_builder import DerivationBuilder
from openmind.inference.service.proof_weigher import ProofWeigher
from openmind.inference.service.resolver import Resolver
from openmind.inference.service.subsumer import Subsumer
from openmind.statement.model.clause import Clause

logger = logging.getLogger(__name__)

#: How the question, denied, is named in a derivation.
DENIED = "the question denied"
AFFIRMED = "the question itself"

#: Why an answer is unknown. They want different things: one wants more time, the other more to reason from.
OUT_OF_TIME = "the time ran out"
OUT_OF_STEPS = "the steps ran out"
NOTHING_BEARS = "nothing among the clauses bears on it"


class GoalProver:
    """What a set of clauses says about a question, by denying it and looking for a contradiction.

    **Goal-driven**, because what follows from a set of rules is endless and only a little of it is ever wanted.
    Chaining forward derives everything and hopes the answer is in it; this starts from the question and only
    follows what could bear on it, so the work is spent where the asking was.

    **It collects every proof it can within the budget, not the first one.** This is the thing that shapes it,
    and it is not thoroughness for its own sake. How likely a conclusion is cannot be worked out from one proof.
    Neither can whether it has one reason behind it or several — and a conclusion supported two independent ways
    is worth more than the same conclusion supported one way, which is something worth being able to see. A prover
    that stops at the first proof throws all of that away before anyone can look at it.

    **It can answer that it does not know**, and says which kind of not knowing. A prover forced to say yes or no
    will say one of them when it knows neither, and a guess dressed as a conclusion is worse than no conclusion,
    because it gets acted on. Out of time wants more time; nothing bearing on the question wants more to reason
    from.

    Widest first: everything reachable in one step, then in two. A shallow answer is never left sitting behind a
    deep one, and the proofs come back simplest first.

    It keeps nothing: built once, it is given the clauses on every call."""

    def __init__(
        self,
        resolver: Resolver | None = None,
        subsumer: Subsumer | None = None,
        derivation_builder: DerivationBuilder | None = None,
        proof_weigher: ProofWeigher | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._resolver = Resolver() if resolver is None else resolver
        self._subsumer = Subsumer() if subsumer is None else subsumer
        self._derivations = DerivationBuilder() if derivation_builder is None else derivation_builder
        self._weigher = ProofWeigher() if proof_weigher is None else proof_weigher
        self._clock = clock

    def ask(self, clauses: Sequence[Clause], goal: Clause, budget: InferenceBudget) -> Answer:
        """What those clauses say about that question.

        Proved where denying it contradicts them. Otherwise disproved where asserting it does, which is a
        different and much stronger thing than failing to prove it. Otherwise unknown, with the reason."""
        started = self._clock()
        against, reason = self._search(clauses, self._denied(goal, DENIED), budget, started)
        if against:
            spent = self._clock() - started
            chance = self._weigher.chance(against, budget)
            logger.info(
                "%s: proved by %d derivation%s in %.3fs, chance %s",
                goal.readable, len(against), "" if len(against) == 1 else "s", spent, chance.readable,
            )
            self._log(against)
            return Answer(goal, PROVED, against, chance, spent)
        left = budget.seconds - (self._clock() - started)
        if left > 0.0:
            for_it, _ = self._search(clauses, self._denied(self._negated(goal), AFFIRMED), budget, started)
            if for_it:
                spent = self._clock() - started
                logger.info("%s: disproved by %d derivations in %.3fs", goal.readable, len(for_it), spent)
                return Answer(goal, DISPROVED, for_it, None, spent)
        spent = self._clock() - started
        logger.info("%s: unknown after %.3fs, %s", goal.readable, spent, reason)
        return Answer(goal, UNKNOWN, (), None, spent, reason)

    def _search(
        self, clauses: Sequence[Clause], support: Sequence[Clause], budget: InferenceBudget, started: float
    ) -> tuple[tuple[Derivation, ...], str]:
        """Every contradiction reachable from that support, widest first."""
        given = [
            (settled, self._derivations.given(settled))
            for settled in (self._resolver.settled(one) for one in clauses)
            if settled is not None
        ]
        frontier = [(one, self._derivations.given(one)) for one in support]
        found: list[Derivation] = []
        seen = {one.readable for one, _ in frontier}
        steps, apart, depth = 0, 0, 0
        reason = NOTHING_BEARS
        while frontier:
            if self._clock() - started >= budget.seconds:
                return tuple(found), OUT_OF_TIME
            if budget.steps is not None and steps >= budget.steps:
                return tuple(found), OUT_OF_STEPS
            if budget.depth is not None and depth >= budget.depth:
                return tuple(found), NOTHING_BEARS
            depth += 1
            wider: list[tuple[Clause, Derivation]] = []
            for clause, derivation in frontier:
                for other, its in (*given, *frontier):
                    apart += 1
                    steps += 1
                    for made, agreed in self._resolver.resolve(clause, other, apart):
                        settled = self._resolver.settled(made)
                        if settled is None:
                            continue
                        joined = self._derivations.joined(settled, RESOLVED, agreed, derivation, its)
                        if settled.empty:
                            found.append(joined)
                            if budget.derivations is not None and len(found) >= budget.derivations:
                                return tuple(found), reason
                            continue
                        if settled.readable in seen:
                            continue
                        seen.add(settled.readable)
                        wider.append((settled, joined))
                    if budget.steps is not None and steps >= budget.steps:
                        break
            frontier = wider
        return tuple(found), reason

    def _denied(self, goal: Clause, named: str) -> tuple[Clause, ...]:
        """The question denied, as clauses.

        A clause is a disjunction, and denying a disjunction denies every part of it — so a question of several
        literals is denied by a clause for each. A variable left free is read as "for any", which is what makes
        asking about any thing at all the same shape as asking about one."""
        return tuple(Clause((literal.denied,), 1.0, named) for literal in goal.literals)

    def _negated(self, goal: Clause) -> Clause:
        return Clause(tuple(one.denied for one in goal.literals), goal.probability, goal.name)

    def _log(self, derivations: Sequence[Derivation]) -> None:
        for derivation in derivations:
            for step in derivation.steps:
                logger.debug("%s", step.readable)
