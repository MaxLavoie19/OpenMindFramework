import logging
from collections.abc import Sequence
from itertools import combinations, islice
from typing import TYPE_CHECKING

from openmind.inference.constant.refusal_constant import REFUSED
from openmind.inference.model.case_index import CaseIndex
from openmind.inference.model.example import Example
from openmind.inference.model.hypothesis import Hypothesis
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal

if TYPE_CHECKING:  # The learner imports this one, so naming it back at run time would close a ring.
    from openmind.inference.service.refusal_learner import RefusalLearner

logger = logging.getLogger(__name__)


class HypothesisTester:
    """Tries candidate bodies and says what each came to.

    **All the work and none of the deciding.** What to try next, what to keep and which rules to end up with are
    questions about everything tried so far; whether *this* body turns away a legal move and which cases it
    refuses are questions about the body alone. Only the second kind is here, which is why it can be asked
    anywhere — in this process or in twenty others — and why the answers are worth keeping.

    **It enumerates its own candidates rather than being handed them.** A body is a combination of a case's own
    readings, so saying which combinations is a case, a size and a range — three numbers and a list of readings.
    Handing over the forty-odd thousand bodies themselves costs more than testing them, and the measured
    experience of parallel rule learners is that generating candidates was never the expensive part: doing it in
    parallel bought under five per cent, because what costs is how many there are to try.

    **Coverage is measured against the whole pool, not against what is still unaccounted for.** What a body
    refuses is a fact about the body and the cases; what is still wanted changes every time a rule is kept. Asked
    the first way the answer keeps, and one body found from one case can account for a case nobody was looking at
    when it was written down — which is the reuse that makes a shared table worth holding.

    It keeps nothing: built once, it is given the readings and the cases on every call."""

    def __init__(self, learner: "RefusalLearner") -> None:
        # Given one rather than making one: it answers with the learner's own notion of what a case holds and of
        # what the game allows, and a tester quietly using a different one would agree with nobody.
        self._learner = learner

    def tried(
        self,
        offered: Sequence[Literal],
        size: int,
        pool: Sequence[Example],
        guard: CaseIndex,
        start: int = 0,
        stop: int | None = None,
        deadline: float | None = None,
        beyond: set[frozenset] | None = None,
    ) -> tuple[Hypothesis, ...]:
        """Every body of that many of those readings, from `start` to `stop`, tried.

        The range is over the combinations in the order `combinations` gives them, which is fixed by the order of
        `offered` — so two callers naming the same range mean the same bodies, and a range is a piece of work
        that can be handed anywhere without handing over the work itself.

        A body that turns away something the game allows is not measured for coverage. Nothing will use it, and
        the reason to carry it at all is so that the next case to reach the same body does not pay for the same
        test.

        `beyond` is what slipped one condition shorter, where the caller has it. A body containing a part that
        already passed the guard refuses no more than that part and costs more to say, so it is skipped without
        being tested — which is what makes searching past the first working size affordable. The range still
        runs over every combination in order, so two callers naming the same range still mean the same bodies;
        what changes is which of them are put to the guard."""
        found: list[Hypothesis] = []
        for chosen in islice(combinations(offered, size), start, stop):
            if deadline is not None and self._learner.clock() >= deadline:
                break
            # Keyed by the conditions as a clause holds them, which is how the table keys a body. Keyed by
            # their denials instead, the lookup never matches and every body past the first size is
            # skipped as dominated — the prune turns into a search that stops one size early and spends
            # the time anyway.
            if not self._learner.worth_trying(frozenset(chosen), beyond):
                continue
            clause = Clause((Literal(REFUSED, ()), *(one.denied for one in chosen)))
            if self._learner.slips(clause, guard):
                found.append(Hypothesis(clause, frozenset(), True))
                continue
            found.append(Hypothesis(clause, self.refusing(clause, pool), False))
        return tuple(found)

    def refusing(self, clause: Clause, pool: Sequence[Example]) -> frozenset[int]:
        """Where in the pool the cases it refuses sit."""
        return frozenset(number for number, one in enumerate(pool) if self._learner.covers(clause, one))

    def how_many(self, offered: Sequence[Literal], size: int) -> int:
        """How many bodies of that size there are to try, so a caller can cut them into ranges.

        Worked out rather than counted, since counting means enumerating them, which is the thing being divided
        up in the first place."""
        held, count = len(offered), 1
        if size > held:
            return 0
        for number in range(size):
            count = count * (held - number) // (number + 1)
        return count
