import logging
from collections.abc import Iterable, Sequence

from openmind.inference.model.hypothesis import Hypothesis
from openmind.statement.model.clause import Clause

logger = logging.getLogger(__name__)


class HypothesisTable:
    """What has been tried and what it came to, held once for everybody trying.

    **Sharing this is not an optimisation, it is the thing that makes many searchers worth having.** Parallel
    constraint-driven ILP measured both ways: workers given separate parts of the space and no way to tell each
    other what failed did so badly the arrangement was left out of the results, and workers searching the same
    space without talking were *slower* than one worker alone. What made it superlinear was sharing what had
    been ruled out. So the table is the design, and the processes are a detail of it.

    **Two things are kept and they earn their place differently.** A body that refuses nothing is a prune —
    conditions only narrow, so nothing built on it can refuse anything either. A body that turns away a legal
    move prunes nothing at all, since a search by increasing size has already tried every subset of it and every
    subset slips too; it is kept so that the next case to reach the same body does not pay for it again.

    **Bounded from the start.** Every published coverage cache has run into memory rather than time — the
    technique works and the table is what grows without limit, so the policy for what to forget is part of the
    design rather than something to add when it falls over. What is forgotten is what was tried and failed,
    oldest first: a useful hypothesis is what all of this is for, and re-testing a failure costs one test while
    losing a success costs a rule.

    It keeps what it is told and nothing else: no readings, no cases, no positions."""

    def __init__(self, most: int = 200_000) -> None:
        #: How many hypotheses may be remembered at once. A caller's budget: it trades memory against how often
        #: a failure is tried again, and it is not a fact about any game.
        self._most = most
        self._held: dict[frozenset, Hypothesis] = {}
        #: Those kept only as a memory of failure, oldest first, which is what is given up when room is needed.
        #: Insertion order is the dict's own, so nothing extra is carried to know which is oldest.
        self._tried = 0
        self._spared = 0

    def __len__(self) -> int:
        return len(self._held)

    @property
    def tried(self) -> int:
        """How many hypotheses have been offered to it, whether or not they were new."""
        return self._tried

    @property
    def spared(self) -> int:
        """How many of those it had already, which is the work the sharing saved."""
        return self._spared

    def knows(self, clause: Clause) -> bool:
        """Whether that body has been tried, by anybody, against this pool."""
        return frozenset(clause.body) in self._held

    def asked(self, clause: Clause) -> Hypothesis | None:
        """What it came to when it was tried, or None where nobody has tried it."""
        return self._held.get(frozenset(clause.body))

    def tell(self, found: Iterable[Hypothesis]) -> int:
        """Takes in what a worker came back with, and says how many of them were new.

        A hypothesis already held is not written again: two workers reaching one body from two cases agree about
        what it refuses, because what it refuses is a fact about the body and the pool and not about who asked."""
        fresh = 0
        for one in found:
            self._tried += 1
            key = one.key
            if key in self._held:
                self._spared += 1
                continue
            self._held[key] = one
            fresh += 1
        self._forget()
        return fresh

    def useful(self) -> tuple[Hypothesis, ...]:
        """Those that refuse something and turn nothing away wrongly, briefest first.

        **What anything choosing a set of rules should be offered, and nothing else.** The one system that does
        this two-stage thing published its numbers: thirty-one thousand candidates, ten of them promising, and an
        exact answer in four seconds — where offering all thirty-one thousand as choices is the arrangement it
        contrasts itself against. Filtering here is what keeps the choosing possible."""
        return tuple(sorted((one for one in self._held.values() if one.useful), key=lambda one: one.size))

    def slipping(self, size: int) -> set[frozenset]:
        """The bodies of that many conditions that turn away something the game allows.

        What a search one condition longer needs, and the reason these are kept at all. A body is worth asking
        the guard about only where every part of it one condition shorter slipped; anything else is dominated
        by the part that did not. Kept as bodies rather than as hypotheses because that is all the asking
        needs, and because the caller is about to look each one up thousands of times."""
        return {key for key, one in self._held.items() if one.slips and len(key) == size}

    def covering(self, place: int) -> tuple[Hypothesis, ...]:
        """Those useful ones that refuse the case at that place, briefest first."""
        return tuple(one for one in self.useful() if place in one.refusing)

    def _forget(self) -> None:
        """Gives up remembered failures, oldest first, once there are more than may be held.

        Never a useful one. Forgetting a failure costs the one test that finds it again; forgetting a success
        costs a rule that nothing will propose a second time, since whatever proposed it has moved on."""
        if len(self._held) <= self._most:
            return
        over = len(self._held) - self._most
        giving = [key for key, one in self._held.items() if not one.useful][:over]
        for key in giving:
            del self._held[key]
        if len(self._held) > self._most:
            logger.info(
                "Holding %d hypotheses, over the %d asked for: every one of them is worth keeping",
                len(self._held), self._most,
            )
        elif giving:
            logger.debug("Forgot %d tried-and-failed hypotheses to stay within %d", len(giving), self._most)

    def readable(self) -> str:
        useful = self.useful()
        return (
            f"{len(self._held)} hypotheses tried, {len(useful)} of them worth choosing from; "
            f"{self._spared} of {self._tried} offers were ones somebody had already tried"
        )
