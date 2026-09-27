import logging
from collections.abc import Sequence

from openmind.inference.model.hypothesis import Hypothesis
from openmind.inference.service.description_length import DescriptionLength
from openmind.statement.model.clause import Clause

logger = logging.getLogger(__name__)


class ConstraintSelector:
    """Which of the constraints worth having are worth keeping, and what is deliberately left over.

    **The thing nothing here could do before.** A set's size has been settled by a repair policy — change the
    policy and the count moved from about fourteen to about two hundred and twenty, which is a number set by an
    afternoon's decision rather than by the evidence. It could not be otherwise: accounting for everything was a
    hard condition, so "fewer rules, and some of it unexplained" was not a worse answer, it was not an answer.

    **Here it is a trade that can go either way.** A constraint costs what it takes to say. It saves what it
    takes to point out the legal moves among the candidates it no longer lets through. Where the second beats
    the first, it pays; where it does not, the candidates it would have accounted for are left over and *said
    to be* left over.

    **Which means the answer grows with the evidence, and that is the point rather than a wrinkle.** The
    constraints are stated once and the legal moves are pointed out in every position, so a further rule earns
    its keep across all of them at once. Measured over one walk: two constraints pay at the first position,
    three by the second, eight by the sixth — and seven more that the old arrangement would have kept are
    declined.

    **Nothing offered here may refuse a move the game allows.** That is not priced and never will be: a code
    would have nothing left to name such a move with, and the project's own reason is older — it is a move OMF
    will never make and will never hear it could have made. `Hypothesis.useful` has already dropped them.

    Greedy, and knowingly. With a data term of this shape the objective is not submodular — adding a constraint
    can change what the next one is worth — so taking the best each time carries no guarantee. What it carries
    instead is a number: every step says what it bought, so a run can be read rather than trusted.

    It keeps nothing: built once, it is given the hypotheses and the positions on every call."""

    def __init__(self, code: DescriptionLength | None = None) -> None:
        self._code = DescriptionLength() if code is None else code

    def selected(
        self,
        offered: Sequence[Hypothesis],
        among: Sequence[int],
        legal: Sequence[int],
        positions: int = 1,
    ) -> tuple[tuple[Clause, ...], frozenset[int]]:
        """The constraints that pay for themselves, and the refused candidates nobody accounts for.

        `among` is how many candidates each position holds and `legal` how many of them the game allows, in the
        order the hypotheses' places run. The places a hypothesis refuses are places in the whole pool, so the
        two sequences say where one position's stretch of it ends and the next begins.

        `positions` is how many boards the evidence stands for. A constraint is stated once and points out
        legal moves in every position it applies to, so what it saves is multiplied by how many there are —
        which is why the number of rules worth buying rises as more is seen.

        **Said this way rather than by piling every legal move ever seen onto one board.** That was the first
        wiring of it, and after two hundred and forty-eight positions the guard held thousands of legal moves
        while a position holds fourteen thousand candidates — so the moment the constraints were good enough to
        cut the survivors below the number of legal moves, there was nothing left to name them among and the
        price went to infinity. It declined constraints *because they worked*, and explained sixty-three per
        cent of what the game refuses where keeping everything explained ninety-seven.

        The leftovers come back rather than being dropped, because a set that deliberately explains less is
        only honest if it says what it has stopped explaining. Counted as a failure it would read as the
        learner having got worse."""
        taken: list[Hypothesis] = []
        refusing: frozenset[int] = frozenset()
        price = self._priced((), refusing, among, legal, positions)
        started = price
        while True:
            best, found, cheapest = None, refusing, price
            for one in offered:
                if one in taken:
                    continue
                widened = refusing | one.refusing
                if widened == refusing:
                    continue
                asked = self._priced(
                    [held.clause for held in (*taken, one)], widened, among, legal, positions
                )
                if asked < cheapest:
                    best, found, cheapest = one, widened, asked
            if best is None:
                break
            taken.append(best)
            refusing, price = found, cheapest
            logger.debug(
                "Kept a constraint of %d conditions, leaving %.0f bits from %.0f", best.size, price, started
            )
        logger.info(
            "Kept %d of %d constraints worth having, at %.0f bits from %.0f; %d refused candidates are left "
            "unaccounted for and are not a failure",
            len(taken), len(offered), price, started, sum(among) - len(refusing) - sum(legal),
        )
        return tuple(one.clause for one in taken), refusing

    def _priced(
        self,
        clauses: Sequence[Clause],
        refusing: frozenset[int],
        among: Sequence[int],
        legal: Sequence[int],
        positions: int,
    ) -> float:
        """What that set costs to say, plus what it leaves to be said across every position it stands for."""
        pairs = self._pairs(refusing, among, legal)
        return self._code.theory(clauses) + positions * sum(
            self._code.naming(survivors, allowed) for survivors, allowed in pairs
        )

    def left_over(self, refusing: frozenset[int], among: Sequence[int], legal: Sequence[int]) -> int:
        """How many candidates the game refuses that the chosen constraints do not account for.

        **A third count and not a second.** What a set lets through has always meant "it failed to account for
        this"; what is left over here means "it was not worth accounting for". Reported as the first, a set
        that deliberately explains less reads as a learner that has got worse — which would be the same
        mistake, in the other direction, as the one this is here to fix."""
        return sum(among) - len(refusing) - sum(legal)

    def _pairs(
        self, refusing: frozenset[int], among: Sequence[int], legal: Sequence[int]
    ) -> list[tuple[int, int]]:
        """How many candidates survive in each position, beside how many of them are legal."""
        found, at = [], 0
        for held, allowed in zip(among, legal, strict=True):
            refused = sum(1 for place in refusing if at <= place < at + held)
            found.append((held - refused, allowed))
            at += held
        return found
