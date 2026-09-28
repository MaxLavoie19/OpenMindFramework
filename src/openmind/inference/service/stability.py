import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: What a term whose value never moves is worth as a way of telling positions apart.
#:
#: Nothing, and it is said as nothing rather than left as a division nobody guarded. A constant column carries no
#: information about which position is which, so however steady it is it steers nothing.
UNVARYING = 0.0


class Stability:
    """How steady a term is: how much it varies over a game against how much it varies from one position to
    the next.

    **A heuristic is something to steer by, and a term nobody can steer by is a term the search should reach
    last.** Two ways of being useless look alike to a fit until it has enough rows to tell them apart. A term
    that never moves says nothing about which position is which. A term that leaps about between one position
    and the next says something, but nothing anybody can act on: a move cannot be chosen for its effect on a
    number that would have jumped anyway. What is wanted is a term that differs a lot across a game and little
    between neighbours, and the ratio of those two variances is that, in one number.

    Clune's general game player measured this and called such terms *stable*, and used the measure to choose
    which features to keep. It is used both ways here, and the two are not the same claim.

    **Ordering costs nothing to be wrong about.** Every term still reaches the search; steadiness only decides
    which it reaches first, exactly as the reasoned seeds do, and a steady term that earns nothing is priced
    out like any other.

    **Dropping is a claim, and it is made only where the arithmetic makes it.** A term whose value never
    varies over everything walked carries no information about which position is which — not *probably*
    useless, but carrying nothing, since a constant column and the fit's own constant say the same thing
    twice. Beyond that, `keeping` is a caller's budget: the steadiest so many, because a search has only so
    many generations and spending them on terms nobody can steer by is spending them on nothing. That is a
    judgement, so it is the caller's and it is off unless asked for.

    Nothing here knows a game. It is handed numbers in the order a walk produced them and gives a number
    back."""

    def of(self, runs: Sequence[Sequence[Sequence[float]]]) -> tuple[float, ...]:
        """How steady each term is, in the terms' order.

        `runs` is one entry per walk, each a list of positions in the order they were walked, each of those
        the terms' values at that position. Several walks rather than one long one because a game ends: the
        step from the last position of a game to the first of the next is not a step anything took, and
        counting it would make every term look wilder than it is.

        A term that never varies over the walks is worth nothing to steer by, however steady, and is given
        nothing rather than a division by zero dressed up as a large number."""
        counted = self._counted(runs)
        if counted is None:
            return ()
        total, between, steps = counted
        found = []
        for at, spread in enumerate(total):
            moved = between[at] / steps if steps else 0.0
            found.append(UNVARYING if spread <= 0.0 else (spread / moved if moved > 0.0 else spread))
        return tuple(found)

    def steadiest(self, runs: Sequence[Sequence[Sequence[float]]], terms: Sequence[object]) -> tuple[object, ...]:
        """Those terms, steadiest first, every one of them kept.

        The order alone, for a caller that wants the search to reach the steady ones sooner and to go on
        reaching all of them."""
        steady = self.of(runs)
        if len(steady) != len(terms):
            logger.debug("Ordering nothing by steadiness: %d terms against %d measured", len(terms), len(steady))
            return tuple(terms)
        order = sorted(range(len(terms)), key=lambda at: -steady[at])
        return tuple(terms[at] for at in order)

    def kept(
        self,
        runs: Sequence[Sequence[Sequence[float]]],
        terms: Sequence[object],
        keeping: int | None = None,
    ) -> tuple[tuple[object, ...], tuple[object, ...]]:
        """Those terms split into the ones worth a generation and the ones that are not, steadiest first.

        A term that never varied is always dropped, and that is arithmetic rather than an opinion: a column
        that is the same everywhere tells nothing apart, and the fit already has a constant.

        `keeping` is how many of the rest to keep, the steadiest first, None for all of them. It is a budget
        and not a threshold: how steady is steady enough has no answer that travels between games, and how
        many generations there are to spend is something the caller knows."""
        steady = self.of(runs)
        if len(steady) != len(terms):
            logger.debug("Keeping every term: %d terms against %d measured", len(terms), len(steady))
            return tuple(terms), ()
        order = sorted(range(len(terms)), key=lambda at: -steady[at])
        varying = [at for at in order if steady[at] > UNVARYING]
        flat = [at for at in order if steady[at] <= UNVARYING]
        held = varying if keeping is None else varying[: max(0, keeping)]
        left = set(held)
        return tuple(terms[at] for at in held), tuple(terms[at] for at in (*varying, *flat) if at not in left)

    def _counted(
        self, runs: Sequence[Sequence[Sequence[float]]]
    ) -> tuple[list[float], list[float], int] | None:
        """The variance of each term over every position walked, the squared step it took between neighbours,
        and how many steps there were; None where nothing was walked."""
        seen = [one for one in runs if len(one) > 0]
        if not seen:
            return None
        width = len(seen[0][0]) if seen[0] else 0
        if not width:
            return None
        values: list[list[float]] = [[] for _ in range(width)]
        between = [0.0] * width
        steps = 0
        for run in seen:
            for at, position in enumerate(run):
                if len(position) != width:
                    return None
                for term in range(width):
                    values[term].append(float(position[term]))
                if at:
                    steps += 1
                    for term in range(width):
                        between[term] += (float(position[term]) - float(run[at - 1][term])) ** 2
        total = []
        for held in values:
            middle = sum(held) / len(held)
            total.append(sum((one - middle) ** 2 for one in held) / len(held))
        return total, between, steps
