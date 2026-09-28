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
    which features to keep. **Here it chooses no term's fate.** The design admits no plausibility filter —
    only a price per term, held-out rows, and games decide what survives — so what this buys is the order the
    search reaches things in, which is the same thing the reasoned seeds buy. A steady term that earns
    nothing is priced out exactly like any other.

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
        """Those terms, steadiest first, and the rest of them after in the order they came.

        Ordering and not choosing: every term given comes back, because what is measured here decides where
        the search looks first and never what it is allowed to look at."""
        steady = self.of(runs)
        if len(steady) != len(terms):
            logger.debug("Ordering nothing by steadiness: %d terms against %d measured", len(terms), len(steady))
            return tuple(terms)
        order = sorted(range(len(terms)), key=lambda at: -steady[at])
        return tuple(terms[at] for at in order)

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
