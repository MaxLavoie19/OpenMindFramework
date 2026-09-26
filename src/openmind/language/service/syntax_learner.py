import collections
import logging
import math
from collections.abc import Sequence

from openmind.inference.service.information import Information
from openmind.language.model.grammar import Grammar
from openmind.language.model.shape import Shape
from openmind.language.model.sort import Sort

logger = logging.getLogger(__name__)


class SyntaxLearner:
    """The shape of a game's notation, from the notations and nothing else.

    **It never sees what happened.** Sorts and shapes are a fact about the strings, and keeping them independent
    of the happenings is what lets the two measurements check each other rather than agree by construction. It
    also means a game's records can be read for their structure before a single rule is known.

    **Sorts are found by what joining them saves.** Every character starts as its own sort, and the pair whose
    joining shortens the whole description most is joined, until no joining shortens it. Characters standing in
    the same places collapse many layouts into one when joined, which is the saving; characters that do not cost
    more to tell apart afterwards and are left alone. So `a` and `b` come together and `b` and `B` do not, and
    nothing had to be told that capitals mean anything.

    **It is meant for a small language.** Chess notation's couple of dozen characters is the size it is for, and
    on that it leaves sorts over-split — chess's ranks divide into the ones pieces sit on and the ones pawns run
    through, which is a fact about play rather than about the notation. Where the joining stalls, because two
    joins that would only pay together are never taken one at a time, that is a reason to read the notation with
    something larger behind the same call rather than to grow this.

    It keeps nothing: built once, it is given the notations on every call."""

    def __init__(self, information: Information | None = None) -> None:
        # The one thing it measures in, given rather than made, so that every learner counting bits
        # counts the same bits.
        self._information = Information() if information is None else information

    def learn(self, said: Sequence[str]) -> Grammar:
        """The sorts and shapes those notations fall into."""
        counted = collections.Counter(said)
        if not counted:
            return Grammar((), ())
        sorts = [frozenset({one}) for one in sorted({one for notation in counted for one in notation})]
        cost = self._cost(counted, sorts)
        logger.info(
            "Starting from %d characters, each its own sort, at %.0f bits", len(sorts), cost
        )
        while True:
            joined = self._best_joining(counted, sorts, cost)
            if joined is None:
                break
            sorts, cost = joined
        found = tuple(Sort(one) for one in sorts)
        shapes = self._shapes(counted, found)
        logger.info(
            "Settled on %d sorts (%s) and %d shapes (%s) at %.0f bits",
            len(found), ", ".join(one.name for one in found),
            len(shapes), "; ".join(one.readable for one in shapes), cost,
        )
        return Grammar(found, shapes)

    def _best_joining(
        self, counted: collections.Counter, sorts: Sequence[frozenset[str]], cost: float
    ) -> tuple[list[frozenset[str]], float] | None:
        """The one joining of two sorts that saves most, or None where none saves anything."""
        best: list[frozenset[str]] | None = None
        at = cost
        for first in range(len(sorts)):
            for second in range(first + 1, len(sorts)):
                held = [one for number, one in enumerate(sorts) if number not in (first, second)]
                held.append(sorts[first] | sorts[second])
                asking = self._cost(counted, held)
                if asking < at:
                    best, at = held, asking
        if best is None:
            return None
        logger.info(
            "Joined into %r, saving %.0f bits; %d sorts left", "".join(sorted(best[-1])), cost - at, len(best)
        )
        return best, at

    def _cost(self, counted: collections.Counter, sorts: Sequence[frozenset[str]]) -> float:
        """What it takes to write the whole corpus down this way, in bits.

        **The corpus is written as its layouts.** Saying which sort each character belongs to and listing the
        shapes are what the description costs, and both shrink as sorts are joined. Saying which shape each
        notation is and which character fills each slot is what the corpus costs, and a joining that puts like
        with like leaves that about where it was: whatever the shape stops telling us, the slot now tells us
        instead.

        **Everything is charged at what it is worth and not at what it could be.** A shape common in the corpus
        costs less to name than a rare one, and a character costs its frequency within its own sort. Charged a
        flat count instead, joining two files costs a whole bit on every notation while saving only the logarithm
        of a shape count, so no two files ever join and what joins instead is whatever is rare enough to be free
        — which is how `#`, `-`, `=` and `O` once came to be one sort, having nothing in common but scarcity.

        **Company kept was tried in place of layouts and was worse.** Written as a walk from one sort to the next
        it gave chess sixteen sorts and four hundred shapes against eleven and a hundred and forty here, because
        a step tells us only which sort comes next and the ranks are several sorts while the files are being
        judged. Files merge freely once ranks are one sort and ranks once files are, and neither pays alone. The
        stall is the method's, not this coding's, and the answer to it is a larger reader behind the same call."""
        of = {character: number for number, one in enumerate(sorts) for character in one}
        naming = math.log2(len(sorts)) if len(sorts) > 1 else 0.0
        shapes: collections.Counter = collections.Counter()
        characters: collections.Counter = collections.Counter()
        for notation, seen in counted.items():
            shapes[tuple(of[one] for one in notation)] += seen
            for one in notation:
                characters[one] += seen
        belonging = len(of) * naming
        listing = sum((len(one) + 1) * naming for one in shapes)
        telling = sum(shapes.values()) * self._spread(shapes)
        filling = 0.0
        for one in sorts:
            held = collections.Counter({character: characters[character] for character in one})
            filling += sum(held.values()) * self._spread(held)
        return belonging + listing + telling + filling

    def _spread(self, counted: collections.Counter) -> float:
        """How unsettled those answers are, in bits: none where they are all the same.

        Asked of the theory rather than worked out here, because it was worked out here twice — the same three
        lines in this file and in the other learner — and a measure two services compute separately is a measure
        they can come to disagree about."""
        return self._information.entropy(counted)

    def _shapes(self, counted: collections.Counter, sorts: Sequence[Sort]) -> tuple[Shape, ...]:
        """The layouts the corpus falls into, commonest first."""
        of = {character: one.name for one in sorts for character in one.characters}
        found: collections.Counter = collections.Counter()
        for notation, seen in counted.items():
            found[Shape(tuple(of[one] for one in notation))] += seen
        return tuple(shape for shape, _ in found.most_common())
