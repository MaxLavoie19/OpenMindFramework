import logging
from collections.abc import Sequence

from openmind.inference.model.chance import Chance
from openmind.inference.model.example import Example
from openmind.rule.model.clause import Clause

logger = logging.getLogger(__name__)

#: What is assumed before anything has been counted: one case either way.
#:
#: Not a guess about the world but a refusal to be certain from nothing. Counting alone says a rule seen once and
#: held once always holds, and says it with the same face as a rule seen ten thousand times. Starting a case either
#: way makes the first few observations move the number a lot and later ones move it little, which is how
#: confidence ought to behave, and it never reaches 0 or 1 by counting — which matters, because a rule believed
#: impossible is never tried again and so never corrected.
BEFORE_HOLDING, BEFORE_NOT = 1.0, 1.0


class ChanceFitter:
    """How often a rule holds, from how often it held.

    A rule learned from six cases and a rule learned from six thousand are not the same rule, and a bare share
    cannot tell them apart: five out of six and five thousand out of six thousand are both 0.83. So what comes
    back carries its spread as well as its middle, and the spread is what says which of the two this is.

    This is estimation, not testing. Nothing here decides whether a rule is good enough to keep — no threshold to
    clear, no bar below which a rule is discarded. A rule barely better than chance can be worth a great deal, and
    what a rule is worth is settled by what it pays, not by how sure of it anyone is.

    It keeps nothing: built once, it is given the cases on every call."""

    def fit(self, clause: Clause, examples: Sequence[Example], covers: object = None) -> Chance:
        """How often that clause held, over the cases it speaks about.

        The cases it speaks about are the ones its body covers: a rule is not wrong about a case it says nothing
        about, and counting those against it would make every rule look worse the more of the world it ignored."""
        speaking = [one for one in examples if covers is None or covers(clause, one)]  # type: ignore[operator]
        return self.counted(sum(1 for one in speaking if one.holds), len(speaking))

    def counted(self, held: int, of: int) -> Chance:
        """How often something holds, from having held that many times out of that many.

        The middle is what was seen, with one case either way assumed beforehand so that nothing is certain from
        nothing. The spread narrows as the counting grows, which is the whole of what the counting buys."""
        if of < 0 or held < 0 or held > of:
            raise ValueError(f"{held} of {of} is not something that can have been counted")
        holding = held + BEFORE_HOLDING
        total = of + BEFORE_HOLDING + BEFORE_NOT
        value = holding / total
        spread = (value * (1.0 - value) / (total + 1.0)) ** 0.5
        return Chance(value, spread, of, True)

    def surer(self, one: Chance, other: Chance) -> bool:
        """Whether the first rests on enough more counting to be preferred where the two disagree."""
        return one.spread < other.spread
