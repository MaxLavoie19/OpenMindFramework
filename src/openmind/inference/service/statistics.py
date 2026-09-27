import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: What is assumed before anything has been counted: one case either way.
#:
#: Not a guess about the world but a refusal to be certain from nothing. Counting alone says a rule seen once and
#: held once always holds, and says it with the same face as a rule seen ten thousand times. Starting a case
#: either way makes the first few observations move the number a lot and later ones move it little, which is how
#: confidence ought to behave — and it never reaches nought or one by counting, which matters, because a rule
#: believed impossible is never tried again and so never corrected.
BEFORE_HOLDING, BEFORE_NOT = 1.0, 1.0


class Statistics:
    """How often a thing holds, how sure that is, and how far it spreads.

    **Here because these are asked all over and answered separately.** How often a rule holds is
    `ChanceFitter`; how often a mechanism is right is `AccuracyScorer`, with its own prior written out again;
    how far numbers spread is a third place. Three callers computing from the same shape of evidence, each with
    its own arithmetic — which is the mark of something that wants a name.

    **Estimation and not testing.** Nothing here decides whether anything is good enough to keep. There is no
    threshold to clear and no bar below which a thing is discarded: something barely better than chance can be
    worth a great deal, and what it is worth is settled by what it pays. That is a decision this project has
    already taken, and this is where it is kept.

    **What comes back carries its spread.** Five of six and five thousand of six thousand are both 0.83, and
    they are not the same claim. A middle without a spread cannot tell them apart, so nothing here returns one.

    It keeps nothing but what is assumed before counting."""

    def __init__(self, before_holding: float = BEFORE_HOLDING, before_not: float = BEFORE_NOT) -> None:
        #: What is assumed either way before a thing has been seen at all. The caller's, because how much to
        #: assume is a question about what is being counted and not about counting.
        self._before_holding = before_holding
        self._before_not = before_not

    def share(self, held: int, of: int) -> float:
        """How often it holds, with a case assumed either way so that nothing is certain from nothing."""
        if of < 0 or held < 0 or held > of:
            raise ValueError(f"{held} of {of} is not something that can have been counted")
        return (held + self._before_holding) / (of + self._before_holding + self._before_not)

    def spread(self, held: int, of: int) -> float:
        """How unsettled that share is: wide when little has been counted, narrowing as counting grows.

        This is the whole of what counting buys. It is what tells a rule seen six times from one seen six
        thousand, and both of them from one nobody has seen at all."""
        value = self.share(held, of)
        total = of + self._before_holding + self._before_not
        return (value * (1.0 - value) / (total + 1.0)) ** 0.5

    def leaning(self, held: float, of: int, toward: float, weight: float) -> float:
        """How often it holds, leaning on what was expected of it before anything was counted.

        A mechanism that says of itself how accurate it is has said something worth keeping until enough has
        been seen to say otherwise. `weight` is how many cases that claim is worth: two means the claim is
        outvoted by the third observation, which is the right order for a claim nobody has checked.

        `held` is a count for anything counted and a share for anything scored by degree: a drawn game holds
        half of one, and half a game is a real thing rather than a rounding of it."""
        if of < 0 or held < 0 or held > of:
            raise ValueError(f"{held} of {of} is not something that can have been counted")
        return (held + toward * weight) / (of + weight)

    def middle(self, values: Sequence[float]) -> float:
        """What those numbers come to on average, or nought where there are none."""
        return sum(values) / len(values) if values else 0.0

    def apart(self, values: Sequence[float]) -> float:
        """How far those numbers spread about their middle — the root of the mean square, not the sample's.

        The population's and not the sample's, because these are what was seen and not a draw from something
        larger: a mechanism's errors are its errors."""
        if not values:
            return 0.0
        held = self.middle(values)
        return (sum((one - held) ** 2 for one in values) / len(values)) ** 0.5
