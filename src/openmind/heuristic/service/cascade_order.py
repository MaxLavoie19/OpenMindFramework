import logging
import math
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

#: How many standard errors a challenger must beat the incumbent by before the order changes.
#:
#: **Two, because one is not a measurement.** A margin inside the noise of the decisions it was measured on has
#: not been observed at all, and reordering on it manufactures the appearance of progress out of sampling. Two
#: standard errors is the usual line for saying a paired difference is there, and it is here rather than in a
#: caller because the alternative is every caller picking its own and none of them saying which.
CONVINCING = 2.0


class CascadeOrder:
    """Decides which rater goes first, by measuring rather than by what a rater is.

    **Being specialised earns a rater nothing.** A rule set fitted on this player's rook endings has no claim
    on rook endings: if a general rater predicts them better, the general one takes the slot. The only
    currency is how much better, on decisions kept back.

    **What better means is the caller's and not this one's.** A rater that plays well is judged on games won;
    a rater that says what somebody will do is judged on how little it was surprised by what they did. Those
    are different currencies and a thing fitted against one is not a model of the other, so this takes a way
    of scoring and never assumes one.

    **Paired, which is the guard that matters.** Raters are compared only on the decisions *all* of them
    answered. Otherwise a rater that declines whenever a position is hard wins by dodging, and a comparison
    over different sets of decisions is not a comparison at all — the scoring is the same, the decisions are
    not, and nothing in the numbers says so.

    **And a margin inside the noise leaves the order alone.** Churning on differences the decisions cannot
    resolve is how a system appears to improve while doing nothing."""

    def ordered(
        self,
        raters: Sequence[str],
        scored: Callable[[str, int], float | None],
        decisions: int,
        convincing: float = CONVINCING,
    ) -> tuple[str, ...]:
        """Those raters, best first, on the decisions every one of them answered.

        `scored` says what a rater made of one decision, by its name and which decision, and None where that
        rater declined it — which is how the pairing is found rather than assumed. Higher is better, whatever
        the currency.

        Raters nothing can be measured on keep the order they came in, after the measured ones. Not knowing
        which of two is better is a reason to leave them alone, never a reason to rank them."""
        answered = [
            at
            for at in range(decisions)
            if all(scored(one, at) is not None for one in raters)
        ]
        if not answered or len(raters) < 2:
            logger.info(
                "Leaving the order alone: %d raters, %d decisions all of them answered", len(raters), len(answered)
            )
            return tuple(raters)
        held = {one: [float(scored(one, at) or 0.0) for at in answered] for one in raters}
        best = max(raters, key=lambda one: sum(held[one]) / len(answered))
        order = [best]
        for one in raters:
            if one == best:
                continue
            margin, width = self._against(held[best], held[one])
            if margin <= convincing * width:
                logger.info(
                    "%s does not beat %s convincingly: %g over %d decisions, within %g standard errors",
                    best,
                    one,
                    margin,
                    len(answered),
                    convincing,
                )
            order.append(one)
        logger.info(
            "Ordered %d raters on %d decisions every one of them answered: %s",
            len(raters),
            len(answered),
            ", ".join(order),
        )
        return tuple(order)

    def _against(self, one: Sequence[float], other: Sequence[float]) -> tuple[float, float]:
        """How much better the first is, and how far that could be out.

        The paired difference and its standard error — paired because the same decisions were scored by both,
        and a difference taken decision by decision is measured where the decisions themselves vary and an
        unpaired one is not."""
        apart = [first - second for first, second in zip(one, other, strict=True)]
        middle = sum(apart) / len(apart)
        if len(apart) < 2:
            return middle, 0.0
        spread = sum((held - middle) ** 2 for held in apart) / (len(apart) - 1)
        return middle, math.sqrt(spread / len(apart))
