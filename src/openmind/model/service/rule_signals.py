import logging
from collections.abc import Sequence

import numpy as np

from openmind.model.model.rule_candidate import RuleCandidate

logger = logging.getLogger(__name__)

#: What each of the four is called, which is what its budget is kept under. Named rather than numbered, so a
#: budget read back out of the knowledge base says what earned it.
WENT_WITH_WINNING = "went with winning"
MOVED_THE_FIT = "moved the fit"
SAYS_SOMETHING_NEW = "says something new"
FIRES_OFTEN = "fires often enough to know"


class WentWithWinning:
    """Wants terms whose readings line up with the payoff, read over the rows they fired on.

    **Over the rows it fired on, and not over all of them.** A detector reads nought nearly everywhere; scored
    against every row it would look like a term that says nothing, which is the mistake this project has a
    standing rule against. Asked only where it spoke, a mate detector that is right every time it fires rates
    as highly as a material count that is right most of the time — which is the point.

    This is the beats-chance strand, and it is a rating rather than a test on purpose. Jensen and Cohen
    measured what testing a hundred candidates on noise does: the best apparent score inflates more than
    fivefold. What bounds the search here is what a signal can afford, not a significance level."""

    @property
    def name(self) -> str:
        return WENT_WITH_WINNING

    def rates(self, candidates: Sequence[RuleCandidate]) -> Sequence[float]:
        return [self._lined_up(one) for one in candidates]

    def _lined_up(self, candidate: RuleCandidate) -> float:
        fired = candidate.fires()
        if int(fired.sum()) < 2:
            return 0.0
        readings, payoffs = candidate.readings[fired], candidate.payoffs[fired]
        if float(readings.std()) == 0.0 or float(payoffs.std()) == 0.0:
            return 0.0
        found = float(np.corrcoef(readings, payoffs)[0, 1])
        # Either direction is a term that says something; which way it points is the weight's business.
        return 0.0 if not np.isfinite(found) else abs(found)


class MovedTheFit:
    """Wants terms the fit misses most when they are taken out.

    **Leave-one-out here is arithmetic, not a hundred refits.** A fitted value is the sum of its terms'
    weighted readings, so the value without one term is the value *minus that term's weighted reading* — every
    leave-one-out score falls out of the readings the fit already took. What would have been an hour of refits
    is a subtraction.

    This is the marginal-contribution strand: not what a rule does alone, but what the set loses without it."""

    @property
    def name(self) -> str:
        return MOVED_THE_FIT

    def rates(self, candidates: Sequence[RuleCandidate]) -> Sequence[float]:
        return [self._missed(one) for one in candidates]

    def _missed(self, candidate: RuleCandidate) -> float:
        if not len(candidate.readings):
            return 0.0
        moved = candidate.weight * np.nan_to_num(candidate.readings, nan=0.0)
        found = float(np.mean(moved**2))
        return found if np.isfinite(found) else 0.0


class SaysSomethingNew:
    """Wants terms least explained by the terms the fit already leant on harder.

    **Novelty against a fixed order, so nothing depends on who bought first.** A candidate is read against the
    candidates the fit weighted more heavily than it, which is an order the fit settled before any signal
    spoke. Read instead against "what has been admitted so far", the same candidate would rate differently
    depending on which signal was asked first, and a rating that moves with the asking order is not a rating.

    The redundancy-versus-novelty tension is not resolved here and is not meant to be. A term coherent with
    what is already there is vouched for by that coherence and adds little; an innovative one may help and has
    nothing supporting it. This signal takes the second side. `WentWithWinning` takes the first. Which of them
    was right about a game is settled by what each earns, not by an argument."""

    @property
    def name(self) -> str:
        return SAYS_SOMETHING_NEW

    def rates(self, candidates: Sequence[RuleCandidate]) -> Sequence[float]:
        order = sorted(range(len(candidates)), key=lambda at: -abs(candidates[at].weight))
        found = [0.0] * len(candidates)
        for place, at in enumerate(order):
            found[at] = self._unlike(candidates[at], [candidates[before] for before in order[:place]])
        return found

    def _unlike(self, candidate: RuleCandidate, stronger: Sequence[RuleCandidate]) -> float:
        """One less the most it looks like any term the fit leant on harder. Nothing stronger means nothing
        says it yet, which is as new as a term gets."""
        mine = np.nan_to_num(candidate.readings, nan=0.0)
        if not len(mine) or float(mine.std()) == 0.0:
            return 0.0
        likest = 0.0
        for other in stronger:
            theirs = np.nan_to_num(other.readings, nan=0.0)
            if float(theirs.std()) == 0.0:
                continue
            found = float(np.corrcoef(mine, theirs)[0, 1])
            if np.isfinite(found):
                likest = max(likest, abs(found))
        return 1.0 - likest


class FiresOften:
    """Wants terms that fired on enough rows for their record to mean something.

    **This is here because both kinds of rule are wanted.** A game needs specific rules and generic ones: a
    detector that speaks once a game and decides it, and a count that speaks every position and is a little
    right each time. The three other signals all reward a term for being sharp on the rows it fired on, which
    is what a specific rule is good at; without this one, nothing in the economy asks for the generic rule that
    holds a position together.

    **It is not coverage folded into a score.** The standing rule is that a rule is never marked down for
    firing rarely, and nothing here marks anything down — this signal declines to spend, and the other three
    are free to buy the rare term with their own budget. A rule needs one voucher, not four. If preferring
    common terms turns out to be a poor way to pick rules, this signal earns less and buys less, which is the
    economy holding it to account rather than an argument settling it."""

    @property
    def name(self) -> str:
        return FIRES_OFTEN

    def rates(self, candidates: Sequence[RuleCandidate]) -> Sequence[float]:
        return [self._fired(one) for one in candidates]

    def _fired(self, candidate: RuleCandidate) -> float:
        if not len(candidate.readings):
            return 0.0
        return float(np.mean(candidate.fires()))
