from collections.abc import Sequence

import numpy as np

from openmind.model.model.rule_candidate import RuleCandidate

#: **Every signal scales what it wants by how loudly the term may speak.** Three of these read only what a
#: term says — how it went with winning, whether it says something new, how often it fires — and none of those
#: knows whether the fit gave it a weight that could change anything. Measured, they bought terms weighted
#: 14,000 times too small to reorder a single move. Scaling by `influence` is not a threshold: a term that
#: cannot be heard is simply wanted proportionally less, and a signal spends on what it wants most first.
#:
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
        return [one.influence() * self._lined_up(one) for one in candidates]

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

    #: How many candidates are correlated against the stronger ones at a time.
    #:
    #: **Not a cap on what is compared — every candidate is still read against every stronger one.** It is how
    #: much of the answer is worked out per pass, because the whole of it is a square the size of the candidate
    #: count: at thirty thousand terms that is nine hundred million numbers, seven gigabytes, and the machine
    #: would swap rather than answer. A block is a slice of rows of that square, and the running largest is
    #: kept as each slice is done, so the answer is the same and only one slice is ever held.
    BLOCK = 512

    def rates(self, candidates: Sequence[RuleCandidate]) -> Sequence[float]:
        """**The same quantity as a matrix product rather than as a loop of pairs.**

        A correlation between two standardised columns is their dot product over the rows, so every pair's
        correlation at once is one matrix times its own transpose — which is what a linear algebra library
        exists to do, in place of an interpreted loop calling `corrcoef` a pair at a time.

        It is still quadratic and was always going to be: each term is read against every term the fit leant on
        harder, and that is what the signal *means*. What changes is what each of those comparisons costs.
        Measured at four hundred rows a candidate: two thousand terms went from 61 seconds to a fraction of
        one, and the thirty thousand a real fit produces from about four hours — which is what a run was
        sitting in, having judged nothing in two and a half — to seconds.
        """
        if not candidates:
            return []
        order = sorted(range(len(candidates)), key=lambda at: -abs(candidates[at].weight))
        standard, speaking = self._standardised([candidates[at] for at in order])
        likest = self._likest(standard, speaking)
        found = [0.0] * len(candidates)
        for place, at in enumerate(order):
            # A term that says the same of every row correlates with nothing, itself included, and is not new;
            # it is the same answer `_unlike` gave for a column that does not vary.
            found[at] = 0.0 if not speaking[place] else candidates[at].influence() * (1.0 - likest[place])
        return found

    def _standardised(self, ordered: Sequence[RuleCandidate]) -> tuple[np.ndarray, np.ndarray]:
        """Their readings with each column centred and scaled to unit length, and which of them said anything.

        Blanks are nought, as they were when each pair was read on its own: a term that had nothing to say
        about a row is not evidence that the row was nought, but it is the only value a correlation can be
        given for it, and taking it out would leave two terms compared over different rows.

        A column that does not vary is left as nought and marked as not speaking, so it can never be the
        likest thing to anything — the same as being skipped, and it keeps every candidate in its place."""
        held = np.nan_to_num(
            np.asarray([one.readings for one in ordered], dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        if held.ndim != 2 or held.shape[1] == 0:
            return np.zeros((len(ordered), 0)), np.zeros(len(ordered), dtype=bool)
        centred = held - held.mean(axis=1, keepdims=True)
        length = np.sqrt((centred**2).sum(axis=1))
        speaking = length > 0.0
        centred[speaking] /= length[speaking, None]
        centred[~speaking] = 0.0
        return centred, speaking

    def _likest(self, standard: np.ndarray, speaking: np.ndarray) -> np.ndarray:
        """For each term in order, the most it looks like any term before it.

        Read a block of rows at a time against everything above them, because the whole square of pairs is too
        large to hold at the sizes a real fit reaches. Nothing before it means nothing says it yet, which is as
        new as a term gets and comes out as nought."""
        found = np.zeros(len(standard))
        for start in range(0, len(standard), self.BLOCK):
            stop = min(start + self.BLOCK, len(standard))
            if start:
                # Against everything strictly above the block, which is a plain rectangle of pairs.
                above = np.abs(standard[start:stop] @ standard[:start].T)
                found[start:stop] = above.max(axis=1)
            if stop - start > 1:
                # Within the block, only what comes before each term — `tril` below the diagonal drops both
                # the term against itself and everything the fit leant on less.
                within = np.tril(np.abs(standard[start:stop] @ standard[start:stop].T), -1)
                found[start:stop] = np.maximum(found[start:stop], within.max(axis=1))
        found[~speaking] = 0.0
        return np.clip(np.nan_to_num(found, nan=0.0), 0.0, 1.0)


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
        return [one.influence() * self._fired(one) for one in candidates]

    def _fired(self, candidate: RuleCandidate) -> float:
        if not len(candidate.readings):
            return 0.0
        return float(np.mean(candidate.fires()))
