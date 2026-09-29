from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuleCost:
    """What a rule has cost to read, as a distribution rather than a number.

    **A mean is not enough to answer the question that matters.** What a caller needs to know is "will this
    rule fit in the time I have left", and that is a question about the dear readings, not the average one. A
    rule that walks the threats in a position is cheap where nothing is threatened and dear where everything
    is — measured on chess, one look-ahead term averaged 426 ms where every other rule in the same ruleset
    averaged 0.2 ms, and an average hides how much of that varies with the position.

    **Mean and squares, gathered the way the search already gathers node costs.** `MonteCarloTreeSearch`
    prices its next node at the mean plus so many spreads of what the nodes before it cost, on a Welford count
    that needs nothing kept but two numbers. This is the same reading, per rule instead of per node.

    `read` is how many times it has been timed, and it is what tells a measurement from a guess: one reading
    gives a mean and no spread at all.

    **Two prices, because two different questions are asked of it.** `priced` is what to assume when deciding
    whether a rule fits the time left, and it is pessimistic — the mean plus spreads, since what matters is
    that the dear reading fits. `hoped` is what to assume when deciding what to try, and it is optimistic —
    the mean less a standard error, so a rule that has been read rarely is given the benefit of the doubt.
    Pulling opposite ways is not a contradiction: one asks what might go wrong and the other what might have
    been missed."""

    seconds: float = 0.0
    #: The running sum of squared deviations, which is what Welford carries; the spread is derived from it.
    squares: float = 0.0
    read: int = 0

    @property
    def spread(self) -> float:
        """How much what it costs varies, or nothing where it has been read fewer than twice."""
        return (self.squares / (self.read - 1)) ** 0.5 if self.read > 1 else 0.0

    def hoped(self) -> float:
        """What it might cost if the readings so far were unlucky: the mean less a standard error of it.

        **Greedy starves what it declines.** A rule priced high is never taken, so it is never timed again,
        so a first reading that happened to be dear condemns it for ever. This is the exploration that
        answers that, and it is UCB's dual: where a bandit is optimistic about a reward it wants large, this
        is optimistic about a cost it wants small, by the same amount and for the same reason.

        **The discount is the uncertainty itself and not a number anybody chose.** A standard error is the
        spread over the root of the count, so it shrinks as readings accumulate and vanishes on a rule that
        has been read often — which is exactly when its price should be believed. Nothing is invented; the
        evidence says how much it doubts itself."""
        if self.read < 2:
            return 0.0
        return max(0.0, self.seconds - self.spread / self.read**0.5)

    def priced(self, caution: float = 1.0) -> float:
        """What reading it once more is assumed to cost: the mean, plus so many spreads.

        **A rule nobody has timed is priced at nothing**, so that it is tried and measured rather than passed
        over for ever on the strength of never having been read. That is the exploration this needs, and the
        reason the count is kept rather than only the mean."""
        if not self.read:
            return 0.0
        return self.seconds + caution * self.spread

    def with_reading(self, seconds: float) -> "RuleCost":
        """This cost with one more timing folded in, by Welford, so nothing has to keep what it has seen."""
        read = self.read + 1
        moved = seconds - self.seconds
        mean = self.seconds + moved / read
        return RuleCost(mean, self.squares + moved * (seconds - mean), read)
