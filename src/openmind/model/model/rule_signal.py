from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from openmind.model.model.rule_candidate import RuleCandidate


@runtime_checkable
class RuleSignal(Protocol):
    """A way of proposing that a rule earns its place.

    **A signal rates; it does not bid.** What it wants is on whatever scale suits it — a correlation, a loss, a
    share of rows — and nothing here makes those comparable, because they never need to be. How much a signal
    gets is its budget. That is the point of a budget over a weight: a signal cannot talk its way to more by
    rating louder, and the leading signal cannot silence the rest.

    **A signal is not a judge.** It says what it would buy, not what is good. A signal that keeps choosing
    poorly earns less and buys less, which is the whole of how it is held to account — so a signal may prefer
    whatever it likes, including things another signal is right to ignore."""

    @property
    def name(self) -> str:
        """What this signal is called, which is what its budget is kept under and what it is paid against."""
        ...

    def rates(self, candidates: Sequence[RuleCandidate]) -> Sequence[float]:
        """How much it wants each of those candidates, in the candidates' order.

        Higher is more wanted. The scale is the signal's own and is never compared across signals; only the
        order within one signal is read, because that is the order it spends in. A rate that is not a number —
        no rows, nothing varying — is nought, which is a candidate this signal does not ask for."""
        ...
