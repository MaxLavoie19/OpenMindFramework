from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Worth:
    """What each thing on the board is worth, and what the reasoning rests on.

    `holdings` is what having one of those is worth to whoever owns it, as how much of what its owner can do goes
    away when it is taken off the board. `ended` is how many positions that were over the reasoning saw, and
    `settled` whether those positions bore out that being able to do less is being worse off — which is what makes
    the rest of it mean anything. Where they did not, the worths are still counted and said to rest on nothing."""

    holdings: tuple[tuple[str, Value, float], ...] = ()
    ended: int = 0
    settled: bool = False
    seconds: float = 0.0

    def of(self, model: str, value: Value) -> float:
        for held, one, worth in self.holdings:
            if held == model and one == value:
                return worth
        return 0.0
