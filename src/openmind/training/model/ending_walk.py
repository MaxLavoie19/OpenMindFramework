from dataclasses import dataclass

from openmind.inference.model.deduction import Deduction


@dataclass(frozen=True, slots=True)
class EndingWalk:
    """One decisive game walked back from its end: its index among the round's training games, and the deductions of its
    positions from the last one backward, the last of them not proven unless the walk ran out of positions or of its
    limit."""

    game: int
    deductions: tuple[Deduction, ...]

    @property
    def proven(self) -> int:
        return sum(1 for deduction in self.deductions if deduction.proven)
