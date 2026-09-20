from dataclasses import dataclass

from openmind.timing.model.clock import Clock


@dataclass(frozen=True, slots=True)
class Budget:
    """What a step may spend: the seconds it has, and the clock those seconds were carved from where the work runs on
    one. A step with no clock spends what it is given and answers to nothing else."""

    seconds: float
    of_clock: Clock | None = None

    def carve(self, seconds: float) -> tuple["Budget", "Budget"]:
        """A budget of those seconds and what is left of this one: what a parent hands a child out of its own time.
        Carving more than there is gives the child what there is and leaves nothing."""
        taken = min(max(seconds, 0.0), max(self.seconds, 0.0))
        return Budget(taken, self.of_clock), Budget(max(self.seconds, 0.0) - taken, self.of_clock)

    def alongside(self, seconds: float) -> "Budget":
        """A budget of its own on the same clock, taking nothing from this one: what a parent hands a child it lets run
        beside it. Seconds aren't all that is shared — a child running alongside also takes a core the parent could
        have used — and nothing here weighs cores."""
        return Budget(max(seconds, 0.0), self.of_clock)
