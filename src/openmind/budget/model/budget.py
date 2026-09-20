from dataclasses import dataclass

from openmind.timing.model.clock import Clock


@dataclass(frozen=True, slots=True)
class Budget:
    """What a step may spend: the seconds it has, and the clock those seconds were carved from where the work runs on
    one. A step with no clock spends what it is given and answers to nothing else."""

    seconds: float
    of_clock: Clock | None = None
