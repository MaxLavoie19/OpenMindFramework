from dataclasses import dataclass

from openmind.rhetoric.model.distance_target import DistanceTarget


@dataclass(frozen=True, slots=True)
class RhetoricalGoal:
    """What a speaker wants from an exchange: target distances and problematicities. De-escalating targets small ones;
    confronting a wrongdoer targets a large distance with them and raised problematicity, possibly with small distances
    to the bystanders on the same question."""

    targets: tuple[DistanceTarget, ...]
