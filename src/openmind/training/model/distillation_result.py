from dataclasses import dataclass

from openmind.rbs.model.rule_base import RuleBase


@dataclass(frozen=True, slots=True)
class DistillationResult:
    """The induced rule base, how many samples it came from and was measured on, the visit-weighted mean absolute
    difference between its ratings and the held-out search results (None without held-out samples to rate), and the
    mean number of conditions per rule."""

    rule_base: RuleBase
    training_samples: int
    held_out_samples: int
    rating_error: float | None
    mean_conditions: float
