from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValueMeasure:
    """How a valuer alone did on the sampled positions: in how many it valued the position, and the mean absolute
    difference there between its value for the player to act and the best action's value (None when it valued none);
    then, choosing one step ahead uniformly among the actions whose outcomes it values highest, the expected number of
    positions where the choice is optimal and the expected value the choice loses (regret), averaged over the
    positions."""

    positions: int
    valued: int
    mean_absolute_error: float | None
    optimal: float
    mean_regret: float
