from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RaterAgreement:
    """How a rater alone did on the sampled positions, picking uniformly among its top-rated actions: in how many
    positions its ratings separate the actions at all, the expected number of positions where the pick is optimal, and
    the expected exact value the pick loses (regret), averaged over the positions."""

    positions: int
    distinguishing: int
    optimal: float
    mean_regret: float
