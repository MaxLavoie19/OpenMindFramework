from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GuidanceTest:
    """Whether guidance changed the search at a number of iterations, paired position by position between the guided
    and the unguided agent: the mean difference, guided minus unguided, in the share of root visits spent on low-value
    (non-optimal) actions and in regret, each with a two-sided Wilcoxon signed-rank p-value; and the positions where
    only the guided or only the unguided agent chose an optimal action, with an exact McNemar p-value."""

    iterations: int
    positions: int
    low_value_share_difference: float
    low_value_share_p: float
    regret_difference: float
    regret_p: float
    optimal_only_guided: int
    optimal_only_unguided: int
    optimal_choice_p: float
