from dataclasses import dataclass

from openmind.evaluation.model.match_results import MatchResults
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_fit import ValueFit
from openmind.training.model.pondering_summary import PonderingSummary


@dataclass(frozen=True, slots=True)
class TrainingRound:
    """One round of a value training loop: its number, the value base it fitted with every price's fit and the chosen
    one, the rows it fitted on and held out, the error on held-out rows, its games against the random policy and
    untrained MCTS, its games against the previous round's agent (None in round 1 without start rules, or without
    evaluation games), how long the round took, and what its pondering gave (None without pondering)."""

    number: int
    value_base: ValueBase
    fits: tuple[ValueFit, ...]
    chosen: ValueFit | None
    training_rows: int
    held_out_rows: int
    held_out_error: float | None
    baselines: tuple[MatchResults, ...]
    against_previous: MatchResults | None
    seconds: float
    pondering: PonderingSummary | None = None
