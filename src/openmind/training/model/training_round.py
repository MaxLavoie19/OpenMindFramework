from dataclasses import dataclass

from openmind.evaluation.model.match_results import MatchResults
from openmind.doxastic.model.rule_record import RuleRecord
from openmind.rbs.model.value_fit import ValueFit


@dataclass(frozen=True, slots=True)
class TrainingRound:
    """One round of a value training loop: its number, the context its position rules were declared under and the
    rules themselves, with every price's fit and the chosen one, the rows it fitted on and held out, the error on held-out rows, its games against the random policy and
    untrained MCTS, its games against the previous round's agent (None in round 1 without start rules, or without
    evaluation games), and how long the round took; `records` holds
    what the game records of the round's games, in the order played."""

    number: int
    context: str
    rules: tuple[RuleRecord, ...]
    fits: tuple[ValueFit, ...]
    chosen: ValueFit | None
    training_rows: int
    held_out_rows: int
    held_out_error: float | None
    baselines: tuple[MatchResults, ...]
    against_previous: MatchResults | None
    seconds: float
    records: tuple[str, ...] = ()
