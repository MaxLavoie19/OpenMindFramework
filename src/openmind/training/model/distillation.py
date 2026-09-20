from dataclasses import dataclass

from openmind.knowledge.model.rule_record import RuleRecord


@dataclass(frozen=True, slots=True)
class Distillation:
    """What came of learning a heuristic from games played: the rules it settled on, and what they rest on.

    `games` is how many were played and `decisive` how many of them anyone came out ahead in — a game everyone was
    paid the same says nothing about how to play better, so a distillation of draws alone rests on nothing.
    `error` is the loss of the chosen fit on positions it was not fitted on, None where nothing was held out."""

    context: str
    rules: tuple[RuleRecord, ...]
    games: int
    decisive: int
    training_rows: int
    held_out_rows: int
    seconds: float
    error: float | None = None
