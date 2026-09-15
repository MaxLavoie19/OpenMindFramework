from dataclasses import dataclass

from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.training.model.value_distillation_settings import ValueDistillationSettings


@dataclass(frozen=True, slots=True)
class ValueTrainingSettings:
    """How a value training loop runs: how many rounds; each round's self-play and fitting, round k seeding with the
    distillation's seed plus k; how many rollout actions an agent plays before valuing a position with value rules; the
    rollout limit and unfinished payoff every agent searches with (None plays rollouts to the end); the games against
    each opponent after every round (0 plays none); the file the start rules came from (None without start rules); and
    the budget of the deduction an agent falls back on when its rules have no clue (None: agents never deduce)."""

    rounds: int
    distillation: ValueDistillationSettings
    rollout_actions: int
    rollout_limit: int | None
    unfinished_payoff: float | None
    evaluation_games: int
    start_file: str | None
    deduction: DeductionBudget | None = None
