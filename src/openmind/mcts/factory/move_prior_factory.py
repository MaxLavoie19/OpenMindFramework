from openmind.mcts.constant.mcts_constant import PRIORS, RATER_PRIOR, UNIFORM_PRIOR, VALUE_PRIOR
from openmind.mcts.model.action_rater import ActionRater
from openmind.mcts.model.move_prior import MovePrior
from openmind.mcts.model.position_valuer import PositionValuer
from openmind.mcts.service.rater_prior import RaterPrior
from openmind.mcts.service.uniform_prior import UniformPrior
from openmind.mcts.service.valuation_prior import ValuationPrior
from openmind.rbs.service.rule_based_game import RuleBasedGame


def create_move_prior(
    kind: str,
    temperature: float,
    rbs: RuleBasedGame,
    rater: ActionRater | None = None,
    valuer: PositionValuer | None = None,
) -> MovePrior:
    """The prior a name stands for: `uniform`; `rater`, the agent's rater's ratings; `value`, its valuer's values. A name
    it doesn't know, or a prior without the model it reads, raises ValueError."""
    if kind == UNIFORM_PRIOR:
        return UniformPrior()
    if kind == RATER_PRIOR:
        if rater is None:
            raise ValueError("The rater prior needs rules that rate actions")
        return RaterPrior(rater, temperature)
    if kind == VALUE_PRIOR:
        if valuer is None:
            raise ValueError("The value prior needs value rules")
        return ValuationPrior(valuer, rbs, rbs.players(), temperature)
    raise ValueError(f"A prior is one of {', '.join(PRIORS)}, not {kind!r}")
