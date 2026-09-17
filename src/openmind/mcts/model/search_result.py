from dataclasses import dataclass, field

from openmind.mcts.constant.mcts_constant import SEARCH_OPTION
from openmind.mcts.model.action_sample import ActionSample
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.mcts.model.hypothesis_result import HypothesisResult
from openmind.world.model.action import Action


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The player acting at the root, every root action's statistics, the action chosen (the most visited, in a
    semi-determinized search the highest expected payoff, or where players act at once one sampled from the average
    strategy), every tree sample, a semi-determinized search's hypotheses, where players act at once the searching
    player's average strategy over its root actions, and the iterations the search completed in the seconds it took
    (a semi-determinized search's totals over its hypotheses), and on a clock the budget the step was given. `depth` is
    the most actions from the root its tree reached (a semi-determinized search's deepest over its hypotheses), and
    `option`, how the move was chosen (see `mcts_constant`), which like the seconds doesn't take part in comparing. The
    seconds and the budget measure a run and don't take part in comparing results, so the same seed gives equal
    results."""

    player: str
    statistics: tuple[ActionStatistics, ...]
    chosen: Action
    samples: tuple[ActionSample, ...]
    hypotheses: tuple[HypothesisResult, ...] = ()
    strategy: tuple[tuple[Action, float], ...] = ()
    iterations: int = 0
    seconds: float = field(default=0.0, compare=False)
    budget: float | None = field(default=None, compare=False)
    depth: int = 0
    option: str = field(default=SEARCH_OPTION, compare=False)
