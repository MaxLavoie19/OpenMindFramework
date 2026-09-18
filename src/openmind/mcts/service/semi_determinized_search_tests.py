import logging
import math
from pathlib import Path

import pytest

from openmind.mcts.model.hypothesis import Hypothesis
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.semi_determinized_search import SemiDeterminizedSearch
from openmind.mcts.service.tree_search import TreeSearch
from openmind.mcts.service.tree_search_tests import Ticking, guessing_game
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.timing.model.time_source import TimeSource
from openmind.timing.service.manual_time_source import ManualTimeSource
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")

HEADS = Action("guess", (("side", "heads"),))
TAILS = Action("guess", (("side", "tails"),))
SETTINGS = SearchSettings(40, math.sqrt(2), 1)


class Believes:
    """A theory of mind believing the coin shows heads with the given chance, whatever the domain says."""

    def __init__(self, heads: float) -> None:
        self._heads = heads

    def hypotheses(self, rbs: RuleBasedSystem, state: State, player: str) -> tuple[tuple[Hypothesis, float], ...]:
        def showing(side: str) -> Hypothesis:
            shown = State(tuple((name, side if name == "coin" else value) for name, value in state.variables))
            return Hypothesis((("coin", side),), ((shown, 1.0),))

        return (showing("heads"), self._heads), (showing("tails"), 1.0 - self._heads)

    def strategy(self, rbs: RuleBasedSystem, state: State, player: str, other: str) -> None:
        return None


def coin_game(tmp_path: Path, secret: str) -> RuleBasedSystem:
    """The guessing game: me guesses a coin's side, and what me believes about it is its theory of mind's business."""
    return guessing_game(tmp_path, secret)


def new_search(time_source: TimeSource | None = None) -> SemiDeterminizedSearch:
    return SemiDeterminizedSearch(
        TreeSearch(
            StateReader(), ActionTextMapper(), time_source=ManualTimeSource() if time_source is None else time_source
        ),
        StateReader(),
        ActionTextMapper(),
    )


def test_a_theory_believing_heads_guesses_heads_where_the_domain_gives_even_chances(tmp_path: Path) -> None:
    rbs = coin_game(tmp_path, "tails")

    result = new_search().search(rbs, rbs.start(), SETTINGS, Believes(0.8))

    expected = {item.action: item.mean_payoff for item in result.statistics}
    assert result.chosen == HEADS
    assert (expected[HEADS], expected[TAILS]) == (pytest.approx(0.8), pytest.approx(0.2))
    assert [(hypothesis.label, hypothesis.probability) for hypothesis in result.hypotheses] == [
        ((("coin", "heads"),), 0.8),
        ((("coin", "tails"),), pytest.approx(0.2)),
    ]
    assert sum(item.visits for item in result.statistics) == SETTINGS.iterations


def test_the_search_follows_the_beliefs_rather_than_what_is_really_so(tmp_path: Path) -> None:
    heads, tails = coin_game(tmp_path, "heads"), coin_game(tmp_path, "tails")

    # The coin really lies differently in the two games; the agent believes the same of both, so it plays the same.
    seen = new_search().search(heads, heads.start(), SETTINGS, Believes(0.7))
    other = new_search().search(tails, tails.start(), SETTINGS, Believes(0.7))

    assert (seen.chosen, seen.statistics) == (other.chosen, other.statistics)
    assert seen.hypotheses == other.hypotheses


class Overconfident(Believes):
    """A theory of mind giving heads 0.8 and tails 0.4."""

    def hypotheses(self, rbs: RuleBasedSystem, observed: State, player: str) -> tuple[tuple[Hypothesis, float], ...]:
        (heads, _), (tails, _) = super().hypotheses(rbs, observed, player)
        return (heads, 0.8), (tails, 0.4)


def test_hypotheses_whose_probabilities_don_t_sum_to_1_raise(tmp_path: Path) -> None:
    rbs = coin_game(tmp_path, "tails")

    with pytest.raises(ValueError, match="summing to 1.2"):
        new_search().search(rbs, rbs.start(), SETTINGS, Overconfident(0.8))


def test_a_hypothesis_with_a_negative_probability_raises(tmp_path: Path) -> None:
    rbs = coin_game(tmp_path, "tails")

    with pytest.raises(ValueError, match="negative probability"):
        new_search().search(rbs, rbs.start(), SETTINGS, Believes(1.2))


def test_time_one_hypothesis_leaves_goes_to_the_hypotheses_after_it(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    rbs = coin_game(tmp_path, "tails")
    settings = SearchSettings(4, math.sqrt(2), 1, seconds=100.0)

    result = new_search(Ticking()).search(rbs, rbs.start(), settings, Believes(0.5))

    assert "me weighs 2 hypotheses: coin='heads' at 0.5; coin='tails' at 0.5; 2, 2 iterations and 100 seconds shared as they go" in caplog.messages
    assert "me has 49.500 seconds of the 99.000 left for this hypothesis" in caplog.messages
    assert "me has 95.000 seconds of the 95.000 left for this hypothesis" in caplog.messages
    assert result.iterations == 4
    assert result.seconds == 9.0


def test_once_the_time_is_up_each_hypothesis_left_gets_one_iteration(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    rbs = coin_game(tmp_path, "tails")
    settings = SearchSettings(None, math.sqrt(2), 1, seconds=1.0)

    result = new_search(Ticking()).search(rbs, rbs.start(), settings, Believes(0.8))

    assert caplog.messages.count("me's time is up, so this hypothesis gets 1 iteration") == 2
    assert result.iterations == 2
    assert sum(item.visits for item in result.statistics) == 2


def test_the_semi_determinized_search_searches_each_hypothesis_by_puct(tmp_path: Path) -> None:
    rbs = coin_game(tmp_path, "tails")
    settings = SearchSettings(40, math.sqrt(2), 1, selection="puct")

    result = new_search().search(rbs, rbs.start(), settings, Believes(0.8))

    assert result.chosen == HEADS
    assert result.iterations == 40
