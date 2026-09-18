import logging
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.mcts.constant.mcts_constant import PUCT, UCB1
from openmind.mcts.model.chance_node import ChanceNode
from openmind.mcts.model.decision_node import DecisionNode
from openmind.mcts.model.guidance import Guidance
from openmind.mcts.model.leaf_valuation import LeafValuation
from openmind.mcts.model.move_prior import MovePrior
from openmind.mcts.model.search_result import SearchResult
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.tree_search import TreeSearch
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.timing.model.time_source import TimeSource
from openmind.timing.service.manual_time_source import ManualTimeSource
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")

type Game = RuleBasedSystem
type Outcomes = Sequence[tuple[float, PythonRule]]


class Favour:
    """A rater rating the named action 1.0 and every other action 0.0."""

    def __init__(self, name: str) -> None:
        self._name = name

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        return tuple(1.0 if action.name == self._name else 0.0 for action in actions)


class Ticking:
    """A time source moving on by `step` seconds after each reading, so a search's time runs with the readings it makes."""

    def __init__(self, step: float = 1.0) -> None:
        self._source = ManualTimeSource()
        self._step = step

    def now(self) -> float:
        now = self._source.now()
        self._source.advance(self._step)
        return now


def search(
    game: Game,
    iterations: int | None,
    exploration: float = math.sqrt(2),
    seed: int | None = 1,
    guidance: Guidance | None = None,
    valuation: LeafValuation | None = None,
    rollout_limit: int | None = None,
    unfinished_payoff: float | None = None,
    possible: tuple[tuple[State, float], ...] | None = None,
    seconds: float | None = None,
    time_source: TimeSource | None = None,
    selection: str = UCB1,
    prior: MovePrior | None = None,
) -> SearchResult:
    tree_search = TreeSearch(
        StateReader(), ActionTextMapper(), time_source=ManualTimeSource() if time_source is None else time_source
    )
    settings = SearchSettings(
        iterations, exploration, seed, rollout_limit, unfinished_payoff, seconds=seconds, selection=selection, prior=prior
    )
    return tree_search.search(game, game.start(), settings, guidance, valuation, possible)


def pay(payoff: float) -> PythonRule:
    return PythonRule(f"payoff = {payoff!r}")


def declared(
    tmp_path: Path,
    state: State,
    legal: dict[str, Sequence[PythonRule]],
    outcomes: dict[str, Outcomes],
    players: Players = Players(("me",), "turn", ("payoff",)),
    parameters: dict[str, tuple[str, PythonRule]] | None = None,
) -> Game:
    """A small game declared into a knowledge base of its own, as a project would declare one."""
    knowledge_base = create_knowledge_base("search", tmp_path)
    declarer = RuleDeclarer(knowledge_base, "search")
    declarer.starts_at(state)
    declarer.played_by(players)
    for action, parameter in (parameters or {}).items():
        declarer.values(action, parameter[0], parameter[1])
    for action, constraints in legal.items():
        declarer.constraints(action, *constraints)
    for action, branches in outcomes.items():
        for number, (chance, rule) in enumerate(branches, start=1):
            declarer.leads_to(action, rule, chance, number if len(branches) > 1 else None)
    return create_rule_based_system(knowledge_base, declarer.done())


def one_move_game(tmp_path: Path, **outcomes: Outcomes) -> Game:
    """Every action is legal until the payoff is set, and every action sets it."""
    no_payoff = PythonRule("payoff is None")
    return declared(
        tmp_path,
        State((("payoff", None), ("turn", "me"))),
        {action: (no_payoff,) for action in outcomes},
        dict(outcomes),
    )


def two_step_game(tmp_path: Path) -> Game:
    """go moves to stage 1, where win pays 1.0 and lose pays 0.0."""
    unset, first, second = PythonRule("payoff is None"), PythonRule("stage == 0"), PythonRule("stage == 1")
    return declared(
        tmp_path,
        State((("payoff", None), ("stage", 0), ("turn", "me"))),
        {"go": (unset, first), "lose": (unset, second), "win": (unset, second)},
        {
            "go": ((1.0, PythonRule("stage = 1")),),
            "lose": ((1.0, pay(0.0)),),
            "win": ((1.0, pay(1.0)),),
        },
    )


def countdown_game(tmp_path: Path) -> Game:
    """step moves from stage 0 to stage 3, one stage at a time; at stage 3, finish pays 1.0."""
    unset = PythonRule("payoff is None")
    return declared(
        tmp_path,
        State((("payoff", None), ("stage", 0), ("turn", "me"))),
        {"step": (unset, PythonRule("stage < 3")), "finish": (unset, PythonRule("stage == 3"))},
        {"step": ((1.0, PythonRule("stage = stage + 1")),), "finish": ((1.0, pay(1.0)),)},
    )


def win_or_lose(tmp_path: Path) -> Game:
    return one_move_game(tmp_path, lose=((1.0, pay(0.0)),), win=((1.0, pay(1.0)),))


def test_picks_a_winning_action_over_a_losing_one(tmp_path: Path) -> None:
    result = search(win_or_lose(tmp_path), iterations=50)

    assert result.chosen == Action("win", ())
    assert {item.action.name: item.mean_payoff for item in result.statistics} == {"lose": 0.0, "win": 1.0}
    assert sum(item.visits for item in result.statistics) == 50


def test_prefers_a_sure_payoff_to_a_coin_flip_with_a_lower_mean(tmp_path: Path) -> None:
    game = one_move_game(tmp_path, gamble=((0.5, pay(1.0)), (0.5, pay(0.0))), safe=((1.0, pay(0.8)),))

    assert search(game, iterations=300, exploration=1.0).chosen == Action("safe", ())


def test_the_same_seed_gives_the_same_result(tmp_path: Path) -> None:
    game = one_move_game(tmp_path, gamble=((0.5, pay(1.0)), (0.5, pay(0.0))), safe=((1.0, pay(0.6)),))

    assert search(game, iterations=100, seed=3) == search(game, iterations=100, seed=3)


def test_no_legal_action_with_an_unset_payoff_raises(tmp_path: Path) -> None:
    game = declared(
        tmp_path,
        State((("done", False), ("payoff", None), ("turn", "me"))),
        {"finish": (PythonRule("done == False"),)},
        {"finish": ((1.0, PythonRule("done = True")),)},
    )

    with pytest.raises(ValueError, match="payoff"):
        search(game, iterations=1)


def test_no_legal_action_at_the_root_raises(tmp_path: Path) -> None:
    over = declared(
        tmp_path,
        State((("payoff", 1.0), ("turn", "me"))),
        {"win": (PythonRule("payoff is None"),)},
        {"win": ((1.0, pay(1.0)),)},
    )

    with pytest.raises(ValueError, match="No legal action"):
        search(over, iterations=1)


def test_logs_the_search_summary(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    search(win_or_lose(tmp_path), iterations=2)

    assert [record.getMessage() for record in caplog.records if record.name == "openmind.mcts.service.tree_search"] == [
        "Searching 2 iterations for me",
        "Searched 2 iterations in 0.000 seconds for me, ucb1, tree depth 1",
        "lose(): 1 visits, mean payoff 0.0 for me",
        "win(): 1 visits, mean payoff 1.0 for me",
        "Most visited: lose()",
    ]


def test_samples_hold_every_expanded_action_with_its_visits_and_mean_payoff(tmp_path: Path) -> None:
    result = search(two_step_game(tmp_path), iterations=20)

    by_name = {sample.action.name: sample for sample in result.samples}
    assert set(by_name) == {"go", "lose", "win"}
    assert (by_name["go"].visits, by_name["go"].mean_payoff) == (20, result.statistics[0].mean_payoff)
    assert by_name["lose"].visits + by_name["win"].visits == 19
    assert (by_name["lose"].mean_payoff, by_name["win"].mean_payoff) == (0.0, 1.0)
    assert all(sample.player == 0 for sample in result.samples)


def test_guidance_expands_the_best_rated_action_first(tmp_path: Path) -> None:
    result = search(win_or_lose(tmp_path), iterations=1, guidance=Guidance(Favour("lose"), 1.0, 0.2))

    assert [(sample.action.name, sample.visits) for sample in result.samples] == [("lose", 1)]


def test_guided_rollouts_follow_the_ratings(tmp_path: Path) -> None:
    result = search(two_step_game(tmp_path), iterations=1, guidance=Guidance(Favour("win"), 1.0, 0.01))

    assert [(sample.action.name, sample.mean_payoff) for sample in result.samples] == [("go", 1.0)]


class Counting:
    """Counts how often a rater is asked to rate."""

    def __init__(self, rater: Favour) -> None:
        self._rater = rater
        self.calls = 0

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        self.calls += 1
        return self._rater.rate(state, actions)


def test_without_guided_rollouts_only_the_tree_nodes_are_rated(tmp_path: Path) -> None:
    guided, unguided = Counting(Favour("win")), Counting(Favour("win"))

    search(two_step_game(tmp_path), iterations=1, guidance=Guidance(guided, 1.0, 0.01))
    search(two_step_game(tmp_path), iterations=1, guidance=Guidance(unguided, 1.0, 0.01, guided_rollouts=False))

    # The root and the node go leads to are rated either way; the guided rollout also rates its one step.
    assert (guided.calls, unguided.calls) == (3, 2)


class Recording:
    """A valuer giving every position the same payoffs, or None, and recording the stage of each position it values."""

    def __init__(self, payoffs: tuple[float, ...] | None) -> None:
        self._payoffs = payoffs
        self.stages: list[object] = []

    def values(self, state: State) -> tuple[float, ...] | None:
        self.stages.append(dict(state.variables).get("stage"))
        return self._payoffs


def test_a_valued_position_takes_the_valuer_payoffs_instead_of_a_rollout(tmp_path: Path) -> None:
    valuer = Recording((0.25,))

    result = search(two_step_game(tmp_path), iterations=1, valuation=LeafValuation(valuer))

    assert [(sample.action.name, sample.mean_payoff) for sample in result.samples] == [("go", 0.25)]
    assert valuer.stages == [1]


def test_a_finished_game_keeps_its_payoffs(tmp_path: Path) -> None:
    valuer = Recording((0.25,))

    result = search(win_or_lose(tmp_path), iterations=2, valuation=LeafValuation(valuer))

    assert {sample.action.name: sample.mean_payoff for sample in result.samples} == {"lose": 0.0, "win": 1.0}
    assert valuer.stages == []


def test_a_valuer_knowing_nothing_leaves_the_rollout_to_play_out(tmp_path: Path) -> None:
    valuer = Recording(None)

    result = search(
        two_step_game(tmp_path), iterations=1, guidance=Guidance(Favour("win"), 1.0, 0.01), valuation=LeafValuation(valuer)
    )

    assert [(sample.action.name, sample.mean_payoff) for sample in result.samples] == [("go", 1.0)]
    assert valuer.stages == [1]


@pytest.mark.parametrize(("rollout_actions", "stages", "payoff"), [(0, [1], 0.25), (1, [2], 0.25), (5, [], 1.0)])
def test_rollout_actions_are_played_before_the_position_is_valued(tmp_path: Path, 
    rollout_actions: int, stages: list[int], payoff: float
) -> None:
    valuer = Recording((0.25,))

    result = search(countdown_game(tmp_path), iterations=1, valuation=LeafValuation(valuer, rollout_actions))

    assert valuer.stages == stages
    assert [(sample.action.name, sample.mean_payoff) for sample in result.samples] == [("step", payoff)]


@pytest.mark.parametrize(("rollout_limit", "payoff"), [(0, 0.25), (1, 0.25), (5, 1.0)])
def test_a_rollout_still_in_play_at_the_limit_gets_the_unfinished_payoff(tmp_path: Path, rollout_limit: int, payoff: float) -> None:
    result = search(countdown_game(tmp_path), iterations=1, rollout_limit=rollout_limit, unfinished_payoff=0.25)

    assert [(sample.action.name, sample.mean_payoff) for sample in result.samples] == [("step", payoff)]


def test_a_valuer_due_at_the_limit_values_the_position_first(tmp_path: Path) -> None:
    valuer = Recording((0.75,))

    result = search(
        countdown_game(tmp_path), iterations=1, valuation=LeafValuation(valuer, 1), rollout_limit=1, unfinished_payoff=0.25
    )

    assert valuer.stages == [2]
    assert [(sample.action.name, sample.mean_payoff) for sample in result.samples] == [("step", 0.75)]


def guessing_game(tmp_path: Path, secret: str) -> Game:
    """me guesses a coin's side: the right side pays 1.0, the wrong one 0.0."""
    return declared(
        tmp_path,
        State((("coin", secret), ("payoff", None), ("turn", "me"))),
        {"guess": (PythonRule("payoff is None"),)},
        {"guess": ((1.0, PythonRule("payoff = 1.0 if side == coin else 0.0")),)},
        parameters={"guess": ("side", PythonRule("('heads', 'tails')"))},
    )


def coin(tmp_path: Path, side: str, chance: float) -> tuple[State, float]:
    """One state the position could be: the coin lying that way, with that chance."""
    return State((("coin", side), ("payoff", None), ("turn", "me"))), chance


def test_the_search_weighs_the_states_the_position_could_be(tmp_path: Path) -> None:
    could_be = (coin(tmp_path, "heads", 0.9), coin(tmp_path, "tails", 0.1))

    result = search(guessing_game(tmp_path, "tails"), iterations=200, possible=could_be)

    assert result.chosen == Action("guess", (("side", "heads"),))


def test_the_search_logs_how_many_states_the_position_could_be(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    could_be = (coin(tmp_path, "heads", 0.9), coin(tmp_path, "tails", 0.1))

    search(guessing_game(tmp_path, "tails"), iterations=2, possible=could_be)

    assert "me sees 2 states that could be true" in caplog.messages


def test_a_single_state_the_position_could_be_is_the_only_one_searched(tmp_path: Path) -> None:
    result = search(guessing_game(tmp_path, "heads"), iterations=20, possible=(coin(tmp_path, "tails", 1.0),))

    assert result.chosen == Action("guess", (("side", "tails"),))


def throw(shape: str) -> Action:
    return Action("throw", (("shape", shape),))


def rock_paper_scissors_search(
    game: Game,
    iterations: int | None,
    predicted: dict[str, tuple[tuple[Action, float], ...]] | None = None,
    player: str | None = "A",
    seed: int = 1,
    seconds: float | None = None,
    time_source: TimeSource | None = None,
) -> SearchResult:
    tree_search = TreeSearch(
        StateReader(), ActionTextMapper(), time_source=ManualTimeSource() if time_source is None else time_source
    )
    settings = SearchSettings(iterations, math.sqrt(2), seed, seconds=seconds)
    return tree_search.search(game, game.start(), settings, player=player, predicted=predicted)


def test_players_acting_at_once_regret_match_toward_even_throws(game: Game) -> None:
    rps = game("rockpaperscissors")
    result = rock_paper_scissors_search(rps, 3000)

    assert [action for action, _ in result.strategy] == [throw("rock"), throw("paper"), throw("scissors")]
    assert all(abs(probability - 1 / 3) < 0.1 for _, probability in result.strategy)
    assert result.chosen in (throw("rock"), throw("paper"), throw("scissors"))
    assert sum(item.visits for item in result.statistics) == 3000
    assert {sample.player for sample in result.samples} == {0, 1}


def test_a_player_predicted_to_throw_rock_mostly_is_answered_with_paper(game: Game) -> None:
    rps = game("rockpaperscissors")
    rock_mostly = ((throw("rock"), 0.6), (throw("paper"), 0.2), (throw("scissors"), 0.2))

    result = rock_paper_scissors_search(rps, 2000, {"B": rock_mostly})

    assert dict(result.strategy)[throw("paper")] > 0.5


def test_a_search_where_players_act_at_once_logs_the_prediction_and_the_average_strategy(game: Game, 
    caplog: pytest.LogCaptureFixture,
) -> None:
    rps = game("rockpaperscissors")
    caplog.set_level(logging.INFO)

    rock_paper_scissors_search(rps, 30, {"B": ((throw("rock"), 1.0),)})

    assert "Searching 30 iterations for A, acting at once with B" in caplog.messages
    assert "B is predicted to play throw(shape='rock')=1.0 throw(shape='paper')=0.0 throw(shape='scissors')=0.0" in caplog.messages
    assert any(message.startswith("Sampled from the average strategy: throw(shape=") for message in caplog.messages)


@pytest.mark.parametrize(
    ("player", "predicted", "message"),
    [
        (None, None, "needs one of them as the searching player"),
        ("A", {"A": ((throw("rock"), 1.0),)}, "its own strategy can't be predicted"),
        ("A", {"B": ((throw("rock"), 0.5),)}, "summing to 1"),
        ("A", {"C": ((throw("rock"), 1.0),)}, "isn't a player to act"),
    ],
)
def test_a_search_where_players_act_at_once_rejects_a_missing_player_or_a_wrong_prediction(game: Game, 
    player: str | None, predicted: dict[str, tuple[tuple[Action, float], ...]] | None, message: str
) -> None:
    rps = game("rockpaperscissors")
    with pytest.raises(ValueError, match=message):
        rock_paper_scissors_search(rps, 10, predicted, player)


@pytest.mark.parametrize(
    ("rollout_limit", "unfinished_payoff", "message"), [(-1, 0.5, "negative"), (3, None, "unfinished payoff")]
)
def test_a_rollout_limit_is_at_least_0_and_comes_with_an_unfinished_payoff(tmp_path: Path, 
    rollout_limit: int, unfinished_payoff: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        search(win_or_lose(tmp_path), iterations=1, rollout_limit=rollout_limit, unfinished_payoff=unfinished_payoff)


@pytest.mark.parametrize(
    ("iterations", "seconds", "done"),
    [(None, 5.0, 5), (50, 5.0, 5), (3, 100.0, 3), (None, 0.5, 1), (50, 0.5, 1)],
    ids=["seconds alone", "seconds first", "iterations first", "time up after one", "time up before the iterations"],
)
def test_a_search_stops_at_its_seconds_or_its_iterations_whichever_comes_first(tmp_path: Path, 
    iterations: int | None, seconds: float, done: int
) -> None:
    result = search(win_or_lose(tmp_path), iterations, seconds=seconds, time_source=Ticking())

    assert result.iterations == done
    assert sum(item.visits for item in result.statistics) == done


@pytest.mark.parametrize(
    ("iterations", "seconds", "done"),
    [(None, 5.0, 5), (50, 5.0, 5), (3, 100.0, 3), (None, 0.5, 1), (50, 0.5, 1)],
    ids=["seconds alone", "seconds first", "iterations first", "time up after one", "time up before the iterations"],
)
def test_a_search_where_players_act_at_once_stops_at_its_seconds_or_its_iterations_whichever_comes_first(game: Game, 
    iterations: int | None, seconds: float, done: int
) -> None:
    rps = game("rockpaperscissors")
    result = rock_paper_scissors_search(rps, iterations, seconds=seconds, time_source=Ticking())

    assert result.iterations == done
    assert sum(item.visits for item in result.statistics) == done


def test_a_search_on_time_logs_its_limits_and_what_it_did_in_them(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    result = search(win_or_lose(tmp_path), 3, seconds=100.0, time_source=Ticking(0.25))
    search(win_or_lose(tmp_path), None, seconds=0.5, time_source=Ticking(0.25))

    assert result.seconds == 0.75
    assert "Searching up to 3 iterations or 100 seconds for me" in caplog.messages
    assert "Searched 3 iterations in 0.750 seconds for me, ucb1, tree depth 1" in caplog.messages
    assert "Searching for 0.5 seconds for me" in caplog.messages
    assert "Searched 2 iterations in 0.750 seconds for me, ucb1, tree depth 1" in caplog.messages


@pytest.mark.parametrize(
    ("iterations", "seconds", "message"),
    [(None, None, "iterations, seconds or both"), (0, None, "at least 1 iteration"), (None, 0.0, "more than 0 seconds")],
)
def test_a_search_needs_at_least_one_iteration_or_some_seconds(tmp_path: Path, 
    iterations: int | None, seconds: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        search(win_or_lose(tmp_path), iterations, seconds=seconds)


class Favouring:
    """A prior giving the named action `share` and splitting the rest evenly between the others."""

    name = "favouring"

    def __init__(self, action: str, share: float) -> None:
        self._action, self._share = action, share

    def priors(self, state: State, actions: tuple[Action, ...]) -> tuple[float, ...]:
        rest = (1.0 - self._share) / (len(actions) - 1)
        return tuple(self._share if action.name == self._action else rest for action in actions)


def three_even_moves(tmp_path) -> Game:
    return one_move_game(tmp_path, **{name: ((1.0, pay(0.5)),) for name in ("a", "b", "c")})


def test_under_puct_moves_the_prior_dislikes_stay_unvisited_until_the_visits_outweigh_their_prior(tmp_path: Path) -> None:
    few = search(three_even_moves(tmp_path), 20, 1.5, selection=PUCT, prior=Favouring("a", 0.98))
    many = search(three_even_moves(tmp_path), 600, 1.5, selection=PUCT, prior=Favouring("a", 0.98))

    assert {item.action.name: item.visits for item in few.statistics} == {"a": 20, "b": 0, "c": 0}
    assert all(item.visits > 0 for item in many.statistics)


def test_under_puct_an_unvisited_move_is_valued_at_the_node_s_mean_payoff_so_far(tmp_path: Path) -> None:
    a, b = Action("a", ()), Action("b", ())
    tried = ChanceNode(a, (), {}, 4, [3.2])
    node = DecisionNode(State((("turn", "me"),)), (a, b), [], {a: tried}, 4, 0, (), (0.4, 0.6))
    tree_search = TreeSearch(StateReader(), ActionTextMapper())

    # a's mean is 0.8, and so is the node's: b, not yet visited, ties with a at 0.8 and wins on its higher prior
    assert tree_search._puct(node, (a, b), 0.0) == b


def wide_countdown(tmp_path: Path, stages: int = 6, dead_ends: int = 9) -> Game:
    """go steps toward a win in `stages`; every other move stops the game at 0.5, so only depth pays."""
    unset = PythonRule("payoff is None")
    stops = [f"stop{number}" for number in range(dead_ends)]
    return declared(
        tmp_path,
        State((("payoff", None), ("stage", 0), ("turn", "me"))),
        {"go": (unset,), **{stop: (unset,) for stop in stops}},
        {
            "go": ((1.0, PythonRule(f"stage = stage + 1\nif stage >= {stages}:\n    payoff = 1.0")),),
            **{stop: ((1.0, pay(0.5)),) for stop in stops},
        },
    )


def test_puct_with_a_sound_prior_builds_a_deeper_tree_than_ucb1_in_the_same_iterations(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    ucb1 = search(wide_countdown(tmp_path), 40)
    puct = search(wide_countdown(tmp_path), 40, 1.5, selection=PUCT, prior=Favouring("go", 0.9))

    assert puct.depth > ucb1.depth
    assert any(message.endswith(f"puct with the favouring prior, tree depth {puct.depth}") for message in caplog.messages)


@pytest.mark.parametrize(("selection", "exploration", "message"), [("greedy", 1.5, "ucb1 or puct"), (PUCT, -1.0, "negative")])
def test_a_search_selects_by_ucb1_or_puct_with_an_exploration_of_0_or_more(tmp_path: Path, selection: str, exploration: float, message: str) -> None:
    tree_search = TreeSearch(StateReader(), ActionTextMapper())
    game = win_or_lose(tmp_path)
    settings = SearchSettings(10, 1.4, 1, selection=selection, puct_exploration=exploration)

    with pytest.raises(ValueError, match=message):
        tree_search.search(game, game.start(), settings)
