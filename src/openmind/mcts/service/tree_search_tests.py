import logging
import math

import pytest

from openmind.agent.factory.rock_paper_scissors_factory import create_rock_paper_scissors_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.mcts.model.guidance import Guidance
from openmind.mcts.model.leaf_valuation import LeafValuation
from openmind.mcts.model.search_result import SearchResult
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.tree_search import TreeSearch
from openmind.observation.model.observation import Observation
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule
from openmind.timing.model.time_source import TimeSource
from openmind.timing.service.manual_time_source import ManualTimeSource
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")

type Game = tuple[Problem, TransitionModel, State]


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
    observation: Observation | None = None,
    completions: tuple[tuple[State, float], ...] | None = None,
    seconds: float | None = None,
    time_source: TimeSource | None = None,
) -> SearchResult:
    tree_search = TreeSearch(
        create_solver(), create_predictor(), StateReader(), ActionTextMapper(), time_source=ManualTimeSource() if time_source is None else time_source
    )
    problem, transitions, state = game
    players = Players(("me",), "turn", ("payoff",))
    settings = SearchSettings(iterations, exploration, seed, rollout_limit, unfinished_payoff, seconds=seconds)
    return tree_search.search(problem, transitions, players, state, settings, guidance, valuation, observation, completions)


def pay(payoff: float) -> PythonRule:
    return PythonRule(f"payoff = {payoff!r}")


def one_move_game(*transitions: Transition) -> Game:
    """Every action is legal until the payoff is set, and every action sets it."""
    no_payoff = PythonRule("payoff is None")
    problem = Problem(tuple(ActionDefinition(transition.action, (), (no_payoff,)) for transition in transitions))
    return problem, TransitionModel(transitions), State((("payoff", None), ("turn", "me")))


def two_step_game() -> Game:
    """go moves to stage 1, where win pays 1.0 and lose pays 0.0."""
    unset = PythonRule("payoff is None")
    problem = Problem(
        (
            ActionDefinition("go", (), (unset, PythonRule("stage == 0"))),
            ActionDefinition("lose", (), (unset, PythonRule("stage == 1"))),
            ActionDefinition("win", (), (unset, PythonRule("stage == 1"))),
        )
    )
    transitions = TransitionModel(
        (
            Transition("go", (Branch(1.0, PythonRule("stage = 1")),)),
            Transition("lose", (Branch(1.0, pay(0.0)),)),
            Transition("win", (Branch(1.0, pay(1.0)),)),
        )
    )
    return problem, transitions, State((("payoff", None), ("stage", 0), ("turn", "me")))


def countdown_game() -> Game:
    """step moves from stage 0 to stage 3, one stage at a time; at stage 3, finish pays 1.0."""
    unset = PythonRule("payoff is None")
    problem = Problem(
        (
            ActionDefinition("step", (), (unset, PythonRule("stage < 3"))),
            ActionDefinition("finish", (), (unset, PythonRule("stage == 3"))),
        )
    )
    transitions = TransitionModel(
        (
            Transition("step", (Branch(1.0, PythonRule("stage = stage + 1")),)),
            Transition("finish", (Branch(1.0, pay(1.0)),)),
        )
    )
    return problem, transitions, State((("payoff", None), ("stage", 0), ("turn", "me")))


def win_or_lose() -> Game:
    return one_move_game(Transition("lose", (Branch(1.0, pay(0.0)),)), Transition("win", (Branch(1.0, pay(1.0)),)))


def test_picks_a_winning_action_over_a_losing_one() -> None:
    result = search(win_or_lose(), iterations=50)

    assert result.chosen == Action("win", ())
    assert {item.action.name: item.mean_payoff for item in result.statistics} == {"lose": 0.0, "win": 1.0}
    assert sum(item.visits for item in result.statistics) == 50


def test_prefers_a_sure_payoff_to_a_coin_flip_with_a_lower_mean() -> None:
    game = one_move_game(
        Transition("gamble", (Branch(0.5, pay(1.0)), Branch(0.5, pay(0.0)))),
        Transition("safe", (Branch(1.0, pay(0.8)),)),
    )

    assert search(game, iterations=300, exploration=1.0).chosen == Action("safe", ())


def test_the_same_seed_gives_the_same_result() -> None:
    game = one_move_game(
        Transition("gamble", (Branch(0.5, pay(1.0)), Branch(0.5, pay(0.0)))),
        Transition("safe", (Branch(1.0, pay(0.6)),)),
    )

    assert search(game, iterations=100, seed=3) == search(game, iterations=100, seed=3)


def test_no_legal_action_with_an_unset_payoff_raises() -> None:
    problem = Problem((ActionDefinition("finish", (), (PythonRule("done == False"),)),))
    finish = Transition("finish", (Branch(1.0, PythonRule("done = True")),))
    game = (problem, TransitionModel((finish,)), State((("done", False), ("payoff", None), ("turn", "me"))))

    with pytest.raises(ValueError, match="payoff"):
        search(game, iterations=1)


def test_no_legal_action_at_the_root_raises() -> None:
    problem, transitions, _ = one_move_game(Transition("win", (Branch(1.0, pay(1.0)),)))

    with pytest.raises(ValueError, match="No legal action"):
        search((problem, transitions, State((("payoff", 1.0), ("turn", "me")))), iterations=1)


def test_logs_the_search_summary(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    search(win_or_lose(), iterations=2)

    assert [record.getMessage() for record in caplog.records if record.name == "openmind.mcts.service.tree_search"] == [
        "Searching 2 iterations for me",
        "Searched 2 iterations in 0.000 seconds for me",
        "lose(): 1 visits, mean payoff 0.0 for me",
        "win(): 1 visits, mean payoff 1.0 for me",
        "Most visited: lose()",
    ]


def test_samples_hold_every_expanded_action_with_its_visits_and_mean_payoff() -> None:
    result = search(two_step_game(), iterations=20)

    by_name = {sample.action.name: sample for sample in result.samples}
    assert set(by_name) == {"go", "lose", "win"}
    assert (by_name["go"].visits, by_name["go"].mean_payoff) == (20, result.statistics[0].mean_payoff)
    assert by_name["lose"].visits + by_name["win"].visits == 19
    assert (by_name["lose"].mean_payoff, by_name["win"].mean_payoff) == (0.0, 1.0)
    assert all(sample.player == 0 for sample in result.samples)


def test_guidance_expands_the_best_rated_action_first() -> None:
    result = search(win_or_lose(), iterations=1, guidance=Guidance(Favour("lose"), 1.0, 0.2))

    assert [(sample.action.name, sample.visits) for sample in result.samples] == [("lose", 1)]


def test_guided_rollouts_follow_the_ratings() -> None:
    result = search(two_step_game(), iterations=1, guidance=Guidance(Favour("win"), 1.0, 0.01))

    assert [(sample.action.name, sample.mean_payoff) for sample in result.samples] == [("go", 1.0)]


class Counting:
    """Counts how often a rater is asked to rate."""

    def __init__(self, rater: Favour) -> None:
        self._rater = rater
        self.calls = 0

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        self.calls += 1
        return self._rater.rate(state, actions)


def test_without_guided_rollouts_only_the_tree_nodes_are_rated() -> None:
    guided, unguided = Counting(Favour("win")), Counting(Favour("win"))

    search(two_step_game(), iterations=1, guidance=Guidance(guided, 1.0, 0.01))
    search(two_step_game(), iterations=1, guidance=Guidance(unguided, 1.0, 0.01, guided_rollouts=False))

    # The root and the node go leads to are rated either way; the guided rollout also rates its one step.
    assert (guided.calls, unguided.calls) == (3, 2)


class Recording:
    """A valuer giving every position the same payoffs, or None, and recording the stage of each position it values."""

    def __init__(self, payoffs: tuple[float, ...] | None) -> None:
        self._payoffs = payoffs
        self.stages: list[object] = []

    def value(self, state: State) -> tuple[float, ...] | None:
        self.stages.append(dict(state.variables).get("stage"))
        return self._payoffs


def test_a_valued_position_takes_the_valuer_payoffs_instead_of_a_rollout() -> None:
    valuer = Recording((0.25,))

    result = search(two_step_game(), iterations=1, valuation=LeafValuation(valuer))

    assert [(sample.action.name, sample.mean_payoff) for sample in result.samples] == [("go", 0.25)]
    assert valuer.stages == [1]


def test_a_finished_game_keeps_its_payoffs() -> None:
    valuer = Recording((0.25,))

    result = search(win_or_lose(), iterations=2, valuation=LeafValuation(valuer))

    assert {sample.action.name: sample.mean_payoff for sample in result.samples} == {"lose": 0.0, "win": 1.0}
    assert valuer.stages == []


def test_a_valuer_knowing_nothing_leaves_the_rollout_to_play_out() -> None:
    valuer = Recording(None)

    result = search(
        two_step_game(), iterations=1, guidance=Guidance(Favour("win"), 1.0, 0.01), valuation=LeafValuation(valuer)
    )

    assert [(sample.action.name, sample.mean_payoff) for sample in result.samples] == [("go", 1.0)]
    assert valuer.stages == [1]


@pytest.mark.parametrize(("rollout_actions", "stages", "payoff"), [(0, [1], 0.25), (1, [2], 0.25), (5, [], 1.0)])
def test_rollout_actions_are_played_before_the_position_is_valued(
    rollout_actions: int, stages: list[int], payoff: float
) -> None:
    valuer = Recording((0.25,))

    result = search(countdown_game(), iterations=1, valuation=LeafValuation(valuer, rollout_actions))

    assert valuer.stages == stages
    assert [(sample.action.name, sample.mean_payoff) for sample in result.samples] == [("step", payoff)]


@pytest.mark.parametrize(("rollout_limit", "payoff"), [(0, 0.25), (1, 0.25), (5, 1.0)])
def test_a_rollout_still_in_play_at_the_limit_gets_the_unfinished_payoff(rollout_limit: int, payoff: float) -> None:
    result = search(countdown_game(), iterations=1, rollout_limit=rollout_limit, unfinished_payoff=0.25)

    assert [(sample.action.name, sample.mean_payoff) for sample in result.samples] == [("step", payoff)]


def test_a_valuer_due_at_the_limit_values_the_position_first() -> None:
    valuer = Recording((0.75,))

    result = search(
        countdown_game(), iterations=1, valuation=LeafValuation(valuer, 1), rollout_limit=1, unfinished_payoff=0.25
    )

    assert valuer.stages == [2]
    assert [(sample.action.name, sample.mean_payoff) for sample in result.samples] == [("step", 0.75)]


def guessing_game(secret: str) -> Game:
    """me guesses a coin's side: the right side pays 1.0, the wrong one 0.0."""
    problem = Problem(
        (ActionDefinition("guess", (Variable("side", DiscreteDomain(("heads", "tails"))),), (PythonRule("payoff is None"),)),)
    )
    guess = Transition("guess", (Branch(1.0, PythonRule("payoff = 1.0 if side == coin else 0.0")),))
    return problem, TransitionModel((guess,)), State((("coin", secret), ("payoff", None), ("turn", "me")))


def hidden_coin(heads: float) -> Observation:
    """me can't see the coin, which is heads with the given chance."""
    return Observation(
        PythonRule("('coin',)"), PythonRule(f"[({{'coin': 'heads'}}, {heads!r}), ({{'coin': 'tails'}}, {1.0 - heads!r})]")
    )


def test_with_an_observation_the_search_is_the_same_whatever_the_hidden_value() -> None:
    heads = search(guessing_game("heads"), iterations=60, observation=hidden_coin(0.5))

    assert heads == search(guessing_game("tails"), iterations=60, observation=hidden_coin(0.5))
    assert dict(heads.samples[0].state.variables)["coin"] == "<hidden>"


def test_with_an_observation_the_search_weighs_the_states_that_could_be_true() -> None:
    result = search(guessing_game("tails"), iterations=200, observation=hidden_coin(0.9))

    assert result.chosen == Action("guess", (("side", "heads"),))


def test_with_an_observation_the_search_logs_how_many_states_could_be_true(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    search(guessing_game("tails"), iterations=2, observation=hidden_coin(0.9))

    assert "me sees 2 states that could be true" in caplog.messages


def test_given_completions_are_the_only_states_that_could_be_true(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    tails = (State((("coin", "tails"), ("payoff", None), ("turn", "me"))), 1.0)

    result = search(guessing_game("heads"), iterations=20, observation=hidden_coin(0.9), completions=(tails,))

    assert result.chosen == Action("guess", (("side", "tails"),))
    assert "me sees 1 states that could be true" in caplog.messages


def test_completions_without_an_observation_raise() -> None:
    tails = (State((("coin", "tails"), ("payoff", None), ("turn", "me"))), 1.0)

    with pytest.raises(ValueError, match="Completions need an observation"):
        search(guessing_game("heads"), iterations=20, completions=(tails,))


def throw(shape: str) -> Action:
    return Action("throw", (("shape", shape),))


def rock_paper_scissors_search(
    iterations: int | None,
    predicted: dict[str, tuple[tuple[Action, float], ...]] | None = None,
    player: str | None = "A",
    seed: int = 1,
    seconds: float | None = None,
    time_source: TimeSource | None = None,
) -> SearchResult:
    domain = create_rock_paper_scissors_domain()
    tree_search = TreeSearch(
        create_solver(), create_predictor(), StateReader(), ActionTextMapper(), time_source=ManualTimeSource() if time_source is None else time_source
    )
    settings = SearchSettings(iterations, math.sqrt(2), seed, seconds=seconds)
    return tree_search.search(
        domain.problem, domain.transitions, domain.players, domain.initial_state, settings, player=player, predicted=predicted
    )


def test_players_acting_at_once_regret_match_toward_even_throws() -> None:
    result = rock_paper_scissors_search(3000)

    assert [action for action, _ in result.strategy] == [throw("rock"), throw("paper"), throw("scissors")]
    assert all(abs(probability - 1 / 3) < 0.1 for _, probability in result.strategy)
    assert result.chosen in (throw("rock"), throw("paper"), throw("scissors"))
    assert sum(item.visits for item in result.statistics) == 3000
    assert {sample.player for sample in result.samples} == {0, 1}


def test_a_player_predicted_to_throw_rock_mostly_is_answered_with_paper() -> None:
    rock_mostly = ((throw("rock"), 0.6), (throw("paper"), 0.2), (throw("scissors"), 0.2))

    result = rock_paper_scissors_search(2000, {"B": rock_mostly})

    assert dict(result.strategy)[throw("paper")] > 0.5


def test_a_search_where_players_act_at_once_logs_the_prediction_and_the_average_strategy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    rock_paper_scissors_search(30, {"B": ((throw("rock"), 1.0),)})

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
def test_a_search_where_players_act_at_once_rejects_a_missing_player_or_a_wrong_prediction(
    player: str | None, predicted: dict[str, tuple[tuple[Action, float], ...]] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        rock_paper_scissors_search(10, predicted, player)


@pytest.mark.parametrize(
    ("rollout_limit", "unfinished_payoff", "message"), [(-1, 0.5, "negative"), (3, None, "unfinished payoff")]
)
def test_a_rollout_limit_is_at_least_0_and_comes_with_an_unfinished_payoff(
    rollout_limit: int, unfinished_payoff: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        search(win_or_lose(), iterations=1, rollout_limit=rollout_limit, unfinished_payoff=unfinished_payoff)


@pytest.mark.parametrize(
    ("iterations", "seconds", "done"),
    [(None, 5.0, 5), (50, 5.0, 5), (3, 100.0, 3), (None, 0.5, 1), (50, 0.5, 1)],
    ids=["seconds alone", "seconds first", "iterations first", "time up after one", "time up before the iterations"],
)
def test_a_search_stops_at_its_seconds_or_its_iterations_whichever_comes_first(
    iterations: int | None, seconds: float, done: int
) -> None:
    result = search(win_or_lose(), iterations, seconds=seconds, time_source=Ticking())

    assert result.iterations == done
    assert sum(item.visits for item in result.statistics) == done


@pytest.mark.parametrize(
    ("iterations", "seconds", "done"),
    [(None, 5.0, 5), (50, 5.0, 5), (3, 100.0, 3), (None, 0.5, 1), (50, 0.5, 1)],
    ids=["seconds alone", "seconds first", "iterations first", "time up after one", "time up before the iterations"],
)
def test_a_search_where_players_act_at_once_stops_at_its_seconds_or_its_iterations_whichever_comes_first(
    iterations: int | None, seconds: float, done: int
) -> None:
    result = rock_paper_scissors_search(iterations, seconds=seconds, time_source=Ticking())

    assert result.iterations == done
    assert sum(item.visits for item in result.statistics) == done


def test_a_search_on_time_logs_its_limits_and_what_it_did_in_them(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    result = search(win_or_lose(), 3, seconds=100.0, time_source=Ticking(0.25))
    search(win_or_lose(), None, seconds=0.5, time_source=Ticking(0.25))

    assert result.seconds == 0.75
    assert "Searching up to 3 iterations or 100 seconds for me" in caplog.messages
    assert "Searched 3 iterations in 0.750 seconds for me" in caplog.messages
    assert "Searching for 0.5 seconds for me" in caplog.messages
    assert "Searched 2 iterations in 0.750 seconds for me" in caplog.messages


@pytest.mark.parametrize(
    ("iterations", "seconds", "message"),
    [(None, None, "iterations, seconds or both"), (0, None, "at least 1 iteration"), (None, 0.0, "more than 0 seconds")],
)
def test_a_search_needs_at_least_one_iteration_or_some_seconds(
    iterations: int | None, seconds: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        search(win_or_lose(), iterations, seconds=seconds)
