import logging
import math

import pytest

from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.csp.service.solver import Solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.mcts.model.guidance import Guidance
from openmind.mcts.model.search_result import SearchResult
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.tree_search import TreeSearch
from openmind.predictor.model.assign import Assign
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
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


def search(
    game: Game,
    iterations: int,
    exploration: float = math.sqrt(2),
    seed: int | None = 1,
    guidance: Guidance | None = None,
) -> SearchResult:
    names = VariableNameMapper()
    interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
    tree_search = TreeSearch(
        Solver(interpreter, expression_text, action_text),
        Predictor(interpreter, names, expression_text, action_text),
        StateReader(),
        action_text,
    )
    problem, transitions, state = game
    players = Players(("me",), "turn", ("payoff",))
    settings = SearchSettings(iterations, exploration, seed)
    return tree_search.search(problem, transitions, players, state, settings, guidance)


def one_move_game(*transitions: Transition) -> Game:
    """Every action is legal until the payoff is set, and every action sets it."""
    no_payoff = Equals(StateVariable("payoff"), Constant(None))
    problem = Problem(tuple(ActionDefinition(transition.action, (), (no_payoff,)) for transition in transitions))
    return problem, TransitionModel(transitions), State((("payoff", None), ("turn", "me")))


def two_step_game() -> Game:
    """go moves to stage 1, where win pays 1.0 and lose pays 0.0."""
    unset = Equals(StateVariable("payoff"), Constant(None))
    problem = Problem(
        (
            ActionDefinition("go", (), (unset, Equals(StateVariable("stage"), Constant(0)))),
            ActionDefinition("lose", (), (unset, Equals(StateVariable("stage"), Constant(1)))),
            ActionDefinition("win", (), (unset, Equals(StateVariable("stage"), Constant(1)))),
        )
    )
    transitions = TransitionModel(
        (
            Transition("go", (Branch(1.0, (Assign(StateVariable("stage"), Constant(1)),)),)),
            Transition("lose", (Branch(1.0, (pay(0.0),)),)),
            Transition("win", (Branch(1.0, (pay(1.0),)),)),
        )
    )
    return problem, transitions, State((("payoff", None), ("stage", 0), ("turn", "me")))


def pay(payoff: float) -> Assign:
    return Assign(StateVariable("payoff"), Constant(payoff))


def win_or_lose() -> Game:
    return one_move_game(Transition("lose", (Branch(1.0, (pay(0.0),)),)), Transition("win", (Branch(1.0, (pay(1.0),)),)))


def test_picks_a_winning_action_over_a_losing_one() -> None:
    result = search(win_or_lose(), iterations=50)

    assert result.chosen == Action("win", ())
    assert {item.action.name: item.mean_payoff for item in result.statistics} == {"lose": 0.0, "win": 1.0}
    assert sum(item.visits for item in result.statistics) == 50


def test_prefers_a_sure_payoff_to_a_coin_flip_with_a_lower_mean() -> None:
    game = one_move_game(
        Transition("gamble", (Branch(0.5, (pay(1.0),)), Branch(0.5, (pay(0.0),)))),
        Transition("safe", (Branch(1.0, (pay(0.8),)),)),
    )

    assert search(game, iterations=300, exploration=1.0).chosen == Action("safe", ())


def test_the_same_seed_gives_the_same_result() -> None:
    game = one_move_game(
        Transition("gamble", (Branch(0.5, (pay(1.0),)), Branch(0.5, (pay(0.0),)))),
        Transition("safe", (Branch(1.0, (pay(0.6),)),)),
    )

    assert search(game, iterations=100, seed=3) == search(game, iterations=100, seed=3)


def test_no_legal_action_with_an_unset_payoff_raises() -> None:
    problem = Problem((ActionDefinition("finish", (), (Equals(StateVariable("done"), Constant(False)),)),))
    finish = Transition("finish", (Branch(1.0, (Assign(StateVariable("done"), Constant(True)),)),))
    game = (problem, TransitionModel((finish,)), State((("done", False), ("payoff", None), ("turn", "me"))))

    with pytest.raises(ValueError, match="payoff"):
        search(game, iterations=1)


def test_no_legal_action_at_the_root_raises() -> None:
    problem, transitions, _ = one_move_game(Transition("win", (Branch(1.0, (pay(1.0),)),)))

    with pytest.raises(ValueError, match="No legal action"):
        search((problem, transitions, State((("payoff", 1.0), ("turn", "me")))), iterations=1)


def test_logs_the_search_summary(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    search(win_or_lose(), iterations=2)

    assert [record.getMessage() for record in caplog.records if record.name == "openmind.mcts.service.tree_search"] == [
        "Searching 2 iterations for me",
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
