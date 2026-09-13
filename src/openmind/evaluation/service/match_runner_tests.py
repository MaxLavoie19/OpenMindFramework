import random

import pytest

from openmind.agent.model.domain import Domain
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.csp.service.solver import Solver
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
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


class Always:
    """A policy that always chooses the same action."""

    def __init__(self, name: str) -> None:
        self._action = Action(name, ())

    def choose(self, domain: Domain, state: State) -> Action:
        return self._action


def new_match_runner() -> MatchRunner:
    names = VariableNameMapper()
    interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
    return MatchRunner(
        Solver(interpreter, expression_text, action_text),
        Predictor(interpreter, names, expression_text, action_text),
        StateReader(),
    )


def first_mover_decides(players: tuple[str, ...] = ("A", "B")) -> Domain:
    """Only the first player acts: win gives them 1 and the other 0, lose the reverse, tie 0.5 each."""
    first, *others = players
    unset = Equals(StateVariable("payoff", (Constant(first),)), Constant(None))

    def payoffs(mine: float, theirs: float) -> tuple[Assign, ...]:
        return (
            Assign(StateVariable("payoff", (Constant(first),)), Constant(mine)),
            *(Assign(StateVariable("payoff", (Constant(other),)), Constant(theirs)) for other in others),
        )

    return Domain(
        "first mover decides",
        State((*((f"payoff({player})", None) for player in sorted(players)), ("turn", first))),
        Problem(tuple(ActionDefinition(name, (), (unset,)) for name in ("win", "lose", "tie"))),
        TransitionModel(
            (
                Transition("win", (Branch(1.0, payoffs(1.0, 0.0)),)),
                Transition("lose", (Branch(1.0, payoffs(0.0, 1.0)),)),
                Transition("tie", (Branch(1.0, payoffs(0.5, 0.5)),)),
            )
        ),
        Players(players, "turn", tuple(f"payoff({player})" for player in players)),
    )


def series(evaluated: str, opponent: str, games: int = 2) -> MatchResults:
    return new_match_runner().series(
        first_mover_decides(), Always(evaluated), Always(opponent), "always " + opponent, games, random.Random(1)
    )


def test_seats_switch_every_game() -> None:
    # game 1: the evaluated policy is A and wins; game 2: the opponent is A, loses, so the evaluated B wins
    assert series("win", "lose") == MatchResults("always lose", 2, 2, 0, 0)


def test_losses_and_wins_count_from_the_evaluated_side() -> None:
    assert series("lose", "lose") == MatchResults("always lose", 2, 1, 0, 1)


def test_equal_payoffs_are_draws() -> None:
    assert series("tie", "tie", games=3) == MatchResults("always tie", 3, 0, 3, 0)


def test_a_domain_without_two_players_raises() -> None:
    with pytest.raises(ValueError, match="two players"):
        new_match_runner().series(
            first_mover_decides(("A",)), Always("win"), Always("win"), "always win", 1, random.Random(1)
        )
