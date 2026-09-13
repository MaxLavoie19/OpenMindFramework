import math

import pytest

from openmind.agent.model.domain import Domain
from openmind.agent.service.agent import Agent
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.csp.service.solver import Solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
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


def win_or_lose() -> Domain:
    no_payoff = Equals(StateVariable("payoff"), Constant(None))
    return Domain(
        "win or lose",
        State((("payoff", None), ("turn", "me"))),
        Problem((ActionDefinition("lose", (), (no_payoff,)), ActionDefinition("win", (), (no_payoff,)))),
        TransitionModel(
            (
                Transition("lose", (Branch(1.0, (Assign(StateVariable("payoff"), Constant(0.0)),)),)),
                Transition("win", (Branch(1.0, (Assign(StateVariable("payoff"), Constant(1.0)),)),)),
            )
        ),
        Players(("me",), "turn", ("payoff",)),
    )


def new_agent() -> Agent:
    names = VariableNameMapper()
    interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
    tree_search = TreeSearch(
        Solver(interpreter, expression_text, action_text),
        Predictor(interpreter, names, expression_text, action_text),
        StateReader(),
        action_text,
    )
    return Agent(tree_search, SearchSettings(50, math.sqrt(2), 1))


def test_choose_returns_the_most_visited_action() -> None:
    domain = win_or_lose()

    assert new_agent().choose(domain, domain.initial_state) == Action("win", ())


def test_search_returns_the_statistics_and_the_tree_samples() -> None:
    domain = win_or_lose()

    result = new_agent().search(domain, domain.initial_state)

    assert result.chosen == Action("win", ())
    assert {sample.action.name for sample in result.samples} == {"lose", "win"}
