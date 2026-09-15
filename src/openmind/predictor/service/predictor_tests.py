import logging

import pytest

from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action
from openmind.world.model.state import State


def predict(state: State, *branches: Branch, definitions: str | None = None) -> tuple[tuple[State, float], ...]:
    model = TransitionModel((Transition("move", branches),), None if definitions is None else PythonRule(definitions))
    return create_predictor().predict(model, state, Action("move", (("to", 2),))).outcomes


def certain(effects: str) -> Branch:
    return Branch(1.0, PythonRule(effects))


def test_effects_set_state_variables_from_the_parameters() -> None:
    assert predict(State((("position", 1),)), certain("position = to")) == ((State((("position", 2),)), 1.0),)


def test_an_index_selects_the_variable() -> None:
    outcomes = predict(State((("cell(1)", None), ("cell(2)", None))), certain("cell[to] = 'X'"))

    assert outcomes == ((State((("cell(1)", None), ("cell(2)", "X"))), 1.0),)


def test_effects_run_in_order_and_see_what_came_before() -> None:
    effects = "position = to\nif position == 2:\n    arrived = True"

    outcomes = predict(State((("arrived", False), ("position", 1))), certain(effects))

    assert outcomes == ((State((("arrived", True), ("position", 2))), 1.0),)


def test_effects_see_the_definitions() -> None:
    outcomes = predict(State((("score", 1),)), certain("score = score + BONUS"), definitions="BONUS = 5")

    assert outcomes == ((State((("score", 6),)), 1.0),)


def test_each_branch_gives_an_outcome_with_its_probability() -> None:
    hit = Branch(0.25, PythonRule("score = 1"))
    miss = Branch(0.75, PythonRule(""))

    assert predict(State((("score", 0),)), hit, miss) == (
        (State((("score", 1),)), 0.25),
        (State((("score", 0),)), 0.75),
    )


def test_an_effect_on_an_index_the_state_lacks_adds_the_variable() -> None:
    outcomes = predict(State((("cell(1)", None), ("cell(2)", None))), certain("cell[3] = 'X'"))

    assert outcomes == ((State((("cell(1)", None), ("cell(2)", None), ("cell(3)", "X"))), 1.0),)


def test_action_without_a_transition_raises() -> None:
    with pytest.raises(KeyError, match="jump"):
        create_predictor().predict(TransitionModel(()), State(()), Action("jump", ()))


def test_logs_the_changes_and_the_outcomes(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="openmind.predictor.service.predictor")

    predict(State((("arrived", False), ("position", 1))), certain("position = to\nif position == 2:\n    arrived = True"))

    assert caplog.messages == [
        "Set arrived = True",
        "Set position = 2",
        "move(to=2) gives 1 outcome(s) with probabilities [1.0]",
    ]
