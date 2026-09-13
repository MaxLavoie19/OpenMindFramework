import logging

import pytest

from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.model.assign import Assign
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.effect import Effect
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.predictor.model.when import When
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State


def new_predictor() -> Predictor:
    names = VariableNameMapper()
    return Predictor(Interpreter(names), names, ExpressionTextMapper(names), ActionTextMapper())


def predict(state: State, *branches: Branch) -> tuple[tuple[State, float], ...]:
    model = TransitionModel((Transition("move", branches),))
    return new_predictor().predict(model, state, Action("move", (("to", 2),))).outcomes


def certain(*effects: Effect) -> Branch:
    return Branch(1.0, effects)


def test_assign_sets_a_state_variable() -> None:
    outcomes = predict(State((("position", 1),)), certain(Assign(StateVariable("position"), ActionParameter("to"))))

    assert outcomes == ((State((("position", 2),)), 1.0),)


def test_assign_target_indices_complete_the_name() -> None:
    state = State((("cell(1)", None), ("cell(2)", None)))

    outcomes = predict(state, certain(Assign(StateVariable("cell", (ActionParameter("to"),)), Constant("X"))))

    assert outcomes == ((State((("cell(1)", None), ("cell(2)", "X"))), 1.0),)


def test_effects_see_earlier_effects() -> None:
    move = Assign(StateVariable("position"), ActionParameter("to"))
    arrive = When(Equals(StateVariable("position"), Constant(2)), (Assign(StateVariable("arrived"), Constant(True)),))

    outcomes = predict(State((("arrived", False), ("position", 1))), certain(move, arrive))

    assert outcomes == ((State((("arrived", True), ("position", 2))), 1.0),)


def test_when_applies_otherwise_when_the_condition_is_false() -> None:
    toggle = When(
        Equals(StateVariable("light"), Constant("on")),
        (Assign(StateVariable("light"), Constant("off")),),
        (Assign(StateVariable("light"), Constant("on")),),
    )

    assert predict(State((("light", "off"),)), certain(toggle)) == ((State((("light", "on"),)), 1.0),)


def test_each_branch_gives_an_outcome_with_its_probability() -> None:
    hit = Branch(0.25, (Assign(StateVariable("score"), Constant(1)),))
    miss = Branch(0.75, ())

    assert predict(State((("score", 0),)), hit, miss) == (
        (State((("score", 1),)), 0.25),
        (State((("score", 0),)), 0.75),
    )


def test_assign_to_an_unknown_variable_raises() -> None:
    with pytest.raises(KeyError, match="speed"):
        predict(State(()), certain(Assign(StateVariable("speed"), Constant(3))))


def test_non_boolean_condition_raises() -> None:
    with pytest.raises(TypeError, match="position"):
        predict(State((("position", 1),)), certain(When(StateVariable("position"), ())))


def test_action_without_a_transition_raises() -> None:
    with pytest.raises(KeyError, match="jump"):
        new_predictor().predict(TransitionModel(()), State(()), Action("jump", ()))


def test_logs_effects_and_outcomes_as_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="openmind.predictor.service.predictor")
    move = Assign(StateVariable("position"), ActionParameter("to"))
    arrive = When(Equals(StateVariable("position"), Constant(2)), (Assign(StateVariable("arrived"), Constant(True)),))

    predict(State((("arrived", False), ("position", 1))), certain(move, arrive))

    assert caplog.messages == [
        "Set position = 2",
        "When position == 2: true",
        "Set arrived = True",
        "move(to=2) gives 1 outcome(s) with probabilities [1.0]",
    ]
