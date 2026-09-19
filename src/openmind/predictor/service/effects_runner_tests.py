import logging

import pytest

from openmind.predictor.service.effects_runner import EffectsRunner
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State


type Outcome = tuple[float, PythonRule]


def runner() -> EffectsRunner:
    return EffectsRunner(create_rule_caller(), ActionTextMapper())


def predict(state: State, *outcomes: Outcome, definitions: str | None = None) -> tuple[tuple[State, float], ...]:
    seen = None if definitions is None else PythonRule(definitions)
    return runner().run(state, Action("move", (("to", 2),)), outcomes, seen).outcomes


def certain(effects: str) -> Outcome:
    return 1.0, PythonRule(effects)


def test_actions_taken_at_once_run_in_turn_multiplying_their_branches_then_the_resolution_runs() -> None:
    pick = (
        (0.5, PythonRule("picked = picked.with_item(player, number)")),
        (0.5, PythonRule("picked = picked.with_item(player, number + 1)")),
    )
    together = (certain("total = picked['A'] + picked['B']"),)
    state = State.of(picked=Map.of({"A": None, "B": None}), total=None)
    joint = JointAction((("A", Action("pick", (("number", 1),))), ("B", Action("pick", (("number", 10),)))))

    outcomes = runner().run_joint(state, joint, {"pick": pick}, together).outcomes

    assert [(outcome.value("total"), probability) for outcome, probability in outcomes] == [
        (11, 0.25),
        (12, 0.25),
        (12, 0.25),
        (13, 0.25),
    ]


def test_an_action_taken_at_once_with_a_parameter_named_player_raises() -> None:
    effects = {"pick": (certain("x = 1"),)}

    with pytest.raises(ValueError, match="parameter named 'player'"):
        runner().run_joint(
            State.of(x=0), JointAction((("A", Action("pick", (("player", "B"),))),)), effects
        )


def test_effects_set_state_variables_from_the_parameters() -> None:
    assert predict(State.of(position=1), certain("position = to")) == ((State.of(position=2), 1.0),)


def test_a_parameter_selects_the_cell_an_effect_places_into() -> None:
    outcomes = predict(State.of(cell=Grid.filled((2,), None)), certain("cell = cell.placed((to,), 'X')"))

    assert outcomes == ((State.of(cell=Grid((2,), (None, "X"))), 1.0),)


def test_effects_run_in_order_and_see_what_came_before() -> None:
    effects = "position = to\nif position == 2:\n    arrived = True"

    outcomes = predict(State.of(arrived=False, position=1), certain(effects))

    assert outcomes == ((State.of(arrived=True, position=2), 1.0),)


def test_effects_see_the_definitions() -> None:
    outcomes = predict(State.of(score=1), certain("score = score + BONUS"), definitions="BONUS = 5")

    assert outcomes == ((State.of(score=6), 1.0),)


def test_each_branch_gives_an_outcome_with_its_probability() -> None:
    hit = (0.25, PythonRule("score = 1"))
    miss = (0.75, PythonRule(""))

    assert predict(State.of(score=0), hit, miss) == (
        (State.of(score=1), 0.25),
        (State.of(score=0), 0.75),
    )


def test_an_action_without_an_effects_rule_raises() -> None:
    with pytest.raises(KeyError, match="jump"):
        runner().run(State(()), Action("jump", ()), ())


def test_logs_the_changes_and_the_outcomes(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="openmind.predictor.service.effects_runner")

    predict(State.of(arrived=False, position=1), certain("position = to\nif position == 2:\n    arrived = True"))

    assert caplog.messages == [
        "Set arrived = True",
        "Set position = 2",
        "move(to=2) gives 1 outcome(s) with probabilities [1.0]",
    ]
