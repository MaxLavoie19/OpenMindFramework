import pytest

from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.expression import Expression
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.mcts.model.action_sample import ActionSample
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.rbs.model.induction_settings import InductionSettings
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.service.rule_inducer import RuleInducer
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value

LIGHT_ON = Equals(StateVariable("light"), Constant("on"))
LIGHT_OFF = Equals(StateVariable("light"), Constant("off"))
CELL_1 = Equals(ActionParameter("cell"), Constant(1))
CELL_2 = Equals(ActionParameter("cell"), Constant(2))


def induce(samples: list[ActionSample], settings: InductionSettings) -> RuleBase:
    names = VariableNameMapper()
    inducer = RuleInducer(Interpreter(names), names, RuleTextMapper(ExpressionTextMapper(names)))
    return inducer.induce("test", tuple(samples), settings)


def press(variables: dict[str, Value], cell: int, payoff: float, visits: int = 10) -> ActionSample:
    state = State(tuple(sorted((variables | {"turn": "me"}).items())))
    return ActionSample(state, 0, Action("press", (("cell", cell),)), visits, payoff)


def summary(rule_base: RuleBase) -> list[tuple[str, tuple[Expression, ...], object, int]]:
    return [(rule.action, rule.conditions, rule.expected_value, rule.visits) for rule in rule_base.rules]


def test_a_state_variable_condition_becomes_a_rule_when_it_moves_the_expected_value() -> None:
    samples = [
        press({"light": "on"}, 1, 0.9),
        press({"light": "on"}, 2, 0.9),
        press({"light": "off"}, 1, 0.1),
        press({"light": "off"}, 2, 0.1),
    ]

    rule_base = induce(samples, InductionSettings(min_visits=1, max_conditions=1, min_rule_visits=1, min_gain=0.1))

    assert summary(rule_base) == [
        ("press", (), pytest.approx(0.5), 40),
        ("press", (LIGHT_ON,), pytest.approx(0.9), 20),
        ("press", (LIGHT_OFF,), pytest.approx(0.1), 20),
    ]


def test_an_indexed_variable_at_the_action_parameters_becomes_a_rule() -> None:
    samples = [
        press({"lamp(1)": "on", "lamp(2)": "off"}, 1, 0.9),
        press({"lamp(1)": "on", "lamp(2)": "off"}, 2, 0.1),
        press({"lamp(1)": "off", "lamp(2)": "on"}, 1, 0.1),
        press({"lamp(1)": "off", "lamp(2)": "on"}, 2, 0.9),
    ]

    rule_base = induce(samples, InductionSettings(min_visits=1, max_conditions=1, min_rule_visits=1, min_gain=0.1))

    lamp_at_cell = StateVariable("lamp", (ActionParameter("cell"),))
    assert summary(rule_base) == [
        ("press", (), pytest.approx(0.5), 40),
        ("press", (Equals(lamp_at_cell, Constant("on")),), pytest.approx(0.9), 20),
        ("press", (Equals(lamp_at_cell, Constant("off")),), pytest.approx(0.1), 20),
    ]


def test_max_conditions_limits_how_many_conditions_a_rule_combines() -> None:
    samples = [
        press({"light": "on"}, 1, 1.0),
        press({"light": "on"}, 2, 0.5),
        press({"light": "off"}, 1, 0.0),
        press({"light": "off"}, 2, 0.0),
    ]

    one = induce(samples, InductionSettings(min_visits=1, max_conditions=1, min_rule_visits=1, min_gain=0.1))
    two = induce(samples, InductionSettings(min_visits=1, max_conditions=2, min_rule_visits=1, min_gain=0.1))

    assert [rule.conditions for rule in one.rules] == [(), (LIGHT_ON,), (LIGHT_OFF,), (CELL_1,), (CELL_2,)]
    assert summary(two)[5:] == [
        ("press", (LIGHT_ON, CELL_1), pytest.approx(1.0), 10),
        ("press", (LIGHT_ON, CELL_2), pytest.approx(0.5), 10),
    ]


def test_samples_below_min_visits_are_ignored() -> None:
    samples = [press({"light": "on"}, 1, 0.9), press({"light": "off"}, 1, 0.0, visits=2)]

    rule_base = induce(samples, InductionSettings(min_visits=5, max_conditions=1, min_rule_visits=1, min_gain=0.1))

    assert summary(rule_base) == [("press", (), pytest.approx(0.9), 10)]


def test_rules_below_min_rule_visits_are_left_out() -> None:
    samples = [press({"light": "on"}, 1, 0.9, visits=40), press({"light": "off"}, 1, 0.1, visits=5)]

    rule_base = induce(samples, InductionSettings(min_visits=1, max_conditions=1, min_rule_visits=10, min_gain=0.05))

    assert [rule.conditions for rule in rule_base.rules] == [(), (LIGHT_ON,)]
