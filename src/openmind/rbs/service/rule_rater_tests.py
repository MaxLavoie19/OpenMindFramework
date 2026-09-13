from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.service.rule_rater import RuleRater
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State

ANY_STATE = Rule("press", (), 0.5, 40)
LIGHT_ON = Rule("press", (Equals(StateVariable("light"), Constant("on")),), 0.9, 20)
LIGHT_ON_CELL_1 = Rule(
    "press", (Equals(StateVariable("light"), Constant("on")), Equals(ActionParameter("cell"), Constant(1))), 1.0, 10
)
LAMP_ON = Rule("press", (Equals(StateVariable("lamp"), Constant("on")),), 0.2, 30)


def new_rater(*rules: Rule) -> RuleRater:
    return RuleRater(RuleBase("test", rules), Interpreter(VariableNameMapper()))


def press(cell: int) -> Action:
    return Action("press", (("cell", cell),))


def test_each_action_gets_its_most_specific_matching_rule() -> None:
    rater = new_rater(ANY_STATE, LIGHT_ON, LIGHT_ON_CELL_1)

    assert rater.rate(State((("light", "on"),)), (press(1), press(2))) == (1.0, 0.9)
    assert rater.rate(State((("light", "off"),)), (press(1),)) == (0.5,)


def test_an_action_without_rules_gets_none() -> None:
    assert new_rater(ANY_STATE).rate(State((("light", "on"),)), (Action("pull", ()),)) == (None,)


def test_explain_gives_the_rule_behind_the_rating() -> None:
    rater = new_rater(ANY_STATE, LIGHT_ON, LIGHT_ON_CELL_1)

    assert rater.explain(State((("light", "on"),)), press(2)) == LIGHT_ON


def test_a_condition_on_a_missing_variable_does_not_hold() -> None:
    rater = new_rater(ANY_STATE, LAMP_ON)

    assert rater.explain(State((("light", "on"),)), press(1)) == ANY_STATE
