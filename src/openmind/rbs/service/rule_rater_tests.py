from openmind.agent.model.domain import Domain
from openmind.csp.model.problem import Problem
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rbs.factory.rbs_factory import create_rule_rater
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.service.consequence_library_tests import place, position, strip_domain
from openmind.rbs.service.rule_rater import RuleRater
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State

LIGHTS = Domain("lights", State((("light", "on"),)), Problem(()), TransitionModel(()), Players(("me",), "turn", ("payoff",)))
ANY_STATE = Rule("press", (), 0.5, 40)
LIGHT_ON = Rule("press", (PythonRule("light == 'on'"),), 0.9, 20)
LIGHT_ON_CELL_1 = Rule("press", (PythonRule("light == 'on'"), PythonRule("cell == 1")), 1.0, 10)
LAMP_ON = Rule("press", (PythonRule("lamp == 'on'"),), 0.2, 30)


def new_rater(*rules: Rule, domain: Domain = LIGHTS) -> RuleRater:
    return create_rule_rater(RuleBase("test", rules), domain)


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


def test_a_priority_rule_rates_before_more_specific_rules() -> None:
    avoid = Rule("press", (PythonRule("cell == 1"),), 0.0, 5, priority=True)

    assert new_rater(ANY_STATE, LIGHT_ON_CELL_1, avoid).rate(State((("light", "on"),)), (press(1),)) == (0.0,)


def test_conditions_read_the_consequences_of_the_domain_rules() -> None:
    wins = Rule("place", (PythonRule("win_chance(action) >= 1"),), 1.0, 30)
    rater = new_rater(Rule("place", (), 0.5, 90), wins, domain=strip_domain())

    assert rater.rate(position({1: "X"}, "X"), (place(2), place(3))) == (1.0, 0.5)
