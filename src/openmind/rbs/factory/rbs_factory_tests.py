from openmind.rbs.factory.rbs_factory import (
    create_rule_generator,
    create_rule_rater,
    create_rule_valuer,
    create_value_generator,
)
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.service.rule_generator import RuleGenerator
from openmind.rbs.service.rule_rater_tests import LIGHTS
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.world.model.action import Action
from openmind.world.model.state import State


def test_create_rule_generator_gives_a_rule_generator() -> None:
    assert isinstance(create_rule_generator(), RuleGenerator)


def test_create_rule_rater_rates_with_the_rule_base() -> None:
    rater = create_rule_rater(RuleBase("test", (Rule("press", (), 0.5, 40),)), LIGHTS)

    assert rater.rate(State(()), (Action("press", ()),)) == (0.5,)


def test_create_value_generator_gives_a_value_generator() -> None:
    assert isinstance(create_value_generator(), ValueGenerator)


def test_create_rule_valuer_values_with_the_value_base() -> None:
    valuer = create_rule_valuer(ValueBase("test", 0.0, 0.0, 2.0, ()), LIGHTS)

    assert valuer.value(State(())) == (1.0,)
