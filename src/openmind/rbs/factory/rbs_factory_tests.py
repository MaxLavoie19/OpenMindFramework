from openmind.rbs.factory.rbs_factory import create_rule_inducer, create_rule_rater
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.service.rule_inducer import RuleInducer
from openmind.world.model.action import Action
from openmind.world.model.state import State


def test_create_rule_inducer_gives_a_rule_inducer() -> None:
    assert isinstance(create_rule_inducer(), RuleInducer)


def test_create_rule_rater_rates_with_the_rule_base() -> None:
    rater = create_rule_rater(RuleBase("test", (Rule("press", (), 0.5, 40),)))

    assert rater.rate(State(()), (Action("press", ()),)) == (0.5,)
