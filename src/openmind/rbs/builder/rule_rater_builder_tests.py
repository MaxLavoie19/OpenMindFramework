import pytest

from openmind.rbs.builder.rule_rater_builder import RuleRaterBuilder
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.world.model.action import Action
from openmind.world.model.state import State


def test_build_gives_a_rater_using_the_rule_base() -> None:
    rater = RuleRaterBuilder().with_rule_base(RuleBase("test", (Rule("press", (), 0.5, 40),))).build()

    assert rater.rate(State(()), (Action("press", ()),)) == (0.5,)


def test_build_rejects_a_missing_rule_base() -> None:
    with pytest.raises(ValueError, match="rule base"):
        RuleRaterBuilder().build()
