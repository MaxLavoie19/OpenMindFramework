import pytest

from openmind.rbs.builder.rule_valuer_builder import RuleValuerBuilder
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.service.rule_rater_tests import LIGHTS
from openmind.world.model.state import State


def test_build_gives_a_valuer_using_the_value_base() -> None:
    valuer = RuleValuerBuilder().with_value_base(ValueBase("lights", 0.0, 0.0, 1.0, ())).with_domain(LIGHTS).build()

    assert valuer.value(State((("light", "on"),))) == (0.5,)


def test_build_rejects_a_missing_value_base() -> None:
    with pytest.raises(ValueError, match="value base"):
        RuleValuerBuilder().with_domain(LIGHTS).build()


def test_build_rejects_a_missing_domain() -> None:
    with pytest.raises(ValueError, match="domain"):
        RuleValuerBuilder().with_value_base(ValueBase("lights", 0.0, 0.0, 1.0, ())).build()
