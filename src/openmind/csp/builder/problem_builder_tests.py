import pytest

from openmind.csp.builder.problem_builder import ProblemBuilder
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.expression.model.constant import Constant


def test_build_keeps_actions_in_order() -> None:
    row = Variable("row", DiscreteDomain((1, 2)))

    problem = (
        ProblemBuilder().with_action("place", (row,), (Constant(True),)).with_action("pass", (), ()).build()
    )

    assert problem == Problem(
        (ActionDefinition("place", (row,), (Constant(True),)), ActionDefinition("pass", (), ()))
    )


def test_with_action_rejects_an_action_already_defined() -> None:
    builder = ProblemBuilder().with_action("pass", (), ())

    with pytest.raises(ValueError, match="pass"):
        builder.with_action("pass", (), ())


def test_with_action_rejects_repeated_variable_names() -> None:
    row = Variable("row", DiscreteDomain((1, 2)))

    with pytest.raises(ValueError, match="row"):
        ProblemBuilder().with_action("place", (row, row), ())
