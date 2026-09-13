import pytest

from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.csp.service.solver import Solver
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.expression import Expression
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State


def solve(definition: ActionDefinition, state: State) -> tuple[Action, ...]:
    return Solver(Interpreter(VariableNameMapper())).solve(Problem((definition,)), state)


def set_bits(*constraints: Expression) -> ActionDefinition:
    bit = DiscreteDomain((0, 1))
    return ActionDefinition("set", (Variable("a", bit), Variable("b", bit)), constraints)


def test_without_constraints_every_combination_is_legal() -> None:
    assert solve(set_bits(), State(())) == (
        Action("set", (("a", 0), ("b", 0))),
        Action("set", (("a", 0), ("b", 1))),
        Action("set", (("a", 1), ("b", 0))),
        Action("set", (("a", 1), ("b", 1))),
    )


def test_constraints_prune_combinations() -> None:
    assert solve(set_bits(Equals(ActionParameter("a"), ActionParameter("b"))), State(())) == (
        Action("set", (("a", 0), ("b", 0))),
        Action("set", (("a", 1), ("b", 1))),
    )


def test_constraints_read_the_state() -> None:
    switch_off = ActionDefinition("switch_off", (), (Equals(StateVariable("light"), Constant("on")),))

    assert solve(switch_off, State((("light", "on"),))) == (Action("switch_off", ()),)
    assert solve(switch_off, State((("light", "off"),))) == ()


def test_unsatisfiable_constraints_give_no_actions() -> None:
    assert solve(set_bits(Constant(False)), State(())) == ()


def test_parameters_are_sorted_by_name() -> None:
    one = DiscreteDomain((1,))
    place = ActionDefinition("place", (Variable("row", one), Variable("col", one)), ())

    assert solve(place, State(())) == (Action("place", (("col", 1), ("row", 1))),)


def test_non_boolean_constraint_raises() -> None:
    with pytest.raises(TypeError, match="set"):
        solve(set_bits(ActionParameter("a")), State(()))
