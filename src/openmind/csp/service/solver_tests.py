import logging

import pytest

from openmind.csp.factory.csp_factory import create_solver
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.csp.model.variable import Variable
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.all_different import AllDifferent
from openmind.expression.model.any_of import AnyOf
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.expression import Expression
from openmind.expression.model.not_ import Not
from openmind.expression.model.state_variable import StateVariable
from openmind.world.model.action import Action
from openmind.world.model.state import State


def solve(definition: ActionDefinition, state: State, limit: int | None = None) -> tuple[Action, ...]:
    return create_solver().solve(Problem((definition,)), state, limit)


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


def test_an_unknown_parameter_raises() -> None:
    with pytest.raises(KeyError, match="c"):
        solve(set_bits(Equals(ActionParameter("c"), Constant(1))), State(()))


def test_a_constraint_on_three_parameters_is_respected() -> None:
    digit = DiscreteDomain((1, 2))
    definition = ActionDefinition(
        "set",
        (Variable("a", digit), Variable("b", digit), Variable("c", digit)),
        (AnyOf((Equals(ActionParameter("a"), Constant(2)), Equals(ActionParameter("b"), ActionParameter("c")))),),
    )

    assert [dict(action.parameters) for action in solve(definition, State(()))] == [
        {"a": 1, "b": 1, "c": 1},
        {"a": 1, "b": 2, "c": 2},
        {"a": 2, "b": 1, "c": 1},
        {"a": 2, "b": 1, "c": 2},
        {"a": 2, "b": 2, "c": 1},
        {"a": 2, "b": 2, "c": 2},
    ]


def test_all_different_uses_the_values_it_reads_from_the_state() -> None:
    digit = DiscreteDomain((1, 2, 3))
    definition = ActionDefinition(
        "fill",
        (Variable("x", digit), Variable("y", digit)),
        (AllDifferent((ActionParameter("x"), ActionParameter("y"), StateVariable("given"))),),
    )

    assert solve(definition, State((("given", 2),))) == (
        Action("fill", (("x", 1), ("y", 3))),
        Action("fill", (("x", 3), ("y", 1))),
    )


def test_the_limit_caps_the_number_of_solutions() -> None:
    assert len(solve(set_bits(), State(()), limit=3)) == 3


def test_the_same_problem_and_state_come_from_the_cache() -> None:
    solver = create_solver()
    problem = Problem((set_bits(Not(Equals(ActionParameter("a"), ActionParameter("b")))),))

    first = solver.solve_with_statistics(problem, State(()))

    assert solver.solve_with_statistics(problem, State(())) is first
    assert solver.solve(problem, State(())) is first[0]


def test_solve_with_statistics_gives_the_solutions_with_what_the_search_did() -> None:
    problem = Problem((set_bits(Equals(ActionParameter("a"), Constant(1))),))

    assert create_solver().solve_with_statistics(problem, State(())) == (
        (Action("set", (("a", 1), ("b", 0))), Action("set", (("a", 1), ("b", 1)))),
        SolveStatistics(2, 2, 0, 0),
    )


def test_statistics_are_summed_over_the_actions() -> None:
    definition = set_bits(Equals(ActionParameter("a"), Constant(1)))
    other = ActionDefinition("put", definition.variables, definition.constraints)

    _, statistics = create_solver().solve_with_statistics(Problem((definition, other)), State(()))

    assert statistics == SolveStatistics(4, 4, 0, 0)


def test_an_action_with_a_false_constraint_adds_nothing_to_the_statistics() -> None:
    _, statistics = create_solver().solve_with_statistics(Problem((set_bits(Constant(False)),)), State(()))

    assert statistics == SolveStatistics(0, 0, 0, 0)


def test_logs_each_try_and_a_summary(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="openmind.csp")

    solve(set_bits(Equals(ActionParameter("a"), Constant(1))), State(()))
    solve(set_bits(Constant(False)), State(()))

    assert caplog.messages == [
        "set: try b = 0",
        "set: try b = 1",
        "set: 2 solutions, 2 assignments, 0 dead ends, 0 values pruned",
        "set: no solution, constraint is false: False",
    ]
