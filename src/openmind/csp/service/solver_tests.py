import logging
import pickle

import pytest

from openmind.csp.factory.csp_factory import create_solver
from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.grid import Grid
from openmind.world.model.action import Action
from openmind.world.model.state import State


type Action_ = tuple[str, dict[str, PythonRule], tuple[PythonRule, ...]]


def solve(action: Action_, state: State, limit: int | None = None) -> tuple[Action, ...]:
    name, values, constraints = action
    return create_solver().solve(state, name, values, constraints, limit=limit)


def solve_seeing(action: Action_, state: State, definitions: PythonRule) -> tuple[Action, ...]:
    """Solving where the action's rules see a definitions script."""
    name, values, constraints = action
    return create_solver().solve(state, name, values, constraints, definitions)


def set_bits(*constraints: str) -> Action_:
    bit = PythonRule("(0, 1)")
    return "set", {"a": bit, "b": bit}, tuple(map(PythonRule, constraints))


def test_without_constraints_every_combination_is_legal() -> None:
    assert solve(set_bits(), State(())) == (
        Action("set", (("a", 0), ("b", 0))),
        Action("set", (("a", 0), ("b", 1))),
        Action("set", (("a", 1), ("b", 0))),
        Action("set", (("a", 1), ("b", 1))),
    )


def test_constraints_prune_combinations() -> None:
    assert solve(set_bits("a == b"), State(())) == (
        Action("set", (("a", 0), ("b", 0))),
        Action("set", (("a", 1), ("b", 1))),
    )


def test_constraints_read_the_state() -> None:
    switch_off = ("switch_off", {}, (PythonRule("light == 'on'"),))

    assert solve(switch_off, State.of(light="on")) == (Action("switch_off", ()),)
    assert solve(switch_off, State.of(light="off")) == ()


def test_constraints_see_the_problem_definitions() -> None:
    action, definitions = set_bits("a == LIMIT", "b < LIMIT"), PythonRule("LIMIT = 1")

    assert solve_seeing(action, State(()), definitions) == (Action("set", (("a", 1), ("b", 0))),)


def test_unsatisfiable_constraints_give_no_actions() -> None:
    assert solve(set_bits("False"), State(())) == ()


def test_parameters_are_sorted_by_name() -> None:
    one = PythonRule("(1,)")
    place = ("place", {"row": one, "col": one}, ())

    assert solve(place, State(())) == (Action("place", (("col", 1), ("row", 1))),)


def test_non_boolean_constraint_raises() -> None:
    with pytest.raises(TypeError, match="set"):
        solve(set_bits("a"), State(()))


def test_a_name_that_is_neither_a_parameter_nor_defined_raises() -> None:
    with pytest.raises(NameError, match="c"):
        solve(set_bits("c == 1"), State(()))


def test_a_constraint_on_three_parameters_is_respected() -> None:
    digit = PythonRule("(1, 2)")
    definition = ("set", {"a": digit, "b": digit, "c": digit}, (PythonRule("a == 2 or b == c"),))

    assert [dict(action.parameters) for action in solve(definition, State(()))] == [
        {"a": 1, "b": 1, "c": 1},
        {"a": 1, "b": 2, "c": 2},
        {"a": 2, "b": 1, "c": 1},
        {"a": 2, "b": 1, "c": 2},
        {"a": 2, "b": 2, "c": 1},
        {"a": 2, "b": 2, "c": 2},
    ]


def test_all_different_uses_the_values_it_reads_from_the_state() -> None:
    digit = PythonRule("(1, 2, 3)")
    definition = ("fill", {"x": digit, "y": digit}, (PythonRule("all_different(x, y, given)"),))

    assert solve(definition, State.of(given=2)) == (
        Action("fill", (("x", 1), ("y", 3))),
        Action("fill", (("x", 3), ("y", 1))),
    )


def test_a_state_domain_gives_a_parameter_the_values_its_rule_reads_from_the_state() -> None:
    go = ("go", {"to": PythonRule("DOORS[room] + ['b']")}, (PythonRule("to != 'c'"),))
    definitions = PythonRule("DOORS = {'hall': ['b', 'a', 'c']}")

    # The values keep the rule's order, b given twice counts once, and constraints still filter them.
    assert solve_seeing(go, State.of(room="hall"), definitions) == (
        Action("go", (("to", "b"),)),
        Action("go", (("to", "a"),)),
    )


def test_a_state_domain_is_read_only_once_the_constraints_without_parameters_hold() -> None:
    go = ("go", {"to": PythonRule("DOORS[room]")}, (PythonRule("open == True"),))

    # DOORS has no hall: reading the parameter's values would raise KeyError.
    assert solve_seeing(go, State.of(open=False, room="hall"), PythonRule("DOORS = {}")) == ()


def test_the_limit_caps_the_number_of_solutions() -> None:
    assert len(solve(set_bits(), State(()), limit=3)) == 3


def test_the_same_action_and_state_come_from_the_cache() -> None:
    solver = create_solver()
    name, values, constraints = set_bits("not a == b")

    first = solver.solve_with_statistics(State(()), name, values, constraints)

    assert solver.solve_with_statistics(State(()), name, values, constraints) is first
    assert solver.solve(State(()), name, values, constraints) is first[0]


def test_solve_with_statistics_gives_the_solutions_with_what_the_search_did() -> None:
    name, values, constraints = set_bits("a == 1")

    assert create_solver().solve_with_statistics(State(()), name, values, constraints) == (
        (Action("set", (("a", 1), ("b", 0))), Action("set", (("a", 1), ("b", 1)))),
        SolveStatistics(2, 2, 0, 0),
    )


def test_an_action_with_a_false_constraint_adds_nothing_to_the_statistics() -> None:
    name, values, constraints = set_bits("False")

    _, statistics = create_solver().solve_with_statistics(State(()), name, values, constraints)

    assert statistics == SolveStatistics(0, 0, 0, 0)


def test_a_copy_sent_to_another_process_leaves_its_cache_behind_and_solves_alike() -> None:
    solver, action = create_solver(), set_bits("a == 1")
    solutions = solve(action, State(()))

    copy = pickle.loads(pickle.dumps(solver))

    assert solve(action, State(())) == solutions


def test_logs_each_try_and_a_summary(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="openmind.csp")

    solve(set_bits("a == 1"), State(()))
    solve(set_bits("False"), State(()))

    assert caplog.messages == [
        "set: try b = 0",
        "set: try b = 1",
        "set: 2 solutions, 2 assignments, 0 dead ends, 0 values pruned",
        "set: no solution, constraint is false: False",
    ]


def test_allows_checks_one_action_against_its_constraints_rather_than_searching() -> None:
    """An action a model computed isn't legal by construction: its constraints run against the values it carries."""
    state = State.of(cell=Grid.filled((3, 3), None))
    values = {"row": PythonRule("(1, 2, 3)"), "col": PythonRule("(1, 2, 3)")}
    empty = (PythonRule("cell[row, col] is None"),)

    solver = create_solver()

    assert solver.allows(state, Action("place", (("col", 2), ("row", 2))), values, empty)
    assert not solver.allows(state, Action("place", (("col", 9), ("row", 2))), values, empty)
    assert not solver.allows(state, Action("place", (("row", 2),)), values, empty)
    assert not solver.allows(
        State.of(cell=Grid((3, 3), ("X",) + (None,) * 8)), Action("place", (("col", 1), ("row", 1))), values, empty
    )


def tour(*domains: tuple[int, ...]) -> Action_:
    """A circuit over as many parameters as domains given, each its own values."""
    names = tuple(f"successor_{place}" for place in range(len(domains)))
    values = {name: PythonRule(repr(domain)) for name, domain in zip(names, domains)}
    return "tour", values, (PythonRule(f"circuit({', '.join(names)})"),)


def test_a_circuit_over_three_parameters_gives_the_two_cycles_through_them() -> None:
    every = (0, 1, 2)

    assert solve(tour(every, every, every), State(())) == (
        Action("tour", (("successor_0", 1), ("successor_1", 2), ("successor_2", 0))),
        Action("tour", (("successor_0", 2), ("successor_1", 0), ("successor_2", 1))),
    )


def test_a_circuit_refuses_the_loops_that_leave_a_position_out() -> None:
    """All-different alone would allow 0 to 1, 1 to 0, 2 to itself: three different values, two separate loops."""
    solutions = solve(tour((0, 1, 2), (0, 1, 2), (0, 1, 2)), State(()))

    assert all(solution.parameters[2][1] != 2 for solution in solutions)


def test_a_circuit_over_one_parameter_is_its_own_cycle() -> None:
    assert solve(tour((0,)), State(())) == (Action("tour", (("successor_0", 0),)),)


def test_a_circuit_takes_the_actions_parameters_and_nothing_else() -> None:
    action = ("tour", {"successor_0": PythonRule("(0,)")}, (PythonRule("circuit(successor_0, elsewhere)"),))

    with pytest.raises(ValueError, match="elsewhere"):
        solve(action, State(()))


def test_a_circuit_refuses_a_parameter_it_names_twice() -> None:
    values = {"successor_0": PythonRule("(0, 1)"), "successor_1": PythonRule("(0, 1)")}
    action = ("tour", values, (PythonRule("circuit(successor_0, successor_0)"),))

    with pytest.raises(ValueError, match="successor_0"):
        solve(action, State(()))


def test_a_circuit_refuses_a_value_that_names_no_position() -> None:
    with pytest.raises(ValueError, match="names no position"):
        solve(tour((1, 5), (0, 1)), State(()))


def test_a_computed_tour_is_checked_against_the_circuit() -> None:
    """What an optimizer gives is checked rather than searched for, so the predicate has to answer on its own."""
    name, values, constraints = tour((0, 1, 2), (0, 1, 2), (0, 1, 2))
    solver = create_solver()
    cycle = Action(name, (("successor_0", 1), ("successor_1", 2), ("successor_2", 0)))
    loops = Action(name, (("successor_0", 1), ("successor_1", 0), ("successor_2", 2)))

    assert (
        solver.allows(State(()), cycle, values, constraints),
        solver.allows(State(()), loops, values, constraints),
    ) == (True, False)
