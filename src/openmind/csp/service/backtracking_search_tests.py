import logging
from itertools import permutations

import pytest

from openmind.csp.model.all_different_group import AllDifferentGroup
from openmind.csp.model.scoped_constraint import ScopedConstraint
from openmind.csp.model.search_space import SearchSpace
from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.csp.model.support_table import SupportTable
from openmind.csp.service.all_different_propagator import AllDifferentPropagator
from openmind.csp.service.arc_consistency import ArcConsistency
from openmind.csp.service.backtracking_search import BacktrackingSearch
from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.state import State
from openmind.structure.model.value import Value


def search(
    variables: dict[str, tuple[Value, ...]],
    tables: tuple[SupportTable, ...] = (),
    groups: tuple[AllDifferentGroup, ...] = (),
    constraints: tuple[ScopedConstraint, ...] = (),
    limit: int | None = None,
) -> tuple[tuple[dict[str, Value], ...], SolveStatistics]:
    space = SearchSpace(
        "set",
        tuple((name, values) for name, values in variables.items()),
        tables,
        groups,
        constraints,
    )
    backtracking = BacktrackingSearch(
        ArcConsistency(),
        AllDifferentPropagator(),
        ConstraintChecker(create_rule_caller()),
    )
    return backtracking.search(space, State(()), limit)


def first_try(caplog: pytest.LogCaptureFixture) -> str:
    return next(message for message in caplog.messages if ": try " in message)


def test_finds_every_solution_of_an_all_different_group() -> None:
    solutions, statistics = search(
        {"a": (1, 2, 3), "b": (1, 2, 3), "c": (1, 2, 3)}, groups=(AllDifferentGroup(("a", "b", "c")),)
    )

    assert sorted((solution["a"], solution["b"], solution["c"]) for solution in solutions) == sorted(
        permutations((1, 2, 3))
    )
    assert statistics.solutions == 6


def test_a_table_keeps_only_allowed_pairs() -> None:
    solutions, _ = search({"a": (1, 2), "b": (1, 2)}, tables=(SupportTable("a", "b", frozenset({(1, 2), (2, 1)})),))

    assert sorted((solution["a"], solution["b"]) for solution in solutions) == [(1, 2), (2, 1)]


def test_the_limit_stops_the_search() -> None:
    solutions, _ = search(
        {"a": (1, 2, 3), "b": (1, 2, 3), "c": (1, 2, 3)}, groups=(AllDifferentGroup(("a", "b", "c")),), limit=2
    )

    assert len(solutions) == 2


def test_no_variables_gives_one_empty_solution() -> None:
    assert search({})[0] == ({},)


def test_an_empty_domain_gives_no_solution() -> None:
    assert search({"a": ()})[0] == ()


def test_a_larger_constraint_leads_to_dead_ends(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="openmind.csp.service.backtracking_search")
    never = create_rule_caller().prepare(PythonRule("a == 3 or b == 3 or c == 3"), ("a", "b", "c"))

    solutions, statistics = search(
        {"a": (1, 2), "b": (1, 2), "c": (1, 2)}, constraints=(ScopedConstraint(never, ("a", "b", "c")),)
    )

    assert solutions == ()
    assert statistics.dead_ends == 4
    assert "set: dead end at b = 1, c has no value left" in caplog.messages


def test_the_variable_with_the_fewest_values_goes_first(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="openmind.csp.service.backtracking_search")

    search({"a": (1, 2, 3), "b": (1, 2)}, groups=(AllDifferentGroup(("a", "b")),))

    assert first_try(caplog) == "set: try b = 1"


def test_among_equal_domains_the_most_constrained_variable_goes_first(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="openmind.csp.service.backtracking_search")

    search({"a": (1, 2), "b": (1, 2), "c": (1, 2)}, groups=(AllDifferentGroup(("b", "c")),))

    assert first_try(caplog) == "set: try b = 1"


def test_with_a_limit_the_least_constraining_value_goes_first(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="openmind.csp.service.backtracking_search")
    variables = {"x": (2, 1), "y": (2, 3, 4)}
    groups = (AllDifferentGroup(("x", "y")),)

    search(variables, groups=groups, limit=1)
    with_limit = first_try(caplog)
    caplog.clear()
    search(variables, groups=groups)

    assert (with_limit, first_try(caplog)) == ("set: try x = 1", "set: try x = 2")


def test_statistics_count_assignments_and_pruned_values() -> None:
    _, statistics = search({"a": (1,), "b": (1, 2), "c": (1, 2, 3)}, groups=(AllDifferentGroup(("a", "b", "c")),))

    assert statistics == SolveStatistics(solutions=1, assignments=0, dead_ends=0, pruned_values=3)
