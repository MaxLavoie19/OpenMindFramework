from openmind.csp.model.all_different_group import AllDifferentGroup
from openmind.csp.model.circuit_constraint import CircuitConstraint
from openmind.csp.model.constraint_index import ConstraintIndex
from openmind.csp.model.scoped_constraint import ScopedConstraint
from openmind.csp.model.search_space import SearchSpace
from openmind.csp.model.support_table import SupportTable
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.rule.model.python_rule import PythonRule

PAIRS = frozenset({(0, 1), (1, 0)})


def scoped(source: str, scope: tuple[str, ...]) -> ScopedConstraint:
    return ScopedConstraint(create_rule_caller().prepare(PythonRule(source), scope), scope)


def indexed() -> ConstraintIndex:
    space = SearchSpace(
        "set",
        (("a", (0, 1)), ("b", (0, 1)), ("c", (0, 1, 2))),
        (SupportTable("a", "b", PAIRS),),
        (AllDifferentGroup(("a", "b")),),
        (scoped("a != b or c == 0", ("a", "b", "c")),),
        (CircuitConstraint(("a", "b", "c")),),
    )
    return ConstraintIndex.of(space)


def test_a_variable_finds_the_tables_it_is_in_and_no_others() -> None:
    index = indexed()

    assert (len(index.tables_of("a")), len(index.tables_of("b")), index.tables_of("c")) == (1, 1, ())


def test_a_variable_finds_the_groups_it_is_in_and_no_others() -> None:
    index = indexed()

    assert (index.groups_of("a"), index.groups_of("c")) == ((AllDifferentGroup(("a", "b")),), ())


def test_a_circuit_is_found_by_its_place_rather_than_by_its_value() -> None:
    """A circuit over a hundred thousand parameters would cost as much to compare as to propagate."""
    index = indexed()

    assert (index.circuit_places_of("a"), index.circuit_places_of("elsewhere")) == ((0,), ())


def test_what_several_changed_variables_touch_is_each_thing_once() -> None:
    index = indexed()

    assert (
        index.groups_touching(("a", "b")),
        index.circuit_places_touching(("a", "b", "c")),
        len(index.constraints_touching(("a", "c"))),
    ) == ((AllDifferentGroup(("a", "b")),), (0,), 1)


def test_a_variable_in_nothing_touches_nothing() -> None:
    index = indexed()

    assert (
        index.groups_touching(("elsewhere",)),
        index.circuit_places_touching(("elsewhere",)),
        index.constraints_touching(("elsewhere",)),
    ) == ((), (), ())


def test_a_value_finds_the_positions_whose_rule_offered_it() -> None:
    index = indexed()

    assert (index.holders(0, 2), index.holders(0, 0), index.holders(0, 9)) == (("c",), ("a", "b", "c"), ())


def test_a_circuits_variables_are_numbered_by_where_they_stand_in_it() -> None:
    assert indexed().positions(0) == {"a": 0, "b": 1, "c": 2}
