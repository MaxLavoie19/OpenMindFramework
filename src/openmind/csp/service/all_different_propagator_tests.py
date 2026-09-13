import pytest

from openmind.csp.model.all_different_group import AllDifferentGroup
from openmind.csp.model.wipeout import Wipeout
from openmind.csp.service.all_different_propagator import AllDifferentPropagator
from openmind.world.model.value import Value


def propagate(domains: dict[str, set[Value]]) -> dict[str, frozenset[Value]]:
    frozen = {name: frozenset(values) for name, values in domains.items()}
    return AllDifferentPropagator().propagate(frozen, AllDifferentGroup(tuple(domains)))


def test_a_fixed_value_is_removed_from_the_others() -> None:
    assert propagate({"a": {1}, "b": {1, 2}}) == {"a": frozenset({1}), "b": frozenset({2})}


def test_a_value_only_one_variable_can_take_goes_to_that_variable() -> None:
    assert propagate({"a": {1, 2}, "b": {1, 2}, "c": {1, 2, 3}}) == {
        "a": frozenset({1, 2}),
        "b": frozenset({1, 2}),
        "c": frozenset({3}),
    }


def test_values_taken_by_a_closed_set_of_variables_are_removed_from_the_others() -> None:
    assert propagate({"a": {1, 2}, "b": {1, 2}, "c": {1, 2, 3, 4}, "d": {1, 3, 4}}) == {
        "a": frozenset({1, 2}),
        "b": frozenset({1, 2}),
        "c": frozenset({3, 4}),
        "d": frozenset({3, 4}),
    }


def test_a_consistent_group_keeps_its_domains_as_they_are() -> None:
    domains = {"a": frozenset({1, 2, 3}), "b": frozenset({1, 2, 3}), "c": frozenset({1, 2, 3})}

    narrowed = AllDifferentPropagator().propagate(domains, AllDifferentGroup(("a", "b", "c")))

    assert all(narrowed[name] is domains[name] for name in domains)


def test_more_variables_than_values_raises_a_wipeout() -> None:
    with pytest.raises(Wipeout):
        propagate({"a": {1, 2}, "b": {1, 2}, "c": {1, 2}})
