import pytest

from openmind.csp.model.all_different_group import AllDifferentGroup
from openmind.csp.model.wipeout import Wipeout
from openmind.csp.repository.domain_repository import DomainRepository
from openmind.csp.service.all_different_propagator import AllDifferentPropagator
from openmind.structure.model.value import Value


def propagate(domains: dict[str, set[Value]]) -> DomainRepository:
    repository = DomainRepository((name, tuple(values)) for name, values in domains.items())
    AllDifferentPropagator().propagate(repository, AllDifferentGroup(tuple(domains)))
    return repository


def test_a_fixed_value_is_removed_from_the_others() -> None:
    assert propagate({"a": {1}, "b": {1, 2}}).values == {"a": {1}, "b": {2}}


def test_a_value_only_one_variable_can_take_goes_to_that_variable() -> None:
    assert propagate({"a": {1, 2}, "b": {1, 2}, "c": {1, 2, 3}}).values == {
        "a": {1, 2},
        "b": {1, 2},
        "c": {3},
    }


def test_values_taken_by_a_closed_set_of_variables_are_removed_from_the_others() -> None:
    assert propagate({"a": {1, 2}, "b": {1, 2}, "c": {1, 2, 3, 4}, "d": {1, 3, 4}}).values == {
        "a": {1, 2},
        "b": {1, 2},
        "c": {3, 4},
        "d": {3, 4},
    }


def test_nothing_reaches_the_trail_where_nothing_is_removed() -> None:
    domains = {"a": {1, 2, 3}, "b": {1, 2, 3}, "c": {1, 2, 3}}

    repository = propagate(domains)

    assert (repository.values, repository.height) == (domains, 0)


def test_more_variables_than_values_raises_a_wipeout() -> None:
    with pytest.raises(Wipeout):
        propagate({"a": {1, 2}, "b": {1, 2}, "c": {1, 2}})
