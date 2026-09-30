import pytest

from openmind.csp.model.constraint_index import ConstraintIndex
from openmind.csp.model.search_space import SearchSpace
from openmind.csp.model.support_table import SupportTable
from openmind.csp.model.wipeout import Wipeout
from openmind.csp.repository.domain_repository import DomainRepository
from openmind.csp.service.arc_consistency import ArcConsistency
from openmind.structure.model.value import Value

SAME = frozenset({(1, 1), (2, 2), (3, 3)})


def propagate(
    domains: dict[str, set[Value]], tables: tuple[SupportTable, ...], changed: tuple[str, ...]
) -> DomainRepository:
    space = SearchSpace("set", tuple((name, tuple(values)) for name, values in domains.items()), tables, (), ())
    repository = DomainRepository(space.variables)
    ArcConsistency().propagate(repository, ConstraintIndex.of(space), changed)
    return repository


def test_values_without_a_partner_are_removed() -> None:
    repository = propagate({"a": {1, 2, 3}, "b": {1, 2}}, (SupportTable("a", "b", SAME),), ("b",))

    assert repository.values == {"a": {1, 2}, "b": {1, 2}}


def test_removals_spread_along_a_chain_of_tables() -> None:
    tables = (SupportTable("a", "b", SAME), SupportTable("b", "c", SAME))

    repository = propagate({"a": {1}, "b": {1, 2}, "c": {1, 2, 3}}, tables, ("a",))

    assert repository.values == {"a": {1}, "b": {1}, "c": {1}}


def test_nothing_reaches_the_trail_where_nothing_is_removed() -> None:
    domains = {"a": {1, 2}, "b": {1, 2}, "c": {1, 2}}

    repository = propagate(domains, (SupportTable("a", "b", SAME),), ("a", "b"))

    assert (repository.values, repository.height) == (domains, 0)


def test_an_emptied_domain_raises_a_wipeout() -> None:
    with pytest.raises(Wipeout) as raised:
        propagate({"a": {1}, "b": {2}}, (SupportTable("a", "b", SAME),), ("a",))

    assert raised.value.variable == "b"
