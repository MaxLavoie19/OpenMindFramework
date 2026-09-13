import pytest

from openmind.csp.model.support_table import SupportTable
from openmind.csp.model.wipeout import Wipeout
from openmind.csp.service.arc_consistency import ArcConsistency

SAME = frozenset({(1, 1), (2, 2), (3, 3)})


def test_values_without_a_partner_are_removed() -> None:
    domains = {"a": frozenset({1, 2, 3}), "b": frozenset({1, 2})}

    narrowed = ArcConsistency().propagate(domains, (SupportTable("a", "b", SAME),), ("b",))

    assert narrowed == {"a": frozenset({1, 2}), "b": frozenset({1, 2})}


def test_removals_spread_along_a_chain_of_tables() -> None:
    tables = (SupportTable("a", "b", SAME), SupportTable("b", "c", SAME))
    domains = {"a": frozenset({1}), "b": frozenset({1, 2}), "c": frozenset({1, 2, 3})}

    narrowed = ArcConsistency().propagate(domains, tables, ("a",))

    assert narrowed == {"a": frozenset({1}), "b": frozenset({1}), "c": frozenset({1})}


def test_unchanged_domains_are_kept_as_they_are() -> None:
    domains = {"a": frozenset({1, 2}), "b": frozenset({1, 2}), "c": frozenset({1, 2})}

    narrowed = ArcConsistency().propagate(domains, (SupportTable("a", "b", SAME),), ("a", "b"))

    assert all(narrowed[name] is domains[name] for name in domains)


def test_an_emptied_domain_raises_a_wipeout() -> None:
    domains = {"a": frozenset({1}), "b": frozenset({2})}

    with pytest.raises(Wipeout) as raised:
        ArcConsistency().propagate(domains, (SupportTable("a", "b", SAME),), ("a",))

    assert raised.value.variable == "b"
