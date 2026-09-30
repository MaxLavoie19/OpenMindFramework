import pytest

from openmind.csp.model.wipeout import Wipeout
from openmind.csp.repository.domain_repository import DomainRepository


def repository() -> DomainRepository:
    return DomainRepository((("a", (1, 2, 3)), ("b", (1, 2))))


def test_a_removal_is_recorded_and_put_back_exactly() -> None:
    held = repository()
    mark = held.height

    held.remove("a", 2)
    held.undo_to(mark)

    assert (held.values, held.height) == ({"a": {1, 2, 3}, "b": {1, 2}}, mark)


def test_removing_a_value_that_is_not_there_records_nothing() -> None:
    held = repository()

    assert (held.remove("a", 9), held.height) == (False, 0)


def test_fixing_takes_away_every_other_value() -> None:
    held = repository()

    held.fix("a", 2)

    assert (held.values["a"], held.settled("a"), held.only("a")) == ({2}, True, 2)


def test_emptying_a_domain_raises_a_wipeout() -> None:
    held = repository()
    held.remove("b", 1)

    with pytest.raises(Wipeout) as raised:
        held.remove("b", 2)

    assert raised.value.variable == "b"


def test_undoing_names_the_variables_that_grew_back_each_once() -> None:
    held = repository()
    held.remove("a", 1)
    held.remove("a", 2)
    held.remove("b", 1)

    assert sorted(held.undo_to(0)) == ["a", "b"]


def test_what_changed_is_read_from_the_mark_rather_than_from_every_variable() -> None:
    held = repository()
    held.remove("a", 1)
    mark = held.height
    held.remove("b", 1)

    assert held.changed_since(mark) == ("b",)


def test_undoing_to_where_it_already_stands_changes_nothing() -> None:
    held = repository()
    held.remove("a", 1)

    assert (held.undo_to(held.height), held.values["a"]) == ((), {2, 3})
