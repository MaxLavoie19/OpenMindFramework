import pytest

from openmind.csp.model.circuit_constraint import CircuitConstraint
from openmind.csp.model.constraint_index import ConstraintIndex
from openmind.csp.model.search_space import SearchSpace
from openmind.csp.model.wipeout import Wipeout
from openmind.csp.repository.circuit_chains import CircuitChains
from openmind.csp.repository.domain_repository import DomainRepository
from openmind.csp.service.circuit_propagator import CircuitPropagator
from openmind.structure.model.value import Value


def circuit(domains: dict[str, tuple[Value, ...]]) -> tuple[DomainRepository, CircuitChains, ConstraintIndex]:
    constraint = CircuitConstraint(tuple(domains))
    space = SearchSpace("set", tuple(domains.items()), (), (), (), (constraint,))
    return DomainRepository(space.variables), CircuitChains(len(domains)), ConstraintIndex.of(space)


def propagate(
    repository: DomainRepository, chains: CircuitChains, index: ConstraintIndex, changed: tuple[str, ...]
) -> None:
    CircuitPropagator().propagate(repository, index, 0, chains, changed)


def test_a_decided_successor_is_taken_from_every_other_position() -> None:
    repository, chains, index = circuit({"a": (1,), "b": (0, 2), "c": (0, 1)})

    propagate(repository, chains, index, ("a",))

    assert repository.values["c"] == {0}


def test_a_chains_own_start_is_refused_at_its_end_while_it_is_short() -> None:
    repository, chains, index = circuit({"a": (1,), "b": (0, 2), "c": (0, 1)})

    propagate(repository, chains, index, ("a",))

    assert repository.values["b"] == {2}, "0 would close a loop of two through a and b, leaving c out"


def test_the_last_step_of_a_full_chain_is_required_to_close_it() -> None:
    repository, chains, index = circuit({"a": (1,), "b": (2,), "c": (0, 1)})

    propagate(repository, chains, index, ("a", "b"))

    assert repository.values["c"] == {0}


def test_a_domain_narrowed_to_one_value_joins_its_chain() -> None:
    """Nothing ever assigns a variable propagation has already settled, so settling is what the propagator reads."""
    repository, chains, index = circuit({"a": (1, 2), "b": (0, 2), "c": (0, 1)})
    repository.fix("a", 1)

    propagate(repository, chains, index, ("a",))

    assert (chains.recorded(0), chains.length_of(0)) == (1, 2)


def test_running_again_over_an_unchanged_repository_joins_nothing_twice() -> None:
    repository, chains, index = circuit({"a": (1,), "b": (0, 2), "c": (0, 1)})
    propagate(repository, chains, index, ("a",))
    settled, joined = repository.height, chains.height

    propagate(repository, chains, index, ("a",))

    assert (repository.height, chains.height) == (settled, joined)


def test_undoing_puts_back_the_domains_and_the_chains() -> None:
    repository, chains, index = circuit({"a": (1,), "b": (0, 2), "c": (0, 1)})
    domains = {name: set(values) for name, values in repository.values.items()}

    propagate(repository, chains, index, ("a",))
    repository.undo_to(0)
    chains.undo_to(0)

    assert (repository.values, chains.recorded(0), chains.length_of(0)) == (domains, None, 1)


def test_a_cycle_that_cannot_close_raises_a_wipeout() -> None:
    repository, chains, index = circuit({"a": (1,), "b": (2,), "c": (2,)})

    with pytest.raises(Wipeout) as raised:
        propagate(repository, chains, index, ("a", "b"))

    assert raised.value.variable == "c"


def test_a_loop_shorter_than_the_whole_is_refused_by_taking_its_closing_value_away() -> None:
    """Which variable is named matters: it is the subtour removal that refuses this, not the closing check — and
    asserting only that *some* Wipeout was raised would pass whichever did it."""
    repository, chains, index = circuit({"a": (1,), "b": (0,), "c": (0, 1)})

    with pytest.raises(Wipeout) as raised:
        propagate(repository, chains, index, ("a", "b"))

    assert raised.value.variable == "b"


def test_the_closing_check_refuses_a_short_loop_it_is_handed_outright() -> None:
    """The removal above is what normally prevents this, so the guard is driven directly. It is what keeps the
    propagator right on its own rather than only in the order it happens to be called in."""
    repository, chains, index = circuit({"a": (1,), "b": (0, 2), "c": (0, 1)})
    chains.mark(0, 1)
    chains.join(0, 1)
    repository.fix("b", 0)

    with pytest.raises(Wipeout) as raised:
        propagate(repository, chains, index, ("b",))

    assert raised.value.variable == "b"


def test_completing_the_chain_takes_every_other_value_from_its_closing_position() -> None:
    repository, chains, index = circuit({"a": (1,), "b": (2,), "c": (0, 1, 2)})
    chains.mark(0, 1)
    chains.join(0, 1)

    propagate(repository, chains, index, ("b",))

    assert repository.values["c"] == {0}
