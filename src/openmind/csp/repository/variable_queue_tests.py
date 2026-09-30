from openmind.csp.repository.domain_repository import DomainRepository
from openmind.csp.repository.variable_queue import VariableQueue
from openmind.structure.model.value import Value


def queued(
    domains: dict[str, tuple[Value, ...]], degree: dict[str, int] | None = None
) -> tuple[VariableQueue, DomainRepository]:
    repository = DomainRepository(domains.items())
    queue = VariableQueue(
        degree or dict.fromkeys(domains, 0), {name: place for place, name in enumerate(domains)}
    )
    for name, values in domains.items():
        queue.push(name, len(values))
    return queue, repository


def test_the_variable_with_the_fewest_values_is_picked() -> None:
    queue, repository = queued({"a": (1, 2, 3), "b": (1, 2)})

    assert queue.pick(repository) == "b"


def test_among_equal_domains_the_one_in_the_most_constraints_is_picked() -> None:
    queue, repository = queued({"a": (1, 2), "b": (1, 2)}, degree={"a": 0, "b": 1})

    assert queue.pick(repository) == "b"


def test_among_equal_domains_and_degrees_the_one_declared_first_is_picked() -> None:
    queue, repository = queued({"a": (1, 2), "b": (1, 2)})

    assert queue.pick(repository) == "a"


def test_a_settled_variable_is_not_picked() -> None:
    queue, repository = queued({"a": (1,), "b": (1, 2)})

    assert queue.pick(repository) == "b"


def test_nothing_is_picked_once_every_variable_is_settled() -> None:
    queue, repository = queued({"a": (1,), "b": (2,)})

    assert queue.pick(repository) is None


def test_an_entry_whose_domain_has_shrunk_since_is_dropped() -> None:
    """A domain is narrowed in place, so entries go stale rather than being found and fixed."""
    queue, repository = queued({"a": (1, 2, 3), "b": (1, 2, 3, 4)})
    repository.remove("b", 1)
    repository.remove("b", 2)
    queue.push("b", len(repository.values["b"]))

    assert queue.pick(repository) == "b"


def test_a_variable_that_grew_back_is_picked_again_once_it_is_offered() -> None:
    queue, repository = queued({"a": (1, 2)})
    repository.fix("a", 1)
    assert queue.pick(repository) is None

    repository.undo_to(0)
    queue.push("a", len(repository.values["a"]))

    assert queue.pick(repository) == "a"
