from openmind.csp.repository.circuit_chains import CircuitChains


def test_every_position_starts_as_a_chain_of_its_own() -> None:
    chains = CircuitChains(3)

    assert (chains.start_of(1), chains.end_of(1), chains.length_of(1)) == (1, 1, 1)


def test_joining_runs_the_chain_from_one_start_to_the_others_end() -> None:
    chains = CircuitChains(3)

    assert chains.join(0, 1) == (0, 1, 2)


def test_joining_again_carries_the_whole_run_along() -> None:
    chains = CircuitChains(3)
    chains.join(0, 1)

    assert chains.join(1, 2) == (0, 2, 3)


def test_what_a_position_was_decided_to_be_followed_by_is_remembered() -> None:
    chains = CircuitChains(3)
    chains.mark(0, 1)

    assert (chains.recorded(0), chains.recorded(1)) == (1, None)


def test_undoing_puts_back_the_marks_and_the_runs() -> None:
    chains = CircuitChains(3)
    chains.mark(0, 1)
    chains.join(0, 1)

    chains.undo_to(0)

    assert (chains.recorded(0), chains.start_of(1), chains.end_of(0), chains.length_of(0)) == (None, 1, 0, 1)


def test_undoing_to_a_mark_in_the_middle_keeps_what_came_before_it() -> None:
    chains = CircuitChains(3)
    chains.mark(0, 1)
    chains.join(0, 1)
    mark = chains.height
    chains.mark(1, 2)
    chains.join(1, 2)

    chains.undo_to(mark)

    assert (chains.recorded(0), chains.recorded(1), chains.length_of(0)) == (1, None, 2)


def test_a_mark_made_twice_is_put_back_to_what_it_was_rather_than_to_nothing() -> None:
    """The propagator's own guard stops a position being marked twice, so the repository is driven directly: what
    it puts back has to be right whoever calls it."""
    chains = CircuitChains(3)
    chains.mark(0, 1)
    mark = chains.height
    chains.mark(0, 2)

    chains.undo_to(mark)

    assert chains.recorded(0) == 1
