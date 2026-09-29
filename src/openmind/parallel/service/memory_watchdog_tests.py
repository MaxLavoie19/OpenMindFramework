import os
import subprocess
import sys

import pytest

from openmind.parallel.factory.memory_watchdog_factory import create_memory_watchdog

pytestmark = pytest.mark.log_level("INFO")


def test_a_budget_of_nothing_is_refused() -> None:
    """A watchdog that permits nothing stops the run it is watching before it starts, which is a misconfigured
    run pretending to be a finding."""
    with pytest.raises(ValueError):
        create_memory_watchdog(0)


def test_it_counts_this_process(tmp_path) -> None:
    held = create_memory_watchdog(1024**4).held()

    assert os.getpid() in held
    assert held[os.getpid()] > 0


def test_it_counts_the_workers_a_run_spawned_and_not_only_the_one_leading_it() -> None:
    """The whole reason this is not `MemoryGuard`. Six workers each comfortably inside a per-process cap can
    still be twenty gigabytes between them, so a watchdog that read only the leader would report a run as
    small while the machine filled."""
    held = create_memory_watchdog(1024**4).held()
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        with_child = create_memory_watchdog(1024**4).held()

        assert child.pid in with_child, "a process this run spawned is this run's memory"
        assert sum(with_child.values()) > sum(held.values())
    finally:
        child.kill()
        child.wait()


def test_a_run_inside_its_budget_is_not_over() -> None:
    assert not create_memory_watchdog(1024**4).over()


def test_a_run_past_its_budget_is_over() -> None:
    """One byte is less than any process holds, so this is the shape of a run that has grown past what the
    machine gave it — the case the watchdog exists for and the one hardest to arrange honestly."""
    assert create_memory_watchdog(1).over()


def test_what_it_reports_says_which_process_held_what() -> None:
    """Somebody reads this at the other end of a night, and "the run used too much" is not a thing anybody can
    act on. Which process held what is."""
    watchdog = create_memory_watchdog(1024**3)

    report = watchdog.report(watchdog.held())

    assert "process" in report and "resident" in report, "columns get a header"
    assert str(os.getpid()) in report
    assert "GB" in report


def test_a_process_that_ends_while_it_is_being_read_is_left_out() -> None:
    """A worker finishing is the ordinary case, not an error. A watchdog that fell over whenever one did would
    be no watchdog at all."""
    watchdog = create_memory_watchdog(1024**4)

    assert watchdog._resident(2**30) is None  # noqa: SLF001
    assert watchdog._parent(2**30) is None  # noqa: SLF001
