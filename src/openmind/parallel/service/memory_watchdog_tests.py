import os
import subprocess
import sys

import pytest

from openmind.parallel.factory.memory_watchdog_factory import create_memory_watchdog

pytestmark = pytest.mark.log_level("INFO")


def test_a_watchdog_with_no_budget_at_all_is_refused() -> None:
    """Watching nothing is a misconfigured run pretending to be a safe one, and it should say so at the start
    rather than sit silent through a night."""
    with pytest.raises(ValueError):
        create_memory_watchdog(0, 0)


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


def test_what_is_watched_is_what_the_run_wrote_and_not_what_the_disk_has_left(tmp_path) -> None:
    """**Free space would have sat silent through the failure this exists for.** The drive the run died on has
    463 GB free, so nothing about free space was ever going to fire; what was unexpected was that the run had
    written tens of gigabytes of its own. Ten gigabytes written is unexpected whether the disk is large or
    small, and it is unexpected long before anything runs out."""
    (tmp_path / "a store").mkdir()
    (tmp_path / "a store" / "rules.jsonl").write_bytes(b"x" * 20_000)

    watchdog = create_memory_watchdog(disk_bytes=10_000, writing_to=(tmp_path,))

    assert watchdog.written()[tmp_path] == 20_000, "what it wrote, however deep"
    assert watchdog.over(), "and twice what it was given"


def test_a_run_that_has_written_little_is_not_over(tmp_path) -> None:
    (tmp_path / "small").write_bytes(b"x" * 10)

    assert not create_memory_watchdog(disk_bytes=1024**3, writing_to=(tmp_path,)).over()


def test_watching_no_disk_at_all_watches_no_disk(tmp_path) -> None:
    """Nought is off, and off has to mean a run that never asked about the disk is never stopped for it."""
    (tmp_path / "big").write_bytes(b"x" * 100_000)

    assert create_memory_watchdog(1024**4, disk_bytes=0, writing_to=(tmp_path,)).written() == {}


def test_a_directory_that_has_gone_counts_as_nothing_rather_than_raising(tmp_path) -> None:
    """A worker's temporary store being cleaned up underneath this is the ordinary case, and a watchdog that
    fell over whenever one was would be no watchdog."""
    gone = tmp_path / "never made"

    assert create_memory_watchdog(disk_bytes=1024**3, writing_to=(gone,)).written() == {gone: 0}


def test_the_report_says_what_the_run_had_grown(tmp_path) -> None:
    """A watchdog firing is the start of an investigation, and "it used too much" is not something anybody can
    act on. Which process held what explains where; what the run had grown explains why."""
    (tmp_path / "a store").write_bytes(b"x" * 5_000)
    watchdog = create_memory_watchdog(1024**3, disk_bytes=1024**3, writing_to=(tmp_path,))

    report = watchdog.report(watchdog.held(), {"heuristics in the pool": 214, "rules across them": 1712})

    assert "heuristics in the pool" in report and "214" in report
    assert "rules across them" in report and "1712" in report
    assert "written" in report and str(tmp_path) in report
    assert "count" in report, "columns get a header"


def test_a_report_with_nothing_to_add_is_still_a_report() -> None:
    """A run that cannot say what it had grown still says what each process held, rather than saying nothing."""
    watchdog = create_memory_watchdog(1024**3)

    assert "resident" in watchdog.report(watchdog.held())


def test_a_run_watching_only_its_disk_reports_only_its_disk(tmp_path) -> None:
    """Memory is not what killed this run and a run may say so. Reading process sizes it never asked about
    would bury the one number that matters under six that do not."""
    (tmp_path / "a store").write_bytes(b"x" * 5_000)

    report = create_memory_watchdog(disk_bytes=1024**3, writing_to=(tmp_path,)).report({})

    assert "resident" not in report
    assert "written" in report
