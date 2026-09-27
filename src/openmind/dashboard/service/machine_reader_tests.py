from pathlib import Path

from openmind.dashboard.constant.dashboard_constant import RUN, WORKER
from openmind.dashboard.service.incremental_line_reader import IncrementalLineReader
from openmind.dashboard.service.machine_reader import MachineReader


def test_a_run_that_says_which_process_is_its_own_is_shown_under_its_name(tmp_path: Path) -> None:
    """The reason this exists: three constraint learners and a heuristics finder ran for a day and a half while
    the panel said there was no training process, because none of their command lines held the one string it
    looked for."""
    proc = _machine(tmp_path, {1000: (0, "python learn_constraints.py --seed 1")})
    found = _reader().status(proc, None, {1000: "plain"})
    assert [(one.pid, one.role, one.run) for one in found.processes] == [(1000, RUN, "plain")]


def test_whatever_a_run_started_belongs_to_that_run(tmp_path: Path) -> None:
    """Found by walking parents rather than by a marker, so a run starting its workers any other way is not
    missed — and a grandchild is still the run's."""
    proc = _machine(
        tmp_path,
        {
            1000: (0, "python learn_constraints.py"),
            1001: (1000, "python -c from multiprocessing.spawn import spawn_main"),
            1002: (1001, "python something it started in turn"),
            2000: (0, "python nothing at all to do with it"),
        },
    )
    found = _reader().status(proc, None, {1000: "plain"})
    assert [(one.pid, one.role, one.run) for one in found.processes] == [
        (1000, RUN, "plain"),
        (1001, WORKER, "plain"),
        (1002, WORKER, "plain"),
    ], "the unrelated process is not shown, and the grandchild is"


def test_a_process_nobody_claimed_is_not_shown(tmp_path: Path) -> None:
    proc = _machine(tmp_path, {2000: (0, "python something else entirely")})
    assert _reader().status(proc, None, {}).processes == ()


def _reader() -> MachineReader:
    return MachineReader(IncrementalLineReader(), clock_ticks=100)


def _machine(where: Path, processes: dict[int, tuple[int, str]]) -> Path:
    """A process file system holding those processes, each as its parent and its command line."""
    proc = where / "proc"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "uptime").write_text("5000.0 40000.0\n")
    (proc / "meminfo").write_text("MemTotal: 65536 kB\nMemAvailable: 32768 kB\n")
    for pid, (parent, command) in processes.items():
        held = proc / str(pid)
        held.mkdir()
        (held / "cmdline").write_bytes(command.replace(" ", "\0").encode("utf-8"))
        # After the closing bracket: state, then the parent, then on to the twentieth, which is when it began.
        rest = " ".join(["0"] * 18)
        (held / "stat").write_text(f"{pid} (python) S {parent} {rest} 100000")
        (held / "status").write_text("VmRSS:\t   1024 kB\n")
    return proc
