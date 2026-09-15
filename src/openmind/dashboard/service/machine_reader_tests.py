from pathlib import Path

from openmind.dashboard.service.incremental_line_reader import IncrementalLineReader
from openmind.dashboard.service.machine_reader import MachineReader

MEMINFO = "MemTotal:       64136000 kB\nMemAvailable:   18354608 kB\nSwapTotal:       8388604 kB\nSwapFree:        1464412 kB\n"


def fake_process(proc: Path, pid: int, command: str, parent: int, started_ticks: int, rss_kb: int) -> None:
    directory = proc / str(pid)
    directory.mkdir()
    (directory / "cmdline").write_bytes(command.replace(" ", "\0").encode() + b"\0")
    fields = ["S", str(parent), *["0"] * 17, str(started_ticks)]
    (directory / "stat").write_text(f"{pid} (python) {' '.join(fields)}\n", encoding="utf-8")
    (directory / "status").write_text(f"Name:\tpython\nVmRSS:\t{rss_kb} kB\n", encoding="utf-8")


def test_the_machine_s_memory_the_training_s_processes_and_earlyoom_s_kills(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text(MEMINFO, encoding="utf-8")
    (proc / "uptime").write_text("1000.00 5000.00\n", encoding="utf-8")
    fake_process(proc, 10, "bash data/log/runs/continue_training_deduction.sh", 1, 10_000, 3_000)
    fake_process(proc, 11, ".venv/bin/python -m openmind.entrypoint.train_values chess --workers 14", 10, 50_000, 2_000_000)
    fake_process(proc, 12, "python -c from multiprocessing.spawn import spawn_main", 11, 60_000, 1_000_000)
    fake_process(proc, 13, "python -c from multiprocessing.spawn import spawn_main", 1, 60_000, 1_000_000)
    fake_process(proc, 14, "/usr/bin/streamlit run app.py", 1, 60_000, 500_000)
    syslog = tmp_path / "syslog"
    syslog.write_text(
        "2026-09-15T04:01:25 maxime-cinamon earlyoom[1518]: low memory! at or below SIGTERM limits\n"
        "2026-09-15T04:01:25 maxime-cinamon earlyoom[1518]: sending SIGTERM to process 9544 uid 1000 \"python\"\n"
        "2026-09-15T04:01:26 maxime-cinamon kernel: something else\n",
        encoding="utf-8",
    )

    status = MachineReader(IncrementalLineReader(), clock_ticks=100).status(proc, syslog)

    assert (status.memory_total, status.memory_available) == (64136000 * 1024, 18354608 * 1024)
    assert (status.swap_total, status.swap_free) == (8388604 * 1024, 1464412 * 1024)
    assert [(process.pid, process.role, process.rss_bytes, process.seconds) for process in status.processes] == [
        (10, "loop", 3_000 * 1024, 900.0),
        (11, "training", 2_000_000 * 1024, 500.0),
        (12, "worker", 1_000_000 * 1024, 400.0),
    ]
    assert status.earlyoom == (
        '2026-09-15T04:01:25 maxime-cinamon earlyoom[1518]: sending SIGTERM to process 9544 uid 1000 "python"',
    )
