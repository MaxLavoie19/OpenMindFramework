import os
from collections import deque
from pathlib import Path

from openmind.dashboard.constant.dashboard_constant import (
    EARLYOOM_LINES,
    LOOP,
    LOOP_SCRIPT,
    TRAINING,
    TRAINING_MODULE,
    WORKER,
    WORKER_MARK,
)
from openmind.dashboard.model.machine_status import MachineStatus
from openmind.dashboard.model.process_status import ProcessStatus
from openmind.dashboard.service.incremental_line_reader import IncrementalLineReader


class MachineReader:
    """Reads the machine from the process file system: memory and swap from `meminfo`, the training's processes from
    every process's command line (the loop script, the training entrypoint, and the workers a training started), with
    the memory each holds and how long it has run; and earlyoom's latest lines about signalling a process, from the
    system log, read a piece at a time."""

    def __init__(self, line_reader: IncrementalLineReader, clock_ticks: int = os.sysconf("SC_CLK_TCK")) -> None:
        self._line_reader = line_reader
        self._clock_ticks = clock_ticks
        self._earlyoom: deque[str] = deque(maxlen=EARLYOOM_LINES)

    def status(self, proc: Path, syslog: Path | None) -> MachineStatus:
        memory = self._meminfo(proc / "meminfo")
        if syslog is not None:
            self._earlyoom.extend(
                line for line in self._line_reader.new_lines(syslog) if "earlyoom" in line and "sending SIG" in line
            )
        return MachineStatus(
            memory.get("MemTotal", 0),
            memory.get("MemAvailable", 0),
            memory.get("SwapTotal", 0),
            memory.get("SwapFree", 0),
            self._processes(proc),
            tuple(self._earlyoom),
        )

    def _meminfo(self, path: Path) -> dict[str, int]:
        values: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            name, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[name] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
        return values

    def _processes(self, proc: Path) -> tuple[ProcessStatus, ...]:
        uptime = float((proc / "uptime").read_text(encoding="utf-8").split()[0])
        found: dict[int, tuple[str, int, int, float]] = {}
        for directory in proc.iterdir():
            if not directory.name.isdigit():
                continue
            try:
                command = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
                stat = (directory / "stat").read_text(encoding="utf-8")
                status = (directory / "status").read_text(encoding="utf-8")
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            fields = stat[stat.rfind(")") + 2 :].split()
            parent, started = int(fields[1]), int(fields[19]) / self._clock_ticks
            rss = next((int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith("VmRSS:")), 0)
            found[int(directory.name)] = (command, parent, rss, uptime - started)
        trainings = {pid for pid, (command, _, _, _) in found.items() if TRAINING_MODULE in command}
        processes: list[ProcessStatus] = []
        for pid, (command, parent, rss, seconds) in sorted(found.items()):
            if LOOP_SCRIPT in command and command.startswith("bash"):
                role = LOOP
            elif pid in trainings:
                role = TRAINING
            elif WORKER_MARK in command and parent in trainings:
                role = WORKER
            else:
                continue
            processes.append(ProcessStatus(pid, role, rss, seconds, command))
        order = {LOOP: 0, TRAINING: 1, WORKER: 2}
        return tuple(sorted(processes, key=lambda process: (order[process.role], process.pid)))
