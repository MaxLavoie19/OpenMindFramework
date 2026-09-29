import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

#: Where a process's resident pages are counted, and what a page is worth.
STATM = "/proc/{pid}/statm"
STAT = "/proc/{pid}/stat"


class MemoryWatchdog:
    """What a whole run holds, across the process leading it and every worker it spawned, and whether that is
    more than the run was given.

    **Not the same job as `MemoryGuard`, which is why it is not the same thing.** The guard keeps *one*
    process under *its* cap, cutting caches back and, in a worker, ending that worker. It cannot see the run:
    six workers each comfortably inside a cap can still be twenty gigabytes between them, and the parent's own
    store growing has no cap at all. So a run can fill a machine without a single process ever being over.

    **Over budget it stops rather than cuts.** A cache cut is a repair and can be got wrong quietly; a run
    that has grown past what it was given has done something nobody predicted, and the useful thing is to have
    it stop while the evidence is still there — what each process held, and how far over the whole was. That
    is a thing to look at, not a thing to survive.

    **The budget is the caller's and there is no default.** How much a machine may give a run is a fact about
    the machine and whoever else is using it, and this has no way to know either."""

    def __init__(self, budget_bytes: int) -> None:
        if budget_bytes < 1:
            raise ValueError(f"A memory watchdog needs a budget of at least 1 byte, not {budget_bytes}")
        self._budget = budget_bytes

    @property
    def budget_bytes(self) -> int:
        return self._budget

    def held(self, pid: int | None = None) -> dict[int, int]:
        """The resident bytes of that process and every process it spawned, by process id.

        A process that ends while this is reading is left out rather than raising: a worker finishing is the
        ordinary case, and a watchdog that fell over whenever one did would be no watchdog."""
        root = os.getpid() if pid is None else pid
        found: dict[int, int] = {}
        for one in self._tree(root):
            resident = self._resident(one)
            if resident is not None:
                found[one] = resident
        return found

    def over(self, pid: int | None = None) -> bool:
        """Whether the run holds more than it was given."""
        return sum(self.held(pid).values()) > self._budget

    def report(self, held: dict[int, int]) -> str:
        """What was held, by process, largest first, for a log line somebody has to read at the other end."""
        whole = sum(held.values())
        lines = [
            f"The run holds {self._gigabytes(whole)} of the {self._gigabytes(self._budget)} it was given, "
            f"over {len(held)} processes",
            f"{'process':>10}  {'resident':>10}",
        ]
        lines += [f"{pid:>10}  {self._gigabytes(bytes_held):>10}" for pid, bytes_held in sorted(held.items(), key=lambda one: -one[1])]
        return "\n".join(lines)

    def _tree(self, root: int) -> list[int]:
        """That process and everything under it, however deep. Read off /proc rather than kept, because a run
        spawns and reaps workers the whole time and a list held here would be stale by the time it mattered."""
        children: dict[int, list[int]] = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            parent = self._parent(int(entry.name))
            if parent is not None:
                children.setdefault(parent, []).append(int(entry.name))
        found, pending = [], [root]
        while pending:
            one = pending.pop()
            if one in found:
                continue
            found.append(one)
            pending.extend(children.get(one, ()))
        return found

    def _parent(self, pid: int) -> int | None:
        """Whose child that process is. The name a process reports can hold spaces and brackets, so the fields
        are read from after the last bracket rather than by splitting the whole line."""
        try:
            line = Path(STAT.format(pid=pid)).read_text(encoding="ascii", errors="replace")
            return int(line[line.rindex(")") + 1 :].split()[1])
        except (OSError, ValueError, IndexError):
            return None

    def _resident(self, pid: int) -> int | None:
        try:
            statm = Path(STATM.format(pid=pid)).read_text(encoding="ascii")
            return int(statm.split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError, AttributeError):
            return None

    def _gigabytes(self, held: int) -> str:
        return f"{held / 1024 ** 3:.2f} GB"
