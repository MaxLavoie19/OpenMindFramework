import logging
import os
from collections.abc import Mapping, Sequence
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

    **Over budget it stops rather than cuts, and what it leaves behind is the point.** A cache cut is a
    repair and can be got wrong quietly; a run that has grown past what it was given has done something nobody
    predicted, and the useful thing is that it stops while the evidence is still there. So this is not an
    error to survive — it is the beginning of an investigation, and the report has to be worth reading at the
    other end of a night: which process held what, what the run itself says it was holding, and where the
    directories it writes to had got to.

    **What it holds is asked of the caller rather than worked out here.** Bytes name a process and nothing
    else; what makes a stop investigable is the run's own count of what it had grown — how many candidates,
    how big a store — and nothing generic can know what those are.

    **The disk is watched beside the memory, because the disk is what has actually killed this run.** Twice,
    at `OSError: [Errno 28] No space left on device`, half way through writing a rule into a knowledge base —
    which leaves a store torn rather than a report written.

    **What is watched is what the run has written, not what the machine has left.** Free space is a fact about
    the machine and about everything else on it: the drive this run died on has 463 GB free, so a watchdog on
    free space would have sat silent through the whole failure and only fired at the very end, by which time
    the store is torn anyway. What a run has *spent* is a fact about the run — ten gigabytes written is
    unexpected whether the disk is large or small, and it is unexpected long before anything runs out.

    **The budgets are the caller's and there are no defaults.** How much a machine may give a run is a fact
    about the machine and whoever else is using it, and this has no way to know either."""

    def __init__(self, budget_bytes: int = 0, disk_bytes: int = 0, writing_to: Sequence[Path] = ()) -> None:
        if budget_bytes < 0 or disk_bytes < 0:
            raise ValueError("A watchdog's budgets cannot be negative")
        if not budget_bytes and not disk_bytes:
            raise ValueError("A watchdog with no budget at all watches nothing")
        self._budget = budget_bytes
        #: How much the run may have written where it writes before it is stopped, and the directories that
        #: counts. Nought watches no disk at all.
        self._disk = disk_bytes
        self._writing_to = tuple(writing_to)

    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def disk_bytes(self) -> int:
        return self._disk

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
        """Whether the run holds more than it was given, or has written more than it was given."""
        if self._budget and sum(self.held(pid).values()) > self._budget:
            return True
        return bool(self._disk) and sum(self.written().values()) > self._disk

    def written(self) -> dict[Path, int]:
        """How many bytes the run has written under each directory it writes to.

        **A stop here is worth more than the write it prevents.** Running out mid-write leaves a store torn
        and a traceback where a report should be; asked beforehand, the same condition is an orderly stop with
        everything still readable.

        A directory that has gone or cannot be read counts as nothing rather than raising — a temporary
        directory being cleaned up underneath this is the ordinary case."""
        return {one: self._under(one) for one in self._writing_to} if self._disk else {}

    def _under(self, where: Path) -> int:
        """What that directory holds, however deep, following no symbolic links."""
        found = 0
        pending = [where]
        while pending:
            try:
                with os.scandir(pending.pop()) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                pending.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                found += entry.stat(follow_symlinks=False).st_size
                        except OSError:
                            continue
            except OSError:
                continue
        return found

    def report(self, held: dict[int, int], holding: Mapping[str, object] = {}) -> str:
        """Everything an investigation starts from: how far over the run was, which process held what, how much
        space is left where it writes, and whatever the run itself says it had grown.

        `holding` is the run's own account — how many candidates, how many rules, how big a store. Bytes name a
        process and nothing else, and "the parent held nine gigabytes" is not something anybody can act on;
        "the parent held nine gigabytes and the pool had reached two hundred heuristics" is."""
        lines: list[str] = []
        if self._budget:
            lines += [
                f"The run holds {self._gigabytes(sum(held.values()))} of the "
                f"{self._gigabytes(self._budget)} it was given, over {len(held)} processes",
                f"{'process':>10}  {'resident':>10}",
            ]
            lines += [
                f"{pid:>10}  {self._gigabytes(bytes_held):>10}"
                for pid, bytes_held in sorted(held.items(), key=lambda one: -one[1])
            ]
        written = self.written()
        if written:
            lines += [
                "",
                f"The run has written {self._gigabytes(sum(written.values()))} of the "
                f"{self._gigabytes(self._disk)} it was given",
                f"{'written':>12}  where it writes",
            ]
            lines += [
                f"{self._gigabytes(bytes_written):>12}  {one}"
                for one, bytes_written in sorted(written.items(), key=lambda one: -one[1])
            ]
        if holding:
            lines += ["", f"{'what the run had grown':>34}  count"]
            lines += [f"{name:>34}  {value}" for name, value in holding.items()]
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
