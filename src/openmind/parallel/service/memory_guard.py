import ctypes
import gc
import logging
import os
import pickle
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from openmind.parallel.constant.parallel_constant import (
    DEFAULT_PROCESS_MEMORY,
    DIAGNOSIS_CALLS,
    MEMORY_CHECK_INTERVAL,
    MEMORY_EXIT_CODE,
    MEMORY_LOG_SECONDS,
    MEMORY_REGROWTH_SHARE,
    MEMORY_SETTLE_SHARE,
    MEMORY_WATCH_SECONDS,
)
from openmind.parallel.model.clearable import Clearable
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.service.memory_meter import MemoryMeter

logger = logging.getLogger(__name__)

_libc: ctypes.CDLL | None = None


class MemoryGuard:
    """Keeps one process's memory down by cutting the caches registered with it back when it goes over its limit. A
    cache calls `remembered` before it keeps an entry: every MEMORY_CHECK_INTERVAL entries the process's memory is read,
    and over the limit every cache drops its oldest entries, keeping MEMORY_SETTLE_SHARE of what it held. Caches are
    held weakly, so a service that is no longer used goes away with its cache.

    Cutting back rather than emptying, because emptying doesn't do what it looks like it does: a process that has
    churned through millions of small objects keeps most of what it frees, so emptying every cache moves the measured
    memory very little while the working set a search keeps coming back to is gone — and is derived again, filled again,
    and emptied again. Dropping a share of the oldest leaves that working set in place, and another cut comes only if the
    memory keeps climbing: a cut that didn't bring it down would not be helped by another.

    A share of the entries rather than a size in bytes, since what an entry costs can't be read off a process: most of
    what it holds is no cache of its own — a search's columns, and everything it has freed without giving back — and
    charging that to the entries cuts the caches many times deeper than the memory calls for.

    In a worker under a memory cap, `watch` also reads the memory every MEMORY_WATCH_SECONDS from a thread of its own.
    Over the cap, it asks for a clear at the next entry — the last resort, giving back whatever can be given — since the
    thread can't safely empty a cache another thread is using; still over the cap after the cap's grace, it writes a
    diagnosis, reports it, and ends the worker with MEMORY_EXIT_CODE."""

    def __init__(self, memory_meter: MemoryMeter, clock: Callable[[], float] = time.monotonic) -> None:
        self._memory_meter = memory_meter
        self._clock = clock
        self._limit = DEFAULT_PROCESS_MEMORY
        self._caches: weakref.WeakSet[Clearable] = weakref.WeakSet()
        self._entries = 0
        self._requested = False
        self._clears = 0
        self._logged_at: float | None = None
        self._cleared: tuple[datetime, int, int, list[tuple[str, int, int]]] | None = None
        self._call: tuple[int, Callable[..., object], Sequence[object]] | None = None
        self._calls: deque[tuple[int, str, int]] = deque(maxlen=DIAGNOSIS_CALLS)
        self._floor: int | None = None
        self._budget: int | None = None
        self._evicted = 0
        self._cut_at: int | None = None

    @property
    def limit_bytes(self) -> int:
        return self._limit

    @property
    def entry_budget(self) -> int | None:
        """How many entries the caches may hold between them, as last worked out, or None before the first reading."""
        return self._budget

    def limit(self, memory_bytes: int) -> None:
        """How many bytes the process holds before its caches are cut back."""
        self._limit = memory_bytes

    def register(self, cache: Clearable) -> None:
        self._caches.add(cache)

    def remembered(self) -> None:
        """Called by a registered cache before it keeps an entry."""
        self._entries += 1
        if self._requested:
            self.clear()
        elif self._entries % MEMORY_CHECK_INTERVAL == 0:
            self.settle()

    def settle(self) -> None:
        """Cuts the caches back when the process is over its limit, dropping their oldest entries.

        Emptying them instead, as this used to do, does not give the memory back: a process that has churned through
        millions of small objects keeps most of what it frees, so the measured memory barely moves while the working set
        a search keeps coming back to is gone — and it is derived again, filled again, and emptied again.

        So the trigger is unchanged, the process being over its limit, and only what follows is different: every cache
        keeps MEMORY_SETTLE_SHARE of what it holds and drops its oldest.

        And it cuts only while the memory is still climbing. A process whose own memory already sits above its limit —
        what it freed and kept, not what it caches — stays over whatever the caches do, and cutting at every reading
        would take them down to nothing, the old emptying in slow motion. So once a cut has been made, the next comes
        only when the memory has grown past where it stood at that cut, by more than MEMORY_REGROWTH_SHARE of the limit;
        a process back under its limit starts afresh. A worker that really runs away is still ended by its cap's watch.
        The floor — what the process holds with nothing cached — is kept as it is read, for the memory diagnosis a
        worker writes when it is ended."""
        used = self._memory_meter.resident_bytes()
        held = self.held_entries()
        if held == 0:
            self._floor = used if self._floor is None else min(self._floor, used)
            return
        self._floor = used if self._floor is None else min(self._floor, used)
        if used <= self._limit:
            self._cut_at = None
            return
        if self._cut_at is not None and used <= self._cut_at + self._limit * MEMORY_REGROWTH_SHARE:
            return
        self._budget = budget = int(held * MEMORY_SETTLE_SHARE)
        for cache in list(self._caches):
            entries = cache.memory_entries()
            if entries:
                cache.evict_memory(int(budget * entries / held))
        self._cut_at = used
        self._evicted += held - self.held_entries()
        self._log_settled(held, used, budget)

    def _log_settled(self, held: int, used: int, budget: int) -> None:
        now = self._clock()
        if self._logged_at is None or now - self._logged_at >= MEMORY_LOG_SECONDS:
            logger.info(
                "Cut the caches back to %d entries of %d: this process held %d bytes, over its limit of %d, and now "
                "holds %d; %d entries dropped since the previous line",
                self.held_entries(),
                held,
                used,
                self._limit,
                self._memory_meter.resident_bytes(),
                self._evicted,
            )
            self._logged_at, self._evicted = now, 0

    def held_entries(self) -> int:
        """How many entries the registered caches hold between them."""
        return sum(cache.memory_entries() for cache in list(self._caches))

    def clear(self) -> None:
        """Empties every registered cache and hands freed memory back to the system. This is the last resort — a worker
        told to give back whatever it can — not the everyday way of staying within the budget, which is `settle`."""
        requested, self._requested = self._requested, False
        if self.held_entries() == 0:
            _trim()
            return
        before = self._memory_meter.resident_bytes()
        kinds: dict[str, list[int]] = {}
        for cache in list(self._caches):
            kind = kinds.setdefault(type(cache).__qualname__, [0, 0])
            kind[0] += 1
            kind[1] += cache.memory_entries()
            cache.clear_memory()
        _trim()
        after = self._memory_meter.resident_bytes()
        caches = sorted(((name, count, entries) for name, (count, entries) in kinds.items()), key=lambda item: -item[2])
        self._cleared = (datetime.now(), before, after, caches)
        self._clears += 1
        now = self._clock()
        if requested or self._logged_at is None or now - self._logged_at >= MEMORY_LOG_SECONDS:
            logger.info(
                "Cleared %d caches holding %d entries %s: this process held %d bytes, its limit being %d, and now holds "
                "%d; %d clears since the previous line",
                sum(count for _, count, _ in caches),
                sum(entries for _, _, entries in caches),
                "as the memory watch asked" if requested else "on reading the memory",
                before,
                self._limit,
                after,
                self._clears,
            )
            self._logged_at, self._clears = now, 0

    def release(self) -> None:
        """Collects garbage and hands freed memory back to the system, as a worker does after each call."""
        gc.collect()
        _trim()

    def calling(self, index: int, function: Callable[..., object], arguments: Sequence[object]) -> None:
        """The call a worker starts, named in a diagnosis and pickled with it."""
        self._call = (index, function, arguments)

    def called(self) -> None:
        """The call a worker finished, listed in a diagnosis with the memory held after it."""
        if self._call is not None:
            self._calls.append((self._call[0], _name(self._call[1]), self._memory_meter.resident_bytes()))
        self._call = None

    def watch(
        self,
        cap: MemoryCap,
        report_over: Callable[[int | None, Path], object],
        end: Callable[[int], object] = os._exit,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        """Limits the process to a share of the cap and watches it in a daemon thread.

        A share of it, not all of it, because the cap is the line past which this worker is ended: the caches are cut
        back before the process reaches it, so the watch has nothing to end. A process with no cap keeps the limit it
        was given, and its caches are cut back only once it is over that."""
        self.limit(int(cap.worker_bytes * MEMORY_SETTLE_SHARE))
        threading.Thread(
            target=self.watch_memory, args=(cap, report_over, end, sleep), name="memory watch", daemon=True
        ).start()

    def watch_memory(
        self,
        cap: MemoryCap,
        report_over: Callable[[int | None, Path], object],
        end: Callable[[int], object],
        sleep: Callable[[float], object],
    ) -> None:
        """The watch itself: returns once it ended the process."""
        over_since: float | None = None
        first_over = 0
        while True:
            sleep(MEMORY_WATCH_SECONDS)
            used = self._memory_meter.resident_bytes()
            if used <= cap.worker_bytes:
                over_since = None
                continue
            if over_since is None:
                over_since, first_over = self._clock(), used
                self._requested = True
                continue
            if self._clock() - over_since < cap.grace_seconds:
                continue
            index = None if self._call is None else self._call[0]
            diagnosis = self._diagnose(cap, first_over, used)
            logger.warning(
                "This worker holds %d bytes, over its memory cap of %d for more than %s seconds, during call %s: ending it; "
                "diagnosis %s",
                used,
                cap.worker_bytes,
                cap.grace_seconds,
                index,
                diagnosis,
            )
            report_over(index, diagnosis)
            end(MEMORY_EXIT_CODE)
            return

    def _diagnose(self, cap: MemoryCap, first_over: int, used: int) -> Path:
        directory = cap.diagnosis_directory
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now():%Y-%m-%d_%H-%M-%S}-worker-{os.getpid()}"
        call = self._call
        lines = [
            f"Worker {os.getpid()} stayed over its memory cap of {cap.worker_bytes} bytes for more than {cap.grace_seconds} "
            "seconds and ended itself.",
            f"It held {first_over} bytes when first seen over the cap, and {used} bytes at the end.",
            "",
            "The call it was running:",
            "none" if call is None else f"call {call[0]}: {_name(call[1])}",
            "",
        ]
        if self._cleared is None:
            lines.append("Its caches were never cleared.")
        else:
            at, before, after, caches = self._cleared
            lines.extend(
                (
                    f"Its caches were last cleared at {at:%H:%M:%S}, from {before} bytes to {after} bytes:",
                    _table(("cache", "copies", "entries"), [(name, str(count), str(entries)) for name, count, entries in caches]),
                )
            )
        # No census of the objects alive: this runs in the watch's own thread, and walking the garbage collector's objects
        # from there hands out references to whatever the worker is building at that instant — a tuple half filled by a
        # generator then fails in the worker with "SystemError: bad argument to internal function", and that error, not
        # the memory, is what the training sees. Running the pickled call again under tracemalloc says far more anyway.
        lines.append("")
        lines.extend(
            (
                f"Its latest {len(self._calls)} calls before this one:",
                _table(("call", "function", "bytes held after"), [(str(i), name, str(held)) for i, name, held in self._calls]),
                "",
            )
        )
        if call is not None:
            pickled = directory / f"{stem}.pickle"
            try:
                with pickled.open("wb") as file:
                    pickle.dump((call[1], tuple(call[2])), file)
                lines.append(f"The call, pickled to be run again alone: {pickled}")
                lines.append(f"Run it again under tracemalloc: openmind-rerun-call {pickled}")
            except Exception as error:  # noqa: BLE001 - any pickling failure only goes in the diagnosis.
                pickled.unlink(missing_ok=True)
                lines.append(f"The call could not be pickled: {error!r}")
        path = directory / f"{stem}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _name(function: Callable[..., object]) -> str:
    return getattr(function, "__qualname__", repr(function))


def _table(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    if not rows:
        return "none"
    widths = [max(len(row[column]) for row in (header, *rows)) for column in range(len(header))]
    return "\n".join("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in (header, *rows))


def _trim() -> None:
    """Asks the C allocator to hand memory freed by Python back to the system; nothing where there is no glibc."""
    global _libc
    try:
        if _libc is None:
            _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass
