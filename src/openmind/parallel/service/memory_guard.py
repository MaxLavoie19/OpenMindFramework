import ctypes
import gc
import logging
import os
import pickle
import threading
import time
import weakref
from collections import Counter, deque
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from openmind.parallel.constant.parallel_constant import (
    DEFAULT_PROCESS_MEMORY,
    DIAGNOSIS_CALLS,
    DIAGNOSIS_TYPES,
    MEMORY_CHECK_INTERVAL,
    MEMORY_EXIT_CODE,
    MEMORY_LOG_SECONDS,
    MEMORY_WATCH_SECONDS,
)
from openmind.parallel.model.clearable import Clearable
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.service.memory_meter import MemoryMeter

logger = logging.getLogger(__name__)

_libc: ctypes.CDLL | None = None


class MemoryGuard:
    """Keeps one process's memory down by emptying the caches registered with it. A cache calls `remembered` before it
    keeps an entry: every MEMORY_CHECK_INTERVAL entries the process's memory is read, and above the limit every
    registered cache is cleared and freed memory is handed back to the system. Caches are held weakly, so a service that
    is no longer used goes away with its cache.

    In a worker under a memory cap, `watch` also reads the memory every MEMORY_WATCH_SECONDS from a thread of its own.
    Over the cap, it asks the caches to clear at their next entry, since the thread can't safely empty a cache another
    thread is using; still over the cap after the cap's grace, it writes a diagnosis, reports it, and ends the worker
    with MEMORY_EXIT_CODE."""

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

    @property
    def limit_bytes(self) -> int:
        return self._limit

    def limit(self, memory_bytes: int) -> None:
        """How many bytes the process holds before its caches are cleared."""
        self._limit = memory_bytes

    def register(self, cache: Clearable) -> None:
        self._caches.add(cache)

    def remembered(self) -> None:
        """Called by a registered cache before it keeps an entry."""
        self._entries += 1
        if self._requested or (
            self._entries % MEMORY_CHECK_INTERVAL == 0 and self._memory_meter.resident_bytes() > self._limit
        ):
            self.clear()

    def clear(self) -> None:
        """Empties every registered cache and hands freed memory back to the system."""
        requested, self._requested = self._requested, False
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
                "Cleared %d caches holding %d entries %s: this process held %d bytes, over its limit of %d, and now holds "
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
        """Limits the process to the cap and watches it in a daemon thread."""
        self.limit(cap.worker_bytes)
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
        lines.append("")
        live = Counter(f"{type(item).__module__}.{type(item).__qualname__}" for item in gc.get_objects())
        lines.extend(
            (
                "Its most numerous objects tracked by the garbage collector at the end:",
                _table(("type", "objects"), [(name, str(count)) for name, count in live.most_common(DIAGNOSIS_TYPES)]),
                "",
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
