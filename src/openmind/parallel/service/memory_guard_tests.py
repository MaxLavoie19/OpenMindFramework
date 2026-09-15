import gc
import logging
import pickle
from pathlib import Path

import pytest

from openmind.parallel.constant.parallel_constant import MEMORY_CHECK_INTERVAL, MEMORY_EXIT_CODE
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.service.memory_guard import MemoryGuard

pytestmark = pytest.mark.log_level("INFO")


class Meter:
    """A memory meter reading whatever the test sets."""

    def __init__(self, held: int) -> None:
        self.held = held

    def resident_bytes(self) -> int:
        return self.held


class Cache:
    def __init__(self, entries: int) -> None:
        self.entries = entries
        self.clears = 0

    def memory_entries(self) -> int:
        return self.entries

    def clear_memory(self) -> None:
        self.entries = 0
        self.clears += 1


def guarded(held: int, limit: int, entries: int = 5) -> tuple[MemoryGuard, Cache]:
    guard = MemoryGuard(Meter(held))  # type: ignore[arg-type]
    guard.limit(limit)
    cache = Cache(entries)
    guard.register(cache)
    return guard, cache


def test_the_caches_are_cleared_when_the_memory_read_every_interval_is_over_the_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    guard, cache = guarded(2_000, 1_000)

    for _ in range(MEMORY_CHECK_INTERVAL - 1):
        guard.remembered()
    before = cache.clears
    guard.remembered()

    assert (before, cache.clears) == (0, 1)
    assert any(message.startswith("Cleared 1 caches holding 5 entries on reading the memory") for message in caplog.messages)


def test_the_caches_are_kept_under_the_limit() -> None:
    guard, cache = guarded(500, 1_000)

    for _ in range(2 * MEMORY_CHECK_INTERVAL):
        guard.remembered()

    assert cache.clears == 0


def test_a_cache_no_longer_used_goes_away_with_its_service(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    guard, cache = guarded(2_000, 1_000)

    del cache
    gc.collect()
    guard.clear()

    assert any(message.startswith("Cleared 0 caches holding 0 entries") for message in caplog.messages)


def test_the_watch_asks_for_a_clear_then_ends_a_worker_still_over_its_cap_with_a_diagnosis(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    now = [0.0]
    guard = MemoryGuard(Meter(3_000), lambda: now[0])  # type: ignore[arg-type]
    cache = Cache(7)
    guard.register(cache)
    guard.calling(3, pow, (2, 5))
    reported: list[tuple[int | None, Path]] = []
    ended: list[int] = []

    def sleep(seconds: float) -> None:
        now[0] += seconds
        if now[0] >= 2.0:
            guard.remembered()

    guard.watch_memory(MemoryCap(1_000, tmp_path, 1.0), lambda index, path: reported.append((index, path)), ended.append, sleep)

    ((index, diagnosis),) = reported
    text = diagnosis.read_text(encoding="utf-8")
    assert (ended, index, cache.clears) == ([MEMORY_EXIT_CODE], 3, 1)
    assert "stayed over its memory cap of 1000 bytes" in text and "call 3: pow" in text
    assert "cache  copies  entries\nCache  1       7" in text
    with diagnosis.with_suffix(".pickle").open("rb") as file:
        function, arguments = pickle.load(file)
    assert function(*arguments) == 32
    assert any("over its memory cap of 1000" in message for message in caplog.messages)


def test_the_watch_lets_a_worker_back_under_its_cap_go_on() -> None:
    meter = Meter(3_000)
    now = [0.0]
    guard = MemoryGuard(meter, lambda: now[0])  # type: ignore[arg-type]
    ended: list[int] = []
    ticks = iter(range(6))

    def sleep(seconds: float) -> None:
        now[0] += seconds
        tick = next(ticks)
        meter.held = 3_000 if tick % 2 == 0 else 500
        if tick == 5:
            raise StopIteration

    with pytest.raises(StopIteration):
        guard.watch_memory(MemoryCap(1_000, Path("unused"), 1.0), lambda index, path: None, ended.append, sleep)

    assert ended == []
