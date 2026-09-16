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
        self.kept: list[int] = []

    def memory_entries(self) -> int:
        return self.entries

    def evict_memory(self, entries: int) -> None:
        self.kept.append(entries)
        self.entries = min(self.entries, max(entries, 0))

    def clear_memory(self) -> None:
        self.entries = 0
        self.clears += 1


def guarded(held: int, limit: int, entries: int = 5) -> tuple[MemoryGuard, Cache]:
    guard = MemoryGuard(Meter(held))  # type: ignore[arg-type]
    guard.limit(limit)
    cache = Cache(entries)
    guard.register(cache)
    return guard, cache


def settled(guard: MemoryGuard, times: int = 1) -> None:
    """Remembers entries until the guard has read the memory that many times."""
    for _ in range(times * MEMORY_CHECK_INTERVAL):
        guard.remembered()


def test_a_process_over_its_limit_drops_the_oldest_entries_rather_than_all_of_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    guard, cache = guarded(1_200, 1_000, entries=1_000)

    settled(guard)

    assert (cache.entries, cache.clears) == (850, 0)  # 85% of what it held stays; the oldest 15% go.
    assert any(
        message.startswith("Cut the caches back to 850 entries of 1000: this process held 1200 bytes, over its limit of 1000")
        for message in caplog.messages
    )


def test_a_process_under_its_limit_keeps_everything_it_holds() -> None:
    guard, cache = guarded(500, 1_000)

    settled(guard, 2)

    assert (cache.entries, cache.kept, cache.clears) == (5, [], 0)


def test_a_cut_that_did_not_bring_the_memory_down_is_not_repeated() -> None:
    """Memory a process frees isn't always given back: a process whose own memory sits over its limit stays over
    whatever its caches do, and cutting at every reading would take them down to nothing."""
    guard, cache = guarded(1_200, 1_000, entries=1_000)

    settled(guard, 5)

    assert (cache.entries, cache.kept) == (850, [850])


def test_the_caches_are_cut_again_once_the_memory_climbs_past_where_it_stood_at_the_last_cut() -> None:
    meter = Meter(1_200)
    guard = MemoryGuard(meter)  # type: ignore[arg-type]
    guard.limit(1_000)
    cache = Cache(1_000)
    guard.register(cache)
    settled(guard)

    meter.held = 1_215  # Within 2% of the limit of where it stood: the wobble of pages, not growth.
    settled(guard)
    assert cache.entries == 850

    meter.held = 1_300  # Grown well past it: the caches are what's growing, so they are cut again.
    settled(guard)
    assert cache.entries == 722


def test_a_process_back_under_its_limit_starts_afresh() -> None:
    meter = Meter(1_200)
    guard = MemoryGuard(meter)  # type: ignore[arg-type]
    guard.limit(1_000)
    cache = Cache(1_000)
    guard.register(cache)
    settled(guard)

    meter.held = 900
    settled(guard)
    meter.held = 1_100  # Over again, though lower than at the first cut.
    settled(guard)

    assert cache.entries == 722


def test_each_cache_drops_its_share_of_what_has_to_go() -> None:
    meter = Meter(1_200)
    guard = MemoryGuard(meter)  # type: ignore[arg-type]
    guard.limit(1_000)
    small, large = Cache(400), Cache(1_600)
    guard.register(small)
    guard.register(large)

    settled(guard)

    assert guard.entry_budget == 1_700
    assert (small.entries, large.entries) == (340, 1_360)


def test_clearing_everything_is_left_for_when_a_worker_must_give_back_what_it_can(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    guard, cache = guarded(2_000, 1_000)

    guard.clear()

    assert (cache.entries, cache.clears) == (0, 1)
    assert any(message.startswith("Cleared 1 caches holding 5 entries on reading the memory") for message in caplog.messages)


def test_a_clear_with_nothing_held_says_nothing() -> None:
    guard, cache = guarded(2_000, 1_000, entries=0)

    guard.clear()

    assert (cache.clears, guard.held_entries()) == (0, 0)


def test_a_cache_no_longer_used_goes_away_with_its_service() -> None:
    guard, cache = guarded(2_000, 1_000)

    del cache
    gc.collect()

    assert guard.held_entries() == 0


def test_a_worker_empties_every_cache_when_a_call_ends_so_nothing_carries_into_the_next(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    guard = MemoryGuard(Meter(500))  # type: ignore[arg-type]
    held_for_the_process, held_for_the_call = Cache(76_030), Cache(2_722)
    guard.register(held_for_the_process)
    guard.register(held_for_the_call)

    guard.release()

    assert (held_for_the_process.entries, held_for_the_call.entries, guard.held_entries()) == (0, 0, 0)
    assert caplog.messages == []  # Every call ends this way: nothing to say about it.


def test_a_cut_starts_afresh_once_a_call_has_ended() -> None:
    meter = Meter(1_200)
    guard = MemoryGuard(meter)  # type: ignore[arg-type]
    guard.limit(1_000)
    cache = Cache(1_000)
    guard.register(cache)
    settled(guard)  # Cut to 850 at 1200 bytes.

    guard.release()
    cache.entries = 1_000  # The next call fills the cache again, the memory standing where it did.
    settled(guard)

    assert cache.entries == 850  # Cut again, though the memory didn't climb past the previous cut.


def test_a_worker_cuts_its_caches_back_before_it_reaches_the_cap_that_would_end_it(tmp_path: Path) -> None:
    guard = MemoryGuard(Meter(100))  # type: ignore[arg-type]

    guard.watch(MemoryCap(1_000, tmp_path, 1.0), lambda index, path: None, lambda code: None, lambda seconds: None)

    assert guard.limit_bytes == 850


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
