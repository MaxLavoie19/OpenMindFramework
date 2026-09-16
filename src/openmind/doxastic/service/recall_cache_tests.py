from openmind.doxastic.constant.doxastic_constant import MEMORY_CHECK_INTERVAL, TOLD
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.record import Record
from openmind.doxastic.service.recall_cache import RecallCache
from openmind.parallel.service.memory_meter import MemoryMeter


class FullMemoryMeter(MemoryMeter):
    """A process always over whatever share it was given."""

    def resident_bytes(self) -> int:
        return 1 << 62


def new_record(record_id: str) -> Record:
    return Record("as said", Provenance(TOLD), id=record_id)


def test_a_record_held_in_context_is_given_back_without_the_store() -> None:
    cache = RecallCache(MemoryMeter())
    cache.keep(new_record("000001"))

    assert cache.get("000001") is not None and cache.get("000002") is None
    assert (len(cache), cache.memory_entries()) == (1, 1)


def test_a_record_dropped_or_cleared_is_no_longer_in_context() -> None:
    cache = RecallCache(MemoryMeter())
    cache.keep(new_record("000001"))
    cache.keep(new_record("000002"))

    cache.drop("000001")

    assert (cache.get("000001"), cache.get("000002") is not None) == (None, True)

    cache.clear_memory()

    assert len(cache) == 0


def test_a_process_over_its_share_of_memory_lets_go_of_what_it_held() -> None:
    cache = RecallCache(FullMemoryMeter())
    cache.limit_memory(1024)

    for number in range(MEMORY_CHECK_INTERVAL):
        cache.keep(new_record(f"{number:06d}"))

    assert len(cache) == 1
