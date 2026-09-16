from openmind.parallel.service.memory_evictor import evict_oldest


def test_the_oldest_entries_go_and_the_newest_stay() -> None:
    kept = {number: str(number) for number in range(10)}

    dropped = evict_oldest(kept, 4)

    assert dropped == 6
    assert list(kept) == [6, 7, 8, 9]


def test_a_cache_already_within_what_it_may_keep_is_left_alone() -> None:
    kept = {1: "one", 2: "two"}

    assert evict_oldest(kept, 2) == 0 and evict_oldest(kept, 5) == 0
    assert list(kept) == [1, 2]


def test_keeping_nothing_empties_the_cache_and_keeping_less_than_nothing_is_the_same() -> None:
    kept = {1: "one", 2: "two"}

    assert evict_oldest(kept, 0) == 2 and kept == {}
    assert evict_oldest({1: "one"}, -5) == 1


def test_an_entry_written_again_counts_as_the_age_it_was_first_given() -> None:
    kept = {1: "one", 2: "two", 3: "three"}
    kept[1] = "one again"

    evict_oldest(kept, 2)

    assert list(kept) == [2, 3]
