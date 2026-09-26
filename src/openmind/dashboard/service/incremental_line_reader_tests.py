from pathlib import Path

from openmind.dashboard.service.incremental_line_reader import IncrementalLineReader


def a_log(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "a.log"
    path.write_text("".join(f"{one}\n" for one in lines), encoding="utf-8")
    return path


def test_the_first_read_gives_everything_written_so_far(tmp_path: Path) -> None:
    path = a_log(tmp_path, "one", "two")

    assert IncrementalLineReader().new_lines(path) == ["one", "two"]


def test_the_next_read_gives_only_what_is_new(tmp_path: Path) -> None:
    """A training log runs to hundreds of megabytes and the page reloads every thirty seconds, so reading it
    whole each time is the difference between a dashboard and a dashboard that eats the machine."""
    path = a_log(tmp_path, "one", "two")
    reader = IncrementalLineReader()
    reader.new_lines(path)

    with path.open("a", encoding="utf-8") as file:
        file.write("three\n")

    assert reader.new_lines(path) == ["three"]


def test_a_read_with_nothing_new_gives_nothing(tmp_path: Path) -> None:
    path = a_log(tmp_path, "one")
    reader = IncrementalLineReader()
    reader.new_lines(path)

    assert reader.new_lines(path) == []


def test_a_line_still_being_written_waits_for_the_rest_of_itself(tmp_path: Path) -> None:
    """A log is appended to while it is read, so half a line is always possible — and half a line parsed as a
    whole one is a wrong reading that never comes back to be corrected."""
    path = a_log(tmp_path, "one")
    reader = IncrementalLineReader()
    reader.new_lines(path)

    with path.open("a", encoding="utf-8") as file:
        file.write("part")

    assert reader.new_lines(path) == []

    with path.open("a", encoding="utf-8") as file:
        file.write("ial\n")

    assert reader.new_lines(path) == ["partial"]


def test_a_file_replaced_by_another_is_read_from_its_start(tmp_path: Path) -> None:
    """A run restarting writes a new log where the old one was. Carrying the old offset would skip as much of
    the new log as the old one had grown to."""
    path = a_log(tmp_path, "one", "two", "three")
    reader = IncrementalLineReader()
    reader.new_lines(path)
    path.unlink()

    a_log(tmp_path, "fresh")

    assert reader.new_lines(path) == ["fresh"]


def test_a_file_that_got_shorter_is_read_from_its_start(tmp_path: Path) -> None:
    """Truncated in place rather than replaced, so the inode is the same and only the size says what happened."""
    path = a_log(tmp_path, "one", "two", "three")
    reader = IncrementalLineReader()
    reader.new_lines(path)

    path.write_text("short\n", encoding="utf-8")

    assert reader.new_lines(path) == ["short"]


def test_a_file_that_is_not_there_gives_nothing_rather_than_raising(tmp_path: Path) -> None:
    """The page is drawn whether or not a run has started, and a missing log is what "not started" looks like."""
    assert IncrementalLineReader().new_lines(tmp_path / "never written") == []


def test_forgetting_a_file_reads_it_whole_again(tmp_path: Path) -> None:
    path = a_log(tmp_path, "one", "two")
    reader = IncrementalLineReader()
    reader.new_lines(path)

    reader.forget(path)

    assert reader.new_lines(path) == ["one", "two"]


def test_two_files_are_followed_apart(tmp_path: Path) -> None:
    """One reader follows every log the page shows, and offsets shared between them would have each skipping
    what the other had read."""
    first, second = tmp_path / "first.log", tmp_path / "second.log"
    first.write_text("a\n", encoding="utf-8")
    second.write_text("b\n", encoding="utf-8")
    reader = IncrementalLineReader()

    assert reader.new_lines(first) == ["a"]
    assert reader.new_lines(second) == ["b"]


def test_a_line_that_is_not_proper_text_is_read_as_best_it_can_be(tmp_path: Path) -> None:
    """A log with a broken byte in it is still a log worth reading; raising would blank the whole page over
    one character."""
    path = tmp_path / "a.log"
    path.write_bytes(b"fine\n\xff\xfe broken\n")

    assert IncrementalLineReader().new_lines(path)[0] == "fine"
