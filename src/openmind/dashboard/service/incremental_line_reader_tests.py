from pathlib import Path

from openmind.dashboard.service.incremental_line_reader import IncrementalLineReader


def test_each_call_gives_only_the_complete_lines_written_since_the_previous_one(tmp_path: Path) -> None:
    log, reader = tmp_path / "run.log", IncrementalLineReader()
    log.write_text("first\nsecond\nthi", encoding="utf-8")

    first = reader.new_lines(log)
    with log.open("a", encoding="utf-8") as file:
        file.write("rd\nfourth\n")
    second = reader.new_lines(log)

    assert (first, second, reader.new_lines(log)) == (["first", "second"], ["third", "fourth"], [])


def test_a_file_that_got_shorter_or_a_missing_one_is_read_from_its_start(tmp_path: Path) -> None:
    log, reader = tmp_path / "run.log", IncrementalLineReader()
    log.write_text("first\nsecond\n", encoding="utf-8")
    reader.new_lines(log)

    log.write_text("new\n", encoding="utf-8")

    assert reader.new_lines(log) == ["new"]
    assert reader.new_lines(tmp_path / "missing.log") == []
