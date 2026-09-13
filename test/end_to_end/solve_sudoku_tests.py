from pathlib import Path

import pytest

from openmind.entrypoint.solve import main

pytestmark = pytest.mark.log_level("INFO")

CLASSIC = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
TOP95_FIRST = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"


def write_collection(directory: Path) -> Path:
    directory.mkdir()
    (directory / "mini.txt").write_text(f"{CLASSIC}\n{TOP95_FIRST}\n", encoding="utf-8")
    return directory


def test_solve_prints_the_single_solution_and_saves_a_log(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(["sudoku", "--log-directory", str(tmp_path)])

    output = capsys.readouterr().out
    assert output.startswith("Solution 1 (probability 1.0):\ncell(1,1) = 5\ncell(1,2) = 3\ncell(1,3) = 4\n")
    assert "cell(9,9) = 9\npayoff = 1.0\nturn = 'solver'\n" in output
    assert output.splitlines()[-1].startswith("sudoku: 1 solution(s), ")
    (log_file,) = (tmp_path / "sudoku").glob("*.log")
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "INFO  openmind.entrypoint.solve Solving sudoku"
    assert any(line.startswith("DEBUG openmind.csp.service.solver fill: 1 solutions, ") for line in lines)
    assert lines[-1].startswith("INFO  openmind.entrypoint.solve sudoku: 1 solution(s), ")


def test_a_collection_prints_a_summary_per_puzzle_then_its_totals(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    puzzles = write_collection(tmp_path / "puzzles")

    main(["sudoku/mini", "--puzzle-directory", str(puzzles), "--log-directory", str(tmp_path / "log")])

    output = capsys.readouterr().out.splitlines()
    assert [line.split(": ")[0] for line in output] == ["sudoku/mini/1", "sudoku/mini/2", "sudoku/mini"]
    assert output[0].startswith("sudoku/mini/1: 1 solution(s), ")
    assert output[1].startswith("sudoku/mini/2: 1 solution(s), ")
    assert output[2].startswith("sudoku/mini: 2 puzzle(s), 2 solution(s), ")
    (log_file,) = (tmp_path / "log" / "sudoku" / "mini").glob("*.log")
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [
        "INFO  openmind.entrypoint.solve Solving sudoku/mini",
        "INFO  openmind.entrypoint.solve Solving sudoku/mini/1",
    ]
    assert "INFO  openmind.entrypoint.solve Solving sudoku/mini/2" in lines
    assert lines[-1].startswith("INFO  openmind.entrypoint.solve sudoku/mini: 2 puzzle(s), 2 solution(s), ")


def test_several_domains_each_print_their_solutions_and_save_their_own_log(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    puzzles = write_collection(tmp_path / "puzzles")

    main(["sudoku", "sudoku/mini/2", "--puzzle-directory", str(puzzles), "--log-directory", str(tmp_path / "log")])

    output = capsys.readouterr().out
    classic, puzzle = output.split("Solution 1 (probability 1.0):\n")[1:]
    assert classic.startswith("cell(1,1) = 5\n")
    assert classic.splitlines()[-1].startswith("sudoku: 1 solution(s), ")
    assert puzzle.startswith("cell(1,1) = 4\n")
    assert puzzle.splitlines()[-1].startswith("sudoku/mini/2: 1 solution(s), ")
    (classic_log,) = (tmp_path / "log" / "sudoku").glob("*.log")
    (puzzle_log,) = (tmp_path / "log" / "sudoku" / "mini" / "2").glob("*.log")
    assert classic_log.read_text(encoding="utf-8").splitlines()[0] == "INFO  openmind.entrypoint.solve Solving sudoku"
    assert (
        puzzle_log.read_text(encoding="utf-8").splitlines()[0]
        == "INFO  openmind.entrypoint.solve Solving sudoku/mini/2"
    )


def test_an_unknown_collection_is_rejected_with_the_collections_found(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    puzzles = write_collection(tmp_path / "puzzles")

    with pytest.raises(SystemExit):
        main(["sudoku/top95", "--puzzle-directory", str(puzzles), "--log-directory", str(tmp_path / "log")])

    assert "no sudoku collection 'top95' in " in capsys.readouterr().err
    assert not (tmp_path / "log").exists()


def test_a_puzzle_number_outside_the_collection_is_rejected(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    puzzles = write_collection(tmp_path / "puzzles")

    with pytest.raises(SystemExit):
        main(["sudoku/mini/3", "--puzzle-directory", str(puzzles), "--log-directory", str(tmp_path / "log")])

    assert "'sudoku/mini/3': mini has puzzles 1 to 2" in capsys.readouterr().err
    assert not (tmp_path / "log").exists()


def test_a_limit_below_one_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["sudoku", "--limit", "0", "--log-directory", str(tmp_path)])
