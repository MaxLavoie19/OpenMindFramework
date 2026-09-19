from collections.abc import Iterator
from pathlib import Path

import pytest

from openmind.entrypoint.play import main
from openmind.testing.service.log_reader import said


def log_lines(log_directory: Path) -> list[str]:
    (log_file,) = (log_directory / "tictactoe" / "fourinarow").glob("*.log")
    return said(log_file)


def test_the_agent_plays_both_sides_until_the_game_is_over(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(
        [
            "tictactoe/fourinarow",
            *("--agent", "X", "--agent", "O", "--iterations", "20", "--seed", "1"),
            *("--log-directory", str(tmp_path)),
        ]
    )

    output = capsys.readouterr().out
    assert output.startswith("cell 1 2 3 4 5 6 7\n   1 . . . . . . .\n")
    assert "X chose drop(col=" in output
    assert "O chose drop(col=" in output
    lines = log_lines(tmp_path)
    assert lines[0].startswith("INFO  openmind.debug.service.debugger Debug session play tictactoe/fourinarow: ")
    assert lines[1] == "INFO  openmind.entrypoint.play Playing tictactoe/fourinarow"
    assert lines[-1] == "INFO  openmind.entrypoint.play No legal action left: game over"


def test_a_human_drops_a_mark_by_column_number(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    replies: Iterator[str] = iter(["4"])

    def reply(prompt: str) -> str:
        try:
            return next(replies)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", reply)

    main(["tictactoe/fourinarow", "--log-directory", str(tmp_path)])

    output = capsys.readouterr().out
    assert "1. drop(col=1)\n" in output
    assert "7. drop(col=7)\n" in output
    assert "   6 . . . X . . .\npayoff = Map(items=(('O', None), ('X', None)))\nturn = 'O'\n" in output
    assert log_lines(tmp_path)[-2:] == [
        "INFO  openmind.entrypoint.play Chose drop(col=4)",
        "INFO  openmind.entrypoint.play Input ended before the game was over",
    ]
