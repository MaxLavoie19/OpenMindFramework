from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path

import pytest

from openmind.agent.service.timekeeper import Timekeeper
from openmind.entrypoint.play import _play, main
from openmind.mcts.service.tree_search_tests import Ticking
from openmind.rbs.factory.rule_factory import create_rule_caller
from openmind.testing.service.log_reader import said
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.timing.model.time_control import TimeControl

X_WINS = ("1", "3", "1", "2", "1")  # X: (1,1) (1,2) (1,3); O: (2,1) (2,2)


type Game = Callable[[str], RuleBasedSystem]


def play(monkeypatch: pytest.MonkeyPatch, log_directory: Path, *answers: str, options: tuple[str, ...] = ()) -> None:
    replies: Iterator[str] = iter(answers)

    def reply(prompt: str) -> str:
        try:
            return next(replies)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", reply)
    main(["tictactoe", *options, "--log-directory", str(log_directory)])


def log_lines(log_directory: Path) -> list[str]:
    (log_file,) = log_directory.glob("tictactoe/*.log")
    return said(log_file)


def test_game_x_wins_ends_with_the_final_payoffs(game: Game, 
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    play(monkeypatch, tmp_path, *X_WINS)

    assert capsys.readouterr().out.endswith("payoff = Map(items=(('O', 0.0), ('X', 1.0)))\nturn = 'O'\n")


def test_invalid_input_asks_again(game: Game, 
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    play(monkeypatch, tmp_path, "x", "0", "10", "1")

    assert capsys.readouterr().out.count("Enter a number from 1 to 9.") == 3
    assert log_lines(tmp_path)[-1] == "INFO  openmind.entrypoint.play Input ended before the game was over"


def test_log_records_each_choice(game: Game, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    play(monkeypatch, tmp_path, *X_WINS)

    assert [line for line in log_lines(tmp_path) if " Chose " in line] == [
        "INFO  openmind.entrypoint.play Chose place(col=1, row=1)",
        "INFO  openmind.entrypoint.play Chose place(col=1, row=2)",
        "INFO  openmind.entrypoint.play Chose place(col=2, row=1)",
        "INFO  openmind.entrypoint.play Chose place(col=2, row=2)",
        "INFO  openmind.entrypoint.play Chose place(col=3, row=1)",
    ]


def test_agent_plays_o_until_the_game_is_over(game: Game, 
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    play(monkeypatch, tmp_path, "1", "1", "1", "1", "1", options=("--agent", "O", "--iterations", "100", "--seed", "1"))

    assert "O chose place(" in capsys.readouterr().out
    lines = log_lines(tmp_path)
    assert any(line.startswith("INFO  openmind.mcts.service.tree_search Most visited: ") for line in lines)
    assert lines[-1] == "INFO  openmind.entrypoint.play No legal action left: game over"


def test_the_agent_can_search_with_a_rollout_limit(game: Game, 
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    options = ("--agent", "O", "--iterations", "9", "--seed", "1", "--rollout-limit", "0", "--unfinished-payoff", "0.25")

    play(monkeypatch, tmp_path, "1", options=options)

    assert "O chose place(" in capsys.readouterr().out
    lines = log_lines(tmp_path)
    assert "INFO  openmind.mcts.service.tree_search place(col=1, row=2): 1 visits, mean payoff 0.25 for O" in lines


def test_a_negative_rollout_limit_is_rejected(game: Game, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--rollout-limit", "-1", "--log-directory", str(tmp_path)])


def test_unknown_agent_player_is_rejected(game: Game, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--agent", "Z", "--log-directory", str(tmp_path)])


def test_on_a_clock_the_clocks_are_shown_and_a_player_whose_time_runs_out_loses(game: Game, 
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every choice takes a second on the referee's clock: with 1.5 seconds each, X runs out on its second move."""
    replies = iter(("1", "2", "3"))
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))

    _play(game("tictactoe"), frozenset(), None, TimeControl(1.5), Timekeeper(create_rule_caller(), Ticking()))

    output = capsys.readouterr().out
    assert "X 0:01.5  O 0:01.5" in output
    assert "X 0:00.5  O 0:01.5" in output
    assert "X's time ran out" in output
    assert output.endswith("payoff = Map(items=(('O', 1.0), ('X', 0.0)))\nturn = 'X'\n")


def test_a_game_on_a_clock_plays_to_its_end_when_no_one_runs_out(game: Game, 
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    play(monkeypatch, tmp_path, *X_WINS, options=("--time-control", "10+0"))

    output = capsys.readouterr().out
    assert "X 10:00.0  O 10:00.0" in output
    assert output.endswith("payoff = Map(items=(('O', 0.0), ('X', 1.0)))\nturn = 'O'\n")
    assert any(" took " in line and " seconds, " in line for line in log_lines(tmp_path))


def test_play_refuses_a_prior_the_agent_has_no_rules_for(game: Game, 
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        play(monkeypatch, tmp_path, options=("--agent", "O", "--selection", "puct", "--prior", "value"))

    assert "--prior value needs rules" in capsys.readouterr().err


def test_the_agent_plays_by_puct(game: Game, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    play(monkeypatch, tmp_path, "5", "1", "9", "3", options=("--agent", "O", "--iterations", "50", "--seed", "1", "--selection", "puct"))

    assert "O chose " in capsys.readouterr().out
    assert any("puct with the uniform prior" in line for line in log_lines(tmp_path))
