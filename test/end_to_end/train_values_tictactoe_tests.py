from pathlib import Path

import pytest

from openmind.agent.service.game_memory import GameMemory
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.entrypoint.train_values import main
from openmind.testing.service.log_reader import said
from openmind.training.mapper.signal_library_json_mapper import SignalLibraryJsonMapper
from openmind.training.repository.signal_library_repository import SignalLibraryRepository

pytestmark = pytest.mark.log_level("INFO")

SMALL = (
    *("--games", "4", "--iterations", "10", "--seed", "1", "--arms", "2", "--workers", "1"),
    *("--seconds", "20", "--memory", "1", "--candidates", "300", "--prices", "0.1,0.01", "--max-steps", "100"),
    *("--rollout-actions", "0", "--deduction-plies", "2", "--deduction-seconds", "1", "--ponder-positions", "2"),
    *("--ponder-endings", "3"),
)


def directories(tmp_path: Path) -> tuple[str, ...]:
    return (
        *("--log-directory", str(tmp_path / "log"), "--signals-directory", str(tmp_path / "signals")),
        *("--knowledge", str(tmp_path / "knowledge")),
    )


def test_training_learns_from_every_game_as_it_ends_and_saves_the_library_and_the_games(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    main(["tictactoe", *SMALL, *directories(tmp_path)])

    (library_file,) = (tmp_path / "signals" / "tictactoe").glob("*.json")
    library = SignalLibraryRepository(SignalLibraryJsonMapper()).load(library_file)
    assert library.domain == "tictactoe" and len(library.value_bases) >= 2
    assert any(record.agreements + record.disagreements > 0 for record in library.records)
    memory = GameMemory(create_knowledge_base("tictactoe", tmp_path / "knowledge"))
    assert len(memory.games("arms")) == 4
    assert capsys.readouterr().out.endswith(f"Saved signal library {library_file}\n")
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    lines = said(log_file)
    assert sum(1 for line in lines if " openmind.training.service.lesson_learner Learned from arms game " in line) == 4
    decisive = sum(1 for line in lines if " was decisive: searching for rules before the next game starts" in line)
    assert decisive == sum(1 for line in lines if " openmind.training.service.rule_searcher Searched rules on " in line)


def test_training_carries_on_from_a_saved_library_numbering_its_games_after_those_remembered(tmp_path: Path) -> None:
    main(["tictactoe", *SMALL, *directories(tmp_path)])
    (library_file,) = (tmp_path / "signals" / "tictactoe").glob("*.json")

    main(["tictactoe", *SMALL, "--signal-library", str(library_file), *directories(tmp_path)])

    memory = GameMemory(create_knowledge_base("tictactoe", tmp_path / "knowledge"))
    assert sorted(game.number for game in memory.games("arms")) == list(range(1, 9))


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (("--prior", "rater"), "--prior rater needs rules"),
        (("--learning-rate", "-0.1"), "--learning-rate needs 0 or more"),
        (("--ponder-positions", "2"), "need --deduction-plies"),
    ],
)
def test_train_values_refuses_what_it_can_t_train_with(
    flags: tuple[str, ...], message: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--games", "1", *flags, *directories(tmp_path)])

    assert message in capsys.readouterr().err
