from pathlib import Path

import pytest

from openmind.agent.service.game_memory import GameMemory
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.entrypoint.train_values import main
from openmind.rbs.model.python_rule import PythonRule
from openmind.agent.factory.game_factory import declare_game
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.testing.service.log_reader import said
from openmind.training.constant.continuous_constant import PROOF_KEYWORD
from openmind.training.mapper.arm_library_json_mapper import ArmLibraryJsonMapper
from openmind.training.model.arm_library import ArmLibrary
from openmind.training.repository.arm_library_repository import ArmLibraryRepository

pytestmark = pytest.mark.log_level("INFO")

SMALL = (
    *("--games", "4", "--iterations", "10", "--seed", "1", "--workers", "1", "--memory", "1"),
    *("--rollout-actions", "0", "--deduction-plies", "2", "--deduction-seconds", "1", "--ponder-endings", "3"),
)


def directories(tmp_path: Path) -> tuple[str, ...]:
    return ("--log-directory", str(tmp_path / "log"), "--knowledge", str(tmp_path / "knowledge"))


def test_games_are_played_and_remembered_and_decisive_games_proofs_are_remembered(tmp_path: Path) -> None:
    main(["tictactoe", *SMALL, *directories(tmp_path)])

    memory = GameMemory(create_knowledge_base("tictactoe", tmp_path / "knowledge"))
    games = memory.games("arms")
    assert len(games) == 4
    # Without an arm library, every game is played by agents without position rules.
    assert all({model.name for model in game.models} == {"no value rules"} for game in games)
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    walked = sum(1 for line in said(log_file) if " openmind.training.service.ending_walker Walked back " in line)
    decisive = sum(1 for game in games if len(set(game.payoffs)) > 1)
    assert walked == decisive
    proofs = create_knowledge_base("tictactoe", tmp_path / "knowledge").recall(keyword=PROOF_KEYWORD)
    assert all(record.provenance.source == "proved" for record in proofs)


def arm(tmp_path: Path, name: str, weight: float) -> str:
    """An arm: a variant of tic-tac-toe valuing positions by the player's mobility at that weight."""
    base = create_knowledge_base("tictactoe", tmp_path / "knowledge")
    declarer = RuleDeclarer(base, f"tictactoe {name}")
    declarer.inherits(declare_game("tictactoe", base))
    declarer.position("here.mobility(me)", PythonRule("here.mobility(me)"), weight)
    return declarer.done()


def test_the_arms_play_with_the_library_s_contexts_and_games_are_numbered_after_those_remembered(tmp_path: Path) -> None:
    first, second = arm(tmp_path, "first", 0.1), arm(tmp_path, "second", -0.1)
    library = ArmLibraryRepository(ArmLibraryJsonMapper()).write(
        ArmLibrary("tictactoe", (("first", first), ("second", second))), tmp_path / "arms.json"
    )
    main(["tictactoe", *SMALL, *directories(tmp_path)])

    main(["tictactoe", *SMALL, "--arm-library", str(library), *directories(tmp_path)])

    memory = GameMemory(create_knowledge_base("tictactoe", tmp_path / "knowledge"))
    assert sorted(game.number for game in memory.games("arms")) == list(range(1, 9))
    later = [game for game in memory.games("arms") if game.number > 4]
    assert {model.name for game in later for model in game.models} == {"first", "second"}


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (("--prior", "rater"), "--prior rater needs rules"),
        (("--ponder-endings", "2"), "--ponder-endings needs --deduction-plies"),
        (("--rollout-limit", "5"), "--rollout-limit needs --unfinished-payoff"),
        (("--unfinished-payoff", "0.5"), "--unfinished-payoff needs --rollout-limit"),
    ],
)
def test_train_values_refuses_what_it_can_t_train_with(
    flags: tuple[str, ...], message: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--games", "1", *flags, *directories(tmp_path)])

    assert message in capsys.readouterr().err
