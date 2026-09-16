from pathlib import Path

from openmind.agent.service.game_memory import GameMemory
from openmind.agent.service.game_memory_tests import FORK, LOSING, game
from openmind.dashboard.service.model_score_reader import ModelScoreReader
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base


def test_every_model_s_games_are_read_from_the_knowledge_base(tmp_path: Path) -> None:
    memory = GameMemory(create_knowledge_base("chess", tmp_path))
    for number, payoffs in enumerate(((1.0, 0.0), (0.5, 0.5), (1.0, 0.0)), start=1):
        memory.remember(game(number, payoffs))

    scores = {score.name: score for score in ModelScoreReader().scores(tmp_path, "chess")}

    assert (scores[LOSING.name].id, scores[LOSING.name].games, scores[LOSING.name].wins, scores[LOSING.name].draws, scores[LOSING.name].losses) == (LOSING.id, 3, 2, 1, 0)
    assert (scores[FORK.name].games, scores[FORK.name].wins, scores[FORK.name].draws, scores[FORK.name].losses) == (3, 0, 1, 2)
    assert scores[FORK.name].score == 0.5 / 3
    assert scores[FORK.name].last_game != "none"


def test_without_a_knowledge_base_there_is_no_model(tmp_path: Path) -> None:
    assert ModelScoreReader().scores(tmp_path, "chess") == ()
