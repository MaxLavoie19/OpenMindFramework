from pathlib import Path

from openmind.agent.service.game_memory import GameMemory
from openmind.agent.service.game_memory_tests import WEIGHTED_ARM, WIN_ARM, game
from openmind.dashboard.service.model_score_reader import ModelScoreReader
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base


def test_every_model_s_games_are_read_from_the_knowledge_base(tmp_path: Path) -> None:
    memory = GameMemory(create_knowledge_base("chess", tmp_path))
    for number, payoffs in enumerate(((1.0, 0.0), (0.5, 0.5), (1.0, 0.0)), start=1):
        memory.remember(game(number, payoffs))

    scores = {score.name: score for score in ModelScoreReader().scores(tmp_path, "chess")}

    assert (scores[WIN_ARM.name].id, scores[WIN_ARM.name].games, scores[WIN_ARM.name].wins, scores[WIN_ARM.name].draws, scores[WIN_ARM.name].losses) == (WIN_ARM.id, 3, 2, 1, 0)
    assert (scores[WEIGHTED_ARM.name].games, scores[WEIGHTED_ARM.name].wins, scores[WEIGHTED_ARM.name].draws, scores[WEIGHTED_ARM.name].losses) == (3, 0, 1, 2)
    assert scores[WEIGHTED_ARM.name].score == 0.5 / 3
    assert scores[WEIGHTED_ARM.name].last_game != "none"


def test_without_a_knowledge_base_there_is_no_model(tmp_path: Path) -> None:
    assert ModelScoreReader().scores(tmp_path, "chess") == ()
