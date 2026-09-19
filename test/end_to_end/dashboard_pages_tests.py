from collections.abc import Callable
from pathlib import Path

from openmind.agent.service.game_memory import GameMemory
from openmind.dashboard.model.dashboard_settings import DashboardSettings
from openmind.dashboard.service.game_browser_tests import play
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.entrypoint.dashboard import page


type Game = Callable[[str], RuleBasedSystem]


def settings(tmp_path: Path) -> DashboardSettings:
    return DashboardSettings("tictactoe", tmp_path / "log", None, knowledge_directory=tmp_path / "knowledge")


def test_the_dashboard_serves_the_list_of_decisive_games_each_game_and_404_for_anything_else(game: Game, tmp_path: Path) -> None:
    memory = GameMemory(create_knowledge_base("tictactoe", tmp_path / "knowledge"))
    play(game("tictactoe"), memory, 1, True)
    play(game("tictactoe"), memory, 2, True)

    status, listing = page(settings(tmp_path), 30, "/games")
    (first_id,) = [kept.id for kept in memory.experiences()][:1]

    assert status == 200 and b"<h2>Decisive games (2)</h2>" in listing
    assert f"<a href='/game/{first_id}'>arms game 1</a>".encode() in listing
    status, game = page(settings(tmp_path), 30, f"/game/{first_id}")
    assert status == 200 and b"<h1>tictactoe arms game 1</h1>" in game and b"Next decisive game" in game
    assert page(settings(tmp_path), 30, "/game/999999")[0] == 404
    assert page(settings(tmp_path), 30, "/elsewhere")[0] == 404
    status, main = page(settings(tmp_path), 30, "/")
    assert status == 200 and b"All decisive games (2)" in main
