from dataclasses import replace
from pathlib import Path

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.model.domain import Domain
from openmind.agent.model.model_description import ModelDescription
from openmind.agent.service.game_memory import GameMemory
from openmind.dashboard.service.latest_game_reader import LatestGameReader
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rule.model.python_rule import PythonRule
from openmind.training.mapper.played_game_summary_mapper import PlayedGameSummaryMapper
from openmind.training.model.played_game import PlayedGame
from openmind.training.service.game_replayer import GameReplayer
from openmind.training.service.self_play_tests import new_self_play

MODELS = (ModelDescription("first", '{"arm": 1}'), ModelDescription("second", '{"arm": 2}'))


def play(domain: Domain, memory: GameMemory, number: int, decisive: bool) -> PlayedGame:
    """Plays games until one is decisive, or drawn, as asked, and remembers it."""
    builders = (AgentBuilder().with_iterations(5).with_exploration(1.4), AgentBuilder().with_iterations(5).with_exploration(1.4))
    for seed in range(1, 200):
        game = new_self_play().play_arm_game(domain, builders, ("first", "second"), seed, seed + 1)
        if (len(set(game.payoffs)) > 1) == decisive:
            memory.remember(PlayedGameSummaryMapper().to_summary(domain, game, "arms", None, number, MODELS, "1. a b"))
            return game
    raise AssertionError("no such game")


def test_the_latest_decisive_game_is_read_position_by_position_and_a_draw_after_it_is_skipped(tmp_path: Path) -> None:
    domain = create_tictactoe_domain()
    memory = GameMemory(create_knowledge_base("tictactoe", tmp_path))
    play(domain, memory, 1, True)
    decisive = play(domain, memory, 2, True)
    play(domain, memory, 3, False)

    view = LatestGameReader().latest_decisive(tmp_path, "tictactoe")

    assert view is not None
    assert (view.label, view.players, view.payoffs, view.record) == ("arms game 2", (("X", "first"), ("O", "second")), decisive.payoffs, "1. a b")
    assert len(view.moves) == len(decisive.actions) and len(view.pictures) == len(decisive.actions) + 1
    assert not view.pictured and view.pictures[0].startswith("cell")


def test_a_domain_that_draws_its_positions_gives_a_picture_of_each(tmp_path: Path) -> None:
    drawn = replace(create_tictactoe_domain(), picture=PythonRule("'<svg>' + ('start' if last is None else last.name) + '</svg>'"))
    memory = GameMemory(create_knowledge_base("tictactoe", tmp_path))
    game = play(drawn, memory, 1, True)

    view = LatestGameReader(lambda name: drawn).latest_decisive(tmp_path, "tictactoe")

    assert view is not None and view.pictured
    assert view.pictures[0] == "<svg>start</svg>" and view.pictures[1] == f"<svg>{game.actions[0].name}</svg>"
    summary = GameMemory(create_knowledge_base("tictactoe", tmp_path)).games()[0]
    assert len(GameReplayer(create_predictor()).positions(drawn, summary)) == len(view.pictures)


def test_without_a_knowledge_base_or_a_decisive_game_there_is_nothing_to_show(tmp_path: Path) -> None:
    domain = create_tictactoe_domain()
    assert LatestGameReader().latest_decisive(tmp_path, "tictactoe") is None
    play(domain, GameMemory(create_knowledge_base("tictactoe", tmp_path)), 1, False)
    assert LatestGameReader().latest_decisive(tmp_path, "tictactoe") is None
