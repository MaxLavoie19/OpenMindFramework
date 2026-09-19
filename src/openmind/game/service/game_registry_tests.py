import pytest

from openmind.game.service.game_declarer import GameDeclarer
from openmind.game.service.game_registry import GameRegistry
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.world.model.players import Players
from openmind.world.model.state import State


def declare_coin(name: str, knowledge_base: KnowledgeBase) -> str:
    declarer = GameDeclarer(knowledge_base, name)
    declarer.starts_at(State.of(payoff=None))
    declarer.played_by(Players(("me",), "payoff"))
    return declarer.done()


def test_a_registered_game_and_its_variant_are_declared_under_their_whole_name(knowledge: KnowledgeBase) -> None:
    registry = GameRegistry({"coin": declare_coin})

    assert registry.declare("coin", knowledge) == "coin"
    assert registry.declare("coin/biased", knowledge) == "coin/biased"
    assert registry.names() == ("coin",)


def test_an_unknown_game_is_refused_naming_the_games_registered(knowledge: KnowledgeBase) -> None:
    with pytest.raises(ValueError, match="Unknown game 'chess'; known games: coin"):
        GameRegistry({"coin": declare_coin}).declare("chess", knowledge)


def test_omf_s_example_games_are_registered_under_the_entry_points() -> None:
    assert set(GameRegistry().names()) >= {"prisonersdilemma", "rockpaperscissors", "sudoku", "tictactoe"}
