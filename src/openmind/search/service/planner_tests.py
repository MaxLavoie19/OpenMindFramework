from collections.abc import Callable

from openmind.heuristic.model.node import Node
from openmind.knowledge.model.policy import Policy
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.factory.search_factory import create_improvised, create_minimax
from openmind.search.model.guidance import Guidance
from openmind.search.model.search_settings import SearchSettings
from openmind.world.model.action import Action
from openmind.world.model.state import State

type Game = Callable[[str], RuleBasedGame]

SETTINGS = SearchSettings()


def played(game: Game, name: str = "tictactoe") -> tuple[RuleBasedGame, Node]:
    playing = game(name)
    return playing, playing.node(playing.start())


def test_improvising_plays_what_the_picked_policy_s_optimizer_gives(game: Game, knowledge: KnowledgeBase) -> None:
    playing, node = played(game)

    strategy = create_improvised().plan(None, knowledge, node, Guidance("X"), SETTINGS)

    assert strategy is not None
    assert strategy.chosen(node.state) in playing.actions(node.state, player="X")
    assert strategy.states() == (node.state,)


def test_improvising_gives_nothing_where_the_player_has_no_action(game: Game, knowledge: KnowledgeBase) -> None:
    _, node = played(game)

    assert create_improvised().plan(None, knowledge, node, Guidance("O"), SETTINGS) is None


def test_improvising_follows_the_policies_the_caller_gave(game: Game, knowledge: KnowledgeBase) -> None:
    playing, node = played(game)
    middle = Policy("the middle", playing.context_id, "take the middle")

    strategy = create_improvised().plan(None, knowledge, node, Guidance("X", policies=(middle,)), SETTINGS)

    assert strategy is not None and strategy.chosen(node.state) is not None


def test_minimax_reads_a_game_out_and_never_loses_a_drawn_one(game: Game, knowledge: KnowledgeBase) -> None:
    """Tic-tac-toe is a draw with best play: from the start, minimax's own value is a draw for both."""
    playing, node = played(game)

    strategy = create_minimax().plan(None, knowledge, node, Guidance("X"), SearchSettings())

    assert strategy is not None
    chosen = strategy.chosen(node.state)
    assert chosen is not None and chosen in playing.actions(node.state, player="X")


def test_minimax_takes_the_win_it_can_see(game: Game, knowledge: KnowledgeBase) -> None:
    from openmind.structure.model.grid import Grid
    from openmind.structure.model.map import Map

    playing, _ = played(game)
    # X to play, with two marks on the top row and the third cell free.
    almost = State.of(
        cell=Grid((3, 3), ("X", "X", None, "O", "O", None, None, None, None)),
        turn="X",
        payoff=Map.of({"X": None, "O": None}),
    )
    node = playing.node(almost)

    strategy = create_minimax().plan(None, knowledge, node, Guidance("X"), SearchSettings())

    assert strategy is not None
    assert strategy.chosen(almost) == Action("place", (("col", 3), ("row", 1)))


def test_minimax_gives_nothing_where_the_game_is_already_over(game: Game, knowledge: KnowledgeBase) -> None:
    from openmind.structure.model.grid import Grid
    from openmind.structure.model.map import Map

    playing, _ = played(game)
    finished = State.of(
        cell=Grid((3, 3), ("X", "X", "X", "O", "O", None, None, None, None)),
        turn="O",
        payoff=Map.of({"X": 1.0, "O": 0.0}),
    )

    assert create_minimax().plan(None, knowledge, playing.node(finished), Guidance("X"), SearchSettings()) is None
