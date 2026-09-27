from collections.abc import Callable

from openmind.heuristic.model.node import Node
from openmind.knowledge.model.policy import Policy
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.factory.search_factory import create_greedy, create_improvised, create_minimax
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


class _Rating:
    """A move value that wants one action above all others.

    Keyed on the whole action and not its name, because every tic-tac-toe action is called `place` and they
    differ only in where — which is the ordinary shape of an action, not a quirk of that game."""

    def __init__(self, wanted: Action | None = None) -> None:
        self._wanted = wanted

    def rate(self, model: object, node: Node, actions: tuple[Action, ...], player: str) -> tuple[float | None, ...]:
        if self._wanted is None:
            return (None,) * len(actions)
        return tuple(1.0 if one == self._wanted else 0.0 for one in actions)


class _Flat:
    """A position value saying every unfinished position is worth the same to everybody, so that what a
    finished one paid is the only thing that can tell two actions apart."""

    def __init__(self, players: int = 2, worth: float = 0.0) -> None:
        self._values = (worth,) * players

    def values(self, model: object, node: Node) -> tuple[float, ...] | None:
        return self._values


def test_greedy_plays_the_action_the_move_value_rates_highest(game: Game, knowledge: KnowledgeBase) -> None:
    """A move value is a policy already: asked what each action is worth to the player taking it, there is
    nothing left to do but take the best. Nothing is expanded and no outcome is asked for."""
    playing, node = played(game)
    wanted = playing.actions(node.state, player="X")[2]

    silent = create_greedy().plan(None, knowledge, node, Guidance("X", move_value=(None, _Rating())), SETTINGS)
    assert silent is None, "a rater that knows nothing about any action rates none of them"

    strategy = create_greedy().plan(
        None, knowledge, node, Guidance("X", move_value=(None, _Rating(wanted))), SETTINGS
    )

    assert strategy is not None
    assert strategy.chosen(node.state) == wanted


def test_greedy_takes_the_win_it_can_see_from_a_position_value_alone(game: Game, knowledge: KnowledgeBase) -> None:
    """No move value, so each action's outcomes are valued and weighed. Every unfinished position is worth the
    same here, so the only thing separating the actions is that one of them finishes the game — and a finished
    position is worth what the game paid, never what a model guessed."""
    from openmind.structure.model.grid import Grid
    from openmind.structure.model.map import Map

    playing, _ = played(game)
    almost = State.of(
        cell=Grid((3, 3), ("X", "X", None, "O", "O", None, None, None, None)),
        turn="X",
        payoff=Map.of({"X": None, "O": None}),
    )
    node = playing.node(almost)

    strategy = create_greedy().plan(None, knowledge, node, Guidance("X", position_value=(None, _Flat())), SETTINGS)

    assert strategy is not None
    assert strategy.chosen(almost) == Action("place", (("col", 3), ("row", 1)))


def test_greedy_knows_nothing_where_neither_heuristic_fills(game: Game, knowledge: KnowledgeBase) -> None:
    """None is knows nothing, not everything worth zero — the agent falls back on improvising."""
    _, node = played(game)

    assert create_greedy().plan(None, knowledge, node, Guidance("X"), SETTINGS) is None


def test_greedy_gives_nothing_where_the_player_has_no_action(game: Game, knowledge: KnowledgeBase) -> None:
    _, node = played(game)

    assert create_greedy().plan(None, knowledge, node, Guidance("O", position_value=(None, _Flat())), SETTINGS) is None


def test_greedy_declines_to_value_positions_where_the_players_act_at_once(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """What an action leads to is not settled by that action alone when somebody else is choosing at the same
    moment, and a planner this shallow cannot ask what they will choose. It declines rather than pretending
    they pass."""
    _, node = played(game, "rockpaperscissors")

    assert (
        create_greedy().plan(None, knowledge, node, Guidance("A", position_value=(None, _Flat())), SETTINGS) is None
    )


def test_greedy_still_plays_a_rated_move_where_the_players_act_at_once(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """The move value needs no outcome, so the guard that stops the other branch does not touch it."""
    playing, node = played(game, "rockpaperscissors")
    wanted = playing.actions(node.state, player="A")[-1]

    strategy = create_greedy().plan(
        None, knowledge, node, Guidance("A", move_value=(None, _Rating(wanted))), SETTINGS
    )

    assert strategy is not None
    assert strategy.chosen(node.state) == wanted
