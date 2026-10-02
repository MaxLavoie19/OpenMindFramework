from collections.abc import Callable

from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.training.mapper.move_row_mapper import MoveRowMapper
from openmind.training.model.played_game import PlayedGame
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction

type Game = Callable[[str], RuleBasedGame]


def a_game(played: RuleBasedGame, chosen=(), steps: int = 2) -> PlayedGame:
    """A game of that many moves, carrying what a search settled on in each."""
    state = played.start()
    states, actions = [state], []
    for column in range(1, steps + 1):
        action = Action("place", (("col", column), ("row", 1)))
        player = played.players().names[len(actions) % 2]
        actions.append(JointAction(((player, action),)))
        state = played.outcomes(state, action).outcomes[0][0]
        states.append(state)
    return PlayedGame(tuple(states), tuple(actions), (1.0, 0.0), chosen=chosen)


def a_move(column: int) -> Action:
    return Action("place", (("col", column), ("row", 1)))


def test_every_move_a_search_weighed_becomes_a_row_worth_what_it_gave_it(game: Game) -> None:
    """**What makes the next search cheaper rather than only better.** Rating moves directly costs one reading
    where valuing the position each move leads to costs one per move. The thing to learn is which moves the
    search kept returning to, and it says so as a chance for each."""
    played = game("tictactoe")
    settled = (((a_move(1), 0.7), (a_move(2), 0.3)),)

    rows = MoveRowMapper().to_rows([a_game(played, chosen=settled, steps=1)])

    assert [(one.action.parameters[0][1], one.target) for one in rows] == [(1, 0.7), (2, 0.3)]
    assert {one.player for one in rows} == {"X"}, "about the side that had the choice"


def test_a_position_gives_as_many_rows_as_the_search_had_moves_to_weigh(game: Game) -> None:
    played = game("tictactoe")
    settled = (
        ((a_move(1), 0.5), (a_move(2), 0.5)),
        ((a_move(3), 1.0),),
    )

    rows = MoveRowMapper().to_rows([a_game(played, chosen=settled, steps=2)])

    assert len(rows) == 3
    assert [one.player for one in rows] == ["X", "X", "O"], "each about whoever was choosing"


def test_a_game_nothing_searched_gives_no_move_rows(game: Game) -> None:
    """A game somebody else played says which move was made and nothing about what the others were worth. A
    mapper that spread the played move's weight over the rest would be teaching a guess."""
    played = game("tictactoe")

    assert MoveRowMapper().to_rows([a_game(played)]) == ()
