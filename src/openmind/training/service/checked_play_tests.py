from collections.abc import Callable

from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game
from openmind.rbs.service.game_relaxer import GameRelaxer
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.training.service.checked_play import CheckedPlay, DecidingGame
from openmind.world.model.action import Action

type Game = Callable[[str], RuleBasedGame]

#: The rule that a cell must be empty. Without it a game believes a taken square can be taken again.
EMPTY_CELL = "tictactoe without place is legal, 4"


def first(game: RuleBasedGame, state):
    """Whatever the believed rules offer first, so what is measured is the rules and not a way of choosing."""
    acting = game.joint_actions(state)
    if not acting:
        return None
    return game.actions(state, player=game.players().names[acting[0][0]])[0]


def always(action: Action):
    return lambda game, state: action


def test_a_game_whose_rules_are_right_plays_every_move_it_proposes(game: Game) -> None:
    """Believed and deciding being the same game is the case that must hold: every move it offers is allowed,
    so nothing is refused and the game runs out on its own."""
    played = game("tictactoe")

    checked = CheckedPlay().play(played, DecidingGame(played), first)

    assert checked.held_up
    assert checked.proposed is None
    assert not played.joint_actions(checked.states[-1])  # it ran until the game was over
    assert checked.plies == len(checked.states) - 1


def test_a_game_that_believes_too_much_is_stopped_where_it_proposes_the_impossible(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """A search over rules that let too much through goes looking for exactly those moves, because nothing
    answers them. Put to the deciding game, the first one ends the line."""
    played = game("tictactoe")
    believing = create_rule_based_game(knowledge, GameRelaxer(knowledge).relax("tictactoe", EMPTY_CELL))

    checked = CheckedPlay().play(believing, DecidingGame(played), always(Action("place", (("col", 1), ("row", 1)))))

    assert not checked.held_up
    assert checked.plies == 1  # it played the square once, then proposed it again
    assert checked.proposed == Action("place", (("col", 1), ("row", 1)))


def test_the_move_it_got_wrong_is_kept_with_the_position_it_offered_it_in(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """A candidate the believed rules got wrong is the one thing a learner most wants and the one thing
    walking at random almost never produces — so it is kept, not merely counted."""
    played = game("tictactoe")
    believing = create_rule_based_game(knowledge, GameRelaxer(knowledge).relax("tictactoe", EMPTY_CELL))
    twice = Action("place", (("col", 2), ("row", 2)))

    checked = CheckedPlay().play(believing, DecidingGame(played), always(twice))

    assert checked.refused == (twice,)
    assert checked.at is not None
    assert checked.at.model("cell").at((2, 2)) == "X"  # offered again where it already stood


def test_what_the_deciding_game_says_a_move_leads_to_is_what_happens(game: Game) -> None:
    """The believed rules say what is allowed; what a move then does is the deciding game's, or a line would
    drift away from the game it is being measured against after one move."""
    played = game("tictactoe")

    checked = CheckedPlay().play(played, DecidingGame(played), always(Action("place", (("col", 2), ("row", 2)))))

    assert checked.states[1].model("cell").at((2, 2)) == "X"


def test_a_game_may_be_cut_short_by_the_steps_it_is_given(game: Game) -> None:
    played = game("tictactoe")

    checked = CheckedPlay().play(played, DecidingGame(played), first, steps=3)

    assert checked.plies == 3
    assert checked.held_up


def test_a_chooser_with_nothing_to_play_ends_the_line_without_blaming_the_rules(game: Game) -> None:
    """Nothing proposed is not something impossible proposed, and a measure that could not tell them apart
    would score a planner's silence as a fault in the rules."""
    played = game("tictactoe")

    checked = CheckedPlay().play(played, DecidingGame(played), lambda game, state: None)

    assert checked.held_up
    assert checked.plies == 0


def test_the_furthest_any_line_got_is_what_a_run_reports(game: Game) -> None:
    """One line says almost nothing; what a set of them says is how far the believed rules can be trusted."""
    played = game("tictactoe")
    checked = [CheckedPlay().play(played, DecidingGame(played), first, steps=one) for one in (2, 5, 3)]

    assert CheckedPlay().furthest(checked) == 5
    assert CheckedPlay().furthest([]) == 0
