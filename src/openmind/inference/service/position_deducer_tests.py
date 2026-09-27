from collections.abc import Callable

import pytest

from openmind.inference.factory.inference_factory import create_position_deducer
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.model.action import Action
from openmind.world.model.state import State

type Game = Callable[[str], RuleBasedGame]


def placed(played: RuleBasedGame, state: State, *cells: tuple[int, int]) -> State:
    """That position with those cells taken in turn, whoever is to move taking each."""
    for column, row in cells:
        state = played.outcomes(state, Action("place", (("col", column), ("row", row)))).outcomes[0][0]
    return state


def test_a_win_in_one_is_proven_in_one_ply(game: Game) -> None:
    """Valuing a position without playing a game: where the rules reach an end, what the position is worth is
    not a guess at all."""
    played = game("tictactoe")
    # X takes two of the first column, O answers beside; X to move with (1, 3) winning.
    state = placed(played, played.start(), (1, 1), (2, 1), (1, 2), (2, 2))

    deduced = create_position_deducer().deduce(played, state, DeductionBudget(plies=1, seconds=5.0))

    assert deduced.proven
    assert deduced.action == Action("place", (("col", 1), ("row", 3)))
    assert deduced.payoffs == (1.0, 0.0)
    assert deduced.plies == 1


def test_it_says_who_the_line_was_reasoned_for(game: Game) -> None:
    """A payoff is a payoff to somebody, and a deduction handed on without saying whose is a deduction that
    will be read the wrong way round by whoever gets it."""
    played = game("tictactoe")

    deduced = create_position_deducer().deduce(played, played.start(), DeductionBudget(plies=1, seconds=5.0))

    assert deduced.player == "X"


def test_a_position_nothing_reaches_the_end_of_proves_nothing_rather_than_guessing(game: Game) -> None:
    """Knowing nothing is the honest answer from an opening at one ply, and a deducer that returned its best
    guess would be a heuristic pretending to be a proof."""
    played = game("tictactoe")

    deduced = create_position_deducer().deduce(played, played.start(), DeductionBudget(plies=1, seconds=5.0))

    assert not deduced.proven
    assert deduced.action is None and deduced.payoffs is None and deduced.line == ()


def test_it_says_how_far_it_looked_even_where_it_proved_nothing(game: Game) -> None:
    """What a budget bought is worth knowing: nothing proven in two plies and nothing proven in eight are
    different findings about the same position."""
    played = game("tictactoe")

    deduced = create_position_deducer().deduce(played, played.start(), DeductionBudget(plies=2, seconds=5.0))

    assert deduced.plies == 2


def test_the_line_shows_the_moves_and_where_each_one_led(game: Game) -> None:
    """A proof nobody can follow is a number to be taken on trust, which is the thing this exists not to be.

    What the line ends in is a position that *paid*, and not one an ending rule called finished: tic-tac-toe
    declares no ending rule, and a game is over where its own rules leave nobody a move. An induced game ends
    the same way, which is why it needs no ending rule either."""
    played = game("tictactoe")
    state = placed(played, played.start(), (1, 1), (2, 1), (1, 2), (2, 2))

    deduced = create_position_deducer().deduce(played, state, DeductionBudget(plies=1, seconds=5.0))

    assert len(deduced.line) == 1
    action, reached = deduced.line[0]
    assert action == deduced.action
    assert reached.model("payoff")["X"] == 1.0
    assert not played.actions(reached, player="O")


def test_deducing_from_a_position_with_no_move_says_so(game: Game) -> None:
    """A finished position is worth what the game paid, never a model's guess — so there is nothing here to
    deduce and asking is a mistake at the caller's end."""
    played = game("tictactoe")
    over = placed(played, played.start(), (1, 1), (2, 1), (1, 2), (2, 2), (1, 3))

    with pytest.raises(ValueError, match="No legal action"):
        create_position_deducer().deduce(played, over, DeductionBudget(plies=2, seconds=5.0))


def test_a_budget_of_nothing_is_refused_rather_than_quietly_proving_nothing(game: Game) -> None:
    """Nothing proven because nothing was allowed reads exactly like nothing proven because nothing is there."""
    played = game("tictactoe")

    with pytest.raises(ValueError, match="at least 1 ply"):
        create_position_deducer().deduce(played, played.start(), DeductionBudget(plies=0, seconds=5.0))
    with pytest.raises(ValueError, match="at least 1 ply"):
        create_position_deducer().deduce(played, played.start(), DeductionBudget(plies=2, seconds=0.0))


def test_what_every_action_pays_is_deduced_not_only_the_best(game: Game) -> None:
    """The same walk `deduce` does, with nothing thrown away: a move value wants what each action is worth,
    and the rules already worked it out on the way to the best one."""
    played = game("tictactoe")
    state = placed(played, played.start(), (1, 1), (2, 1), (1, 2), (2, 2))

    rated = create_position_deducer().deduce_moves(played, state, DeductionBudget(plies=1, seconds=5.0))

    paid = dict(rated)
    winning = Action("place", (("col", 1), ("row", 3)))
    assert paid[winning] == (1.0, 0.0), "the win pays what the game pays there"
    assert len(paid) < len(played.actions(state)), "one ply proves the win and not the rest"


def test_what_every_action_pays_agrees_with_the_best_action_deduced(game: Game) -> None:
    """Whatever `deduce` picks must be among the highest-paying `deduce_moves` found, or the two are reading
    the same rules differently."""
    played = game("tictactoe")
    deducer = create_position_deducer()
    for cells in ((), ((1, 1),), ((1, 1), (2, 2)), ((1, 1), (2, 2), (1, 2)), ((1, 1), (2, 1), (1, 2), (2, 2))):
        state = placed(played, played.start(), *cells)
        budget = DeductionBudget(plies=4, seconds=20.0)
        deduced = deducer.deduce(played, state, budget)
        rated = dict(deducer.deduce_moves(played, state, budget))
        if not deduced.proven or not rated:
            continue
        mover = played.players().names.index(played.acting_player(state))
        assert deduced.action in rated, "the action it chose is one it deduced a payoff for"
        assert rated[deduced.action][mover] == max(one[mover] for one in rated.values())


def test_deducing_what_every_action_pays_needs_a_legal_action_and_a_budget(game: Game) -> None:
    played = game("tictactoe")
    with pytest.raises(ValueError):
        create_position_deducer().deduce_moves(played, played.start(), DeductionBudget(plies=0, seconds=5.0))
    with pytest.raises(ValueError):
        create_position_deducer().deduce_moves(played, played.start(), DeductionBudget(plies=1, seconds=0.0))


def test_the_walk_leaves_behind_every_position_it_proved(game: Game) -> None:
    """Proving one position proves many, and all of it but the answer was being thrown away. This is the
    difference between labelling a search's root and labelling its whole tree."""
    played = game("tictactoe")
    deducer = create_position_deducer()
    # Far enough in that the plies reach an end: nothing is provable within four plies of an empty board,
    # the earliest win being the fifth.
    state = placed(played, played.start(), (1, 1), (2, 1), (1, 2))
    seen = {}

    deducer.deduce(played, state, DeductionBudget(plies=6, seconds=20.0), seen)
    proven = deducer.proven(seen)

    assert len(proven) > 1, "one question asked, many positions answered"
    assert all(len(payoffs) == len(played.players().names) for _, payoffs in proven)
    assert len({state for state, _ in proven}) == len(proven), "a state decided at several depths appears once"


def test_a_memo_given_to_one_deduction_is_reused_by_the_next(game: Game) -> None:
    """Positions of the same game share their futures, so the second question starts from what the first
    already decided."""
    played = game("tictactoe")
    deducer = create_position_deducer()
    budget = DeductionBudget(plies=5, seconds=20.0)
    start = placed(played, played.start(), (1, 1), (2, 1))
    after = placed(played, start, (1, 2))

    shared = {}
    deducer.deduce(played, start, budget, shared)
    after_first = set(shared)
    deducer.deduce(played, after, budget, shared)

    alone = {}
    deducer.deduce(played, after, budget, alone)

    assert after_first <= set(shared), "the memo is added to, never replaced"
    assert set(alone) & after_first, "the second question asked about states the first had already decided"


def test_a_deduction_keeps_its_own_memo_where_none_is_given(game: Game) -> None:
    """The memo is an offer, not a requirement: nothing that called this before has to pass one."""
    played = game("tictactoe")
    deducer = create_position_deducer()

    deduced = deducer.deduce(played, played.start(), DeductionBudget(plies=2, seconds=5.0))

    assert deduced.plies >= 1
    assert deducer.proven({}) == ()
