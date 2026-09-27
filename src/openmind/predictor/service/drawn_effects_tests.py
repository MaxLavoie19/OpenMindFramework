from openmind.inference.model.example import Example
from openmind.predictor.service.drawn_effects import DrawnEffects
from openmind.statement.model.clause import Clause
from openmind.statement.model.consequence import Consequence
from openmind.statement.model.drawn import always, column, other, row
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant
from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.grid import Grid
from openmind.world.model.state import State

NAMES = CellNames(("a", "b", "c"), ("3", "2", "1"))
PLAYERS = ("white", "black")


def a_board() -> State:
    return State.of(
        piece=Grid.of([[None, None, None], ["pawn", "pawn", None], [None, None, None]], NAMES),
        turn="white",
    )


def whose(state: State) -> str:
    return state.value("turn")


def moving(**held) -> Consequence:
    """That the mover's square empties, which needs nothing read off the position."""
    fields = {
        "change": "Removed",
        "action": "move",
        "model": "piece",
        "where": (row("source"), column("source")),
    }
    return Consequence(**{**fields, **held})


def passing() -> Consequence:
    """That the turn becomes the other player's, which can only be drawn where whose it is now can be read."""
    return Consequence("Told", "move", "turn", value=other())


def playing(**held) -> DrawnEffects:
    return DrawnEffects(acting=whose, players=PLAYERS, **held)


def test_what_was_learned_of_an_action_is_what_the_action_does():
    """`EffectsRunner` raises without an effects rule, so everything else a learned game has — its legality, its
    values, its start — buys nothing until what it learned an action does can be made to happen."""
    after = playing().after((moving(),), a_board(), {"source": "a2", "target": "b3"}, "move")

    assert after.model("piece").at((2, 1)) is None
    assert after.model("piece").at((2, 2)) == "pawn"  # nothing else was touched


def test_whose_turn_it_becomes_is_drawn_from_whose_it_is():
    after = playing().after((passing(),), a_board(), {"source": "a2", "target": "b3"}, "move")

    assert after.value("turn") == "black"


def test_with_no_way_to_read_who_is_acting_nothing_is_drawn_about_the_other_player():
    """A visible failure instead of a quiet wrong one: a game whose turn passing was learned this way simply
    does not pass its turn, rather than passing it to somebody arbitrary."""
    after = DrawnEffects(players=PLAYERS).after((passing(),), a_board(), {"source": "a2"}, "move")

    assert after.value("turn") == "white"


def test_a_consequence_whose_conditions_the_position_does_not_meet_does_not_happen():
    """Without asking, every move takes something, clears every castling right and moves a castling's rook.
    A capture learned from a single sighting would fire on every move."""
    only_black = Clause((Literal("refused", ()), Literal("turn", (Constant("black"),)).denied))
    taking = moving(when=(only_black,))
    asked = playing(readings=_readings(), holds=_never_where_white)

    after = asked.after((taking,), a_board(), {"source": "a2", "target": "b3"}, "move")

    assert after.model("piece").at((2, 1)) == "pawn"  # white is to play, so it did not happen


def test_conditions_go_unasked_where_the_caller_gave_no_way_to_ask_them():
    """Not a fallback kept for tidiness: a caller with no learner — a game with no constraints yet — has no
    means to answer and should not be stopped from making anything happen."""
    only_black = Clause((Literal("refused", ()), Literal("turn", (Constant("black"),)).denied))

    after = playing().after((moving(when=(only_black,)),), a_board(), {"source": "a2", "target": "b3"}, "move")

    assert after.model("piece").at((2, 1)) is None


def _readings():
    from openmind.inference.service.candidate_readings import CandidateReadings

    return CandidateReadings()


def _never_where_white(clauses, case: Example) -> bool:
    """Stands in for the refusal learner: these conditions hold only where black is to play."""
    return any(one.predicate == "turn" and one.arguments[-1] == Constant("black") for one in case.literals)
