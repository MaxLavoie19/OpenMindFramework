from dataclasses import dataclass

from openmind.inference.model.example import Example
from openmind.inference.service.candidate_readings import CandidateReadings
from openmind.inference.service.hypothetical import ALLOWED, TAKEN_AFTER, Hypothetical
from openmind.inference.service.refusal_learner import REFUSED, RefusalLearner
from openmind.predictor.service.consequence_drawer import ConsequenceDrawer
from openmind.statement.model.consequence import Consequence
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Functor, Number, Variable
from openmind.structure.model.grid import Grid
from openmind.structure.model.kind import Kind
from openmind.structure.model.record import Record
from openmind.structure.model.schema import ActionKind
from openmind.world.model.action import Action
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Cell(Record):
    row: int
    column: int


ROWS = Kind("row", values=(1, 2))
CELL = Kind("cell", parts=(("row", ROWS), ("column", ROWS)), builds=Cell)
MOVE = ActionKind("move", (("origin", CELL), ("destination", CELL)))


def a_position():
    """Two rows of two. Something stands at row 2, column 2, and nowhere else."""
    return State.of(grid=Grid.of([[None, None], [None, "a thing"]]))


def learner():
    readings = CandidateReadings()
    return RefusalLearner(readings, hypothetical=Hypothetical(readings, MOVE))


def refused_when(*literals):
    return Clause((Literal(REFUSED, ()), *(one.denied for one in literals)))


def landing_on_something():
    """A rule of the lower layer: it asks no hypothetical, so it is what a hypothetical is put to."""
    return refused_when(
        Literal("destination", (Variable("R"), Variable("C"))),
        Literal("grid", (Variable("R"), Variable("C"), Constant("a thing"))),
    )


def refused_where_somebody_could_land_on_two_two():
    """A rule of the upper layer: it refuses a move where the move from one, one to two, two would be allowed by
    the rules below. Contrived, and the shape is the shape king safety has."""
    return refused_when(
        Literal(ALLOWED, (Number(2), Number(2), Number(1), Number(1))),
    )


def a_case(state, origin, destination):
    return Example(
        CandidateReadings().read(state, Action("move", (("destination", destination), ("origin", origin)))),
        False,
        state,
    )


def test_a_hypothetical_is_answered_by_the_rules_that_ask_no_hypothetical():
    """The layer is worked out from the constraints, not declared. The one asking sits above the one answering by
    the fact of asking, which is what stops the question coming back round to itself."""
    hypothetical = Hypothetical(CandidateReadings(), MOVE)
    clauses = (landing_on_something(), refused_where_somebody_could_land_on_two_two())

    assert hypothetical.below(clauses) == (landing_on_something(),)


def test_a_question_about_another_action_is_answered_of_this_same_position():
    """Moving onto the thing at two, two is refused below, so the question comes back no, and the rule asking it
    does not fire."""
    held = learner()
    clauses = (landing_on_something(), refused_where_somebody_could_land_on_two_two())

    assert not held.covers(refused_where_somebody_could_land_on_two_two(), a_case(a_position(), Cell(1, 1), Cell(1, 2)), clauses)


def test_the_same_question_comes_back_yes_where_the_rules_below_allow_it():
    """With nothing standing at two, two, the hypothetical move is allowed below, so the rule above fires — and
    refuses a move that has nothing wrong with it of its own. That is the shape of every rule about what could
    happen rather than what is happening."""
    held = learner()
    clauses = (landing_on_something(), refused_where_somebody_could_land_on_two_two())
    empty = State.of(grid=Grid.of([[None, None], [None, None]]))

    assert held.covers(refused_where_somebody_could_land_on_two_two(), a_case(empty, Cell(1, 1), Cell(1, 2)), clauses)


def test_a_hypothetical_nobody_can_build_is_not_answered_yes():
    """A question that cannot be put is not a question answered in the affirmative. Terms standing for nothing
    settled, or too few of them to make an action, leave the rule not firing rather than firing wrongly."""
    held = learner()
    asking = refused_when(Literal(ALLOWED, (Number(2), Number(2), Variable("Unsettled"), Number(1))))

    assert not held.covers(asking, a_case(a_position(), Cell(1, 1), Cell(1, 2)), (asking,))


def test_the_rules_that_answer_are_still_there_when_the_question_is_not_asked_first():
    """**The layer has to survive the walk down the body, and it did not.**

    Every existing rule of this kind asked its hypothetical as its only condition, so the set of rules that may
    answer it was still in hand when it was asked. Put a plain reading in front — whose turn it is, what stands
    somewhere — and the walk recursed without carrying that set, so the question was put to *no* rules at all.
    Nothing below refuses the reply when there is nothing below, so it came back yes.

    Yes is the one answer it must never default to. A question that cannot be put is not a question answered in
    the affirmative: answered that way, a rule about what could happen fires everywhere, and king safety refused
    all twenty legal moves of the opening position while looking like a rule that was merely too strong.

    A reading with a variable in it and not a settled one, because a settled reading is answered in `covers`
    itself and never reaches the walk — so the first attempt at this test passed against the fault."""
    held = learner()
    asking = refused_when(
        Literal("grid", (Number(2), Number(2), Variable("What"))),
        Literal(ALLOWED, (Number(2), Number(2), Number(1), Number(1))),
    )
    clauses = (landing_on_something(), asking)

    assert not held.covers(asking, a_case(a_position(), Cell(1, 1), Cell(1, 2)), clauses)


def test_a_rule_asking_what_could_be_taken_is_above_the_rules_that_ask_nothing():
    """Every kind of asking puts a rule in the layer above, and a rule asking what the other side could then take
    asks two questions rather than one — what they could do, and whether they are allowed to.

    Pinned because it was left out of the layering while it was the only one of the three no rule was yet written
    in, so nothing showed. Left out, it answers itself: whether my king is safe after my move would depend on
    whether theirs is safe after their reply, for ever."""
    hypothetical = Hypothetical(CandidateReadings(), MOVE)
    asking = refused_when(Literal(TAKEN_AFTER, (Constant("white"), Constant("king"))))

    assert hypothetical.asked(asking)
    assert hypothetical.below((landing_on_something(), asking)) == (landing_on_something(),)


def test_a_hypothetical_is_asked_after_the_readings_that_settle_what_it_is_about():
    """**It is answered by asking rather than by looking up, so nothing in the case matches it.** Ordering the
    conditions by how many readings match each one therefore sorts it to the very front — ahead of the reading
    that says whose turn it is, which is what its first argument stands for.

    Asked there it answers no, because a term standing for nothing settled names nobody, and it answers no
    silently. `refused :- turn(Mover), taken(Mover, king)` would simply never fire, and would look like a king
    who is never in danger rather than like a question asked too early."""
    held = learner()
    body = [
        Literal(TAKEN_AFTER, (Variable("Mover"), Constant("king"))),
        Literal("turn", (Variable("Mover"),)),
    ]

    assert [one.predicate for one in held._in_order(body, a_case(a_position(), Cell(1, 1), Cell(1, 2)))] == [  # noqa: SLF001
        "turn", TAKEN_AFTER,
    ]


def test_what_the_predictor_has_worked_out_is_asked_for_and_not_kept():
    """**The two halves learn at once, so one cannot hold a copy of the other.** What a move does is still being
    worked out while what is refused is being worked out, and a constraint reaching past this board reaches it
    through drawings that have changed since anything was built.

    Pinned because handing the drawings over as a value is the natural way to write it and is wrong in the one
    way that never shows: a loop building its hypothetical before its first position hands over nothing, every
    question about the board a move leads to answers "nothing happens", and it does so in silence for the whole
    run rather than failing."""
    told: list = []
    hypothetical = Hypothetical(CandidateReadings(), MOVE, doing=lambda: told)

    assert hypothetical.doing == []

    told.append("a drawing made after this was built")

    assert hypothetical.doing == ["a drawing made after this was built"]


def test_without_a_hypothetical_service_the_question_is_simply_not_answered():
    """A learner built for a game that never asks such a thing carries none of this, and a rule that asks anyway
    does not fire."""
    assert not RefusalLearner().covers(
        refused_where_somebody_could_land_on_two_two(),
        a_case(a_position(), Cell(1, 1), Cell(1, 2)),
        (refused_where_somebody_could_land_on_two_two(),),
    )


@dataclass(frozen=True, slots=True)
class Held(Record):
    whose: str
    what: str


def a_board_of_things():
    """Two rows of two holding two players' things, and beside them a grid of something nobody can take."""
    return State.of(
        grid=Grid.of([[Held("white", "pawn"), Held("black", "king")], [Held("white", "king"), None]]),
        square_colour=Grid.of([["light", "dark"], ["dark", "light"]]),
        turn="white",
    )


def taking():
    """What a move does, as much of it as the questions need: it removes something from the board."""
    return (Consequence("Removed", "move", "grid"),)


def asking():
    """A hypothetical that knows who acts and can draw a board, which is what makes a question puttable."""
    return Hypothetical(
        CandidateReadings(), MOVE, ConsequenceDrawer(), doing=taking(), players=("white", "black"),
        acting=lambda state: state.model("turn").value,
    )


def test_a_question_about_what_this_could_cost_is_offered_for_every_kind_the_game_has():
    """The one rule this layer was built for was answerable and unaskable: a clause carrying it is checked
    correctly, and nothing ever built one. Nothing here says which kind matters — losing a pawn is legal and
    losing the king is not, and which is which is the guard's to decide."""
    where = a_board_of_things()

    found = asking().askable(Example((), True, where))

    assert {str(one.arguments[1].name) for one in found} == {"pawn", "king"}
    assert all(one.predicate == TAKEN_AFTER for one in found)


def test_whose_it_is_is_whoever_is_acting_in_that_case():
    """Said ground with this case's own mover, because that is how every other rule about ownership has been
    found: ground per case, and turned into a variable by widening across cases whose movers differ."""
    found = asking().askable(Example((), True, a_board_of_things()))

    assert {str(one.arguments[0].name) for one in found} == {"white"}


def test_a_player_s_own_name_is_not_a_kind_of_thing_to_lose():
    """A record says whose a thing is and what it is in the same breath, and whose is already being said."""
    found = asking().askable(Example((), True, a_board_of_things()))

    assert "black" not in {str(one.arguments[1].name) for one in found}


def test_nothing_is_offered_where_the_question_cannot_be_put():
    """Without a predictor there is no board to ask about, and without knowing who acts there is nobody to ask
    it for. A question that cannot be put is not a question answered yes."""
    where = a_board_of_things()

    assert Hypothetical(CandidateReadings(), MOVE).askable(Example((), True, where)) == ()
    assert asking().askable(Example((), True, None)) == ()


def test_a_search_is_offered_the_questions_beside_the_readings():
    """What was missing: a body is assembled out of what a case carries, and a question is not carried."""
    readings = CandidateReadings()

    held = RefusalLearner(
        readings,
        hypothetical=Hypothetical(
            readings, MOVE, ConsequenceDrawer(), doing=taking(), players=("white", "black"),
            acting=lambda state: state.model("turn").value,
        ),
    )
    case = Example((Literal("turn", (Constant("white"),)),), True, a_board_of_things())

    offered = held.offered(case)

    assert any(one.predicate == TAKEN_AFTER for one in offered), "the question is among what a body may use"
    assert any(one.predicate == "turn" for one in offered), "and the readings are still there"


def test_a_learner_with_no_hypothetical_is_offered_the_readings_and_nothing_else():
    """A game whose predictor has worked out nothing yet is not offered questions it cannot answer."""
    case = Example((Literal("turn", (Constant("white"),)),), True, a_board_of_things())

    assert all(one.predicate != TAKEN_AFTER for one in RefusalLearner(CandidateReadings()).offered(case))


def test_a_kind_nothing_can_be_removed_from_is_not_asked_about():
    """Chess keeps its squares' colours on a grid beside its pieces. Asking every grid gives "taken from white,
    once this is done: a light square" — a question nothing can ever answer yes, which would never stand as a
    constraint but costs a whole pass over the board to find out each time."""
    found = asking().askable(Example((), True, a_board_of_things()))

    assert {str(one.arguments[1].name) for one in found} == {"pawn", "king"}


def test_nothing_is_asked_where_nothing_a_move_does_can_lose_anything():
    """A game whose moves have only ever been seen to set a scalar has nothing to lose, as far as anybody
    knows. Nothing stands on a scalar."""
    telling = Hypothetical(
        CandidateReadings(), MOVE, ConsequenceDrawer(), doing=(Consequence("Told", "move", "turn"),),
        players=("white", "black"), acting=lambda state: state.model("turn").value,
    )

    assert telling.askable(Example((), True, a_board_of_things())) == ()


def test_a_predictor_saying_a_capture_as_a_move_is_asked_the_same_questions_as_one_saying_it_as_a_removal():
    """The two halves have to speak one language. A capture is sayable as a removal and then a move, or as the
    move alone, and both make the same board out of every position — so nothing in fitting a predictor against
    boards prefers either, and which one it settles on must not decide what can be asked."""
    where = a_board_of_things()

    def asked(change):
        return {
            str(one.arguments[1].name)
            for one in Hypothetical(
                CandidateReadings(), MOVE, ConsequenceDrawer(), doing=(Consequence(change, "move", "grid"),),
                players=("white", "black"), acting=lambda state: state.model("turn").value,
            ).askable(Example((), True, where))
        }

    assert asked("Removed") == asked("Moved") == asked("Placed") == {"pawn", "king"}


