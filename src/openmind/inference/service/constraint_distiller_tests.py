from openmind.inference.model.example import Example
from openmind.inference.service.constraint_distiller import ConstraintDistiller
from openmind.inference.service.refusal_learner import REFUSED, RefusalLearner
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Functor, Number, Variable

PAWN = Functor("piece", (Constant("white"), Constant("pawn")))
ROOK = Functor("piece", (Constant("white"), Constant("rook")))


def a_case(literals, where="one position"):
    return Example(tuple(literals), True, where)


def refused_when(*literals):
    return Clause((Literal(REFUSED, ()), *(one.denied for one in literals)))


def standing_on(row, piece):
    return [Literal("origin", (Number(row), Number(1))), Literal("grid", (Number(row), Number(1), piece))]


def moving_a_rook_while_a_pawn_stands_elsewhere():
    """A move the game allows, in a position that also holds a pawn. It is what keeps the rule tied to where the
    move starts: a constraint refusing whenever a pawn stands anywhere at all would refuse this."""
    return a_case(
        [
            Literal("origin", (Number(7), Number(1))),
            Literal("grid", (Number(7), Number(1), ROOK)),
            Literal("grid", (Number(3), Number(1), PAWN)),
        ]
    )


def test_five_constraints_about_five_pawns_become_one_about_pawns():
    """The whole point of the thing. Five constraints each about one pawn on one square are correct, and they are
    a list of pawns rather than a rule. Folded, what comes out says it of a pawn wherever it stands — and refuses
    candidates none of the five did, which is what a rule does and a list cannot."""
    clauses = [refused_when(*standing_on(row, PAWN)) for row in range(2, 7)]
    refused = [a_case(standing_on(row, PAWN)) for row in range(2, 7)]
    allowed = [a_case(standing_on(2, ROOK)), moving_a_rook_while_a_pawn_stands_elsewhere()]

    found = ConstraintDistiller().distilled(clauses, refused, allowed)

    assert len(found) == 1
    said = {one.predicate: one.arguments for one in found[0].body}
    assert isinstance(said["origin"][0], Variable)
    assert said["origin"][0] == said["grid"][0]
    assert said["grid"][2] == PAWN


def test_what_the_five_became_refuses_a_pawn_they_never_met():
    """A rule reaches further than the cases it was made from. That is the difference being paid for, and it is
    worth stating as a test rather than as a hope."""
    learner = RefusalLearner()
    clauses = [refused_when(*standing_on(row, PAWN)) for row in range(2, 7)]
    refused = [a_case(standing_on(row, PAWN)) for row in range(2, 7)]
    unseen = a_case(standing_on(8, PAWN))

    found = ConstraintDistiller(learner).distilled(clauses, refused, [a_case(standing_on(2, ROOK)), moving_a_rook_while_a_pawn_stands_elsewhere()])

    assert not learner.refuses(clauses, unseen)
    assert learner.refuses(found, unseen)


def test_a_fold_that_would_refuse_a_legal_move_is_refused():
    """The guard. Covering more of what the game refuses is the point; covering anything it allows is a move OMF
    will never make, and nothing will ever tell it what it missed."""
    clauses = [refused_when(*standing_on(row, PAWN)) for row in range(2, 7)]
    refused = [a_case(standing_on(row, PAWN)) for row in range(2, 7)]
    allowed = [a_case(standing_on(4, PAWN))]

    found = ConstraintDistiller().distilled(clauses, refused, allowed)

    assert len(found) == 5


def test_nothing_that_was_refused_is_let_through():
    """Without this, the cheapest set of constraints is no constraints at all: it refuses nothing the game allows
    because it refuses nothing whatever."""
    learner = RefusalLearner()
    clauses = [refused_when(*standing_on(row, PAWN)) for row in range(2, 7)]
    refused = [a_case(standing_on(row, PAWN)) for row in range(2, 7)]

    found = ConstraintDistiller(learner).distilled(clauses, refused, [a_case(standing_on(2, ROOK)), moving_a_rook_while_a_pawn_stands_elsewhere()])

    assert all(learner.refuses(found, one) for one in refused)


def test_a_condition_the_constraint_refuses_the_same_moves_without_is_taken_out():
    """A constraint learned in one position carries that whole position. A board reading is not surplus because it
    is always true — it is surplus because the constraint refuses the same moves without it."""
    telling = Literal("blocked", (Constant(True),))
    carried = Literal("grid", (Number(7), Number(7), ROOK))
    refused = [a_case([telling, carried])]
    allowed = [a_case([Literal("blocked", (Constant(False),)), carried])]

    found = ConstraintDistiller().distilled([refused_when(telling, carried)], refused, allowed)

    assert found[0].body == (telling,)


def test_a_fold_that_would_say_nothing_at_all_is_not_a_constraint():
    """A constraint with no conditions refuses every candidate there is, so it cannot be a cheaper way of saying
    anything."""
    white = Literal("mover", (Constant("white"),))
    black = Literal("mover", (Constant("black"),))
    refused = [a_case([white]), a_case([black])]
    allowed = [a_case([Literal("mover", (Constant("red"),))])]

    found = ConstraintDistiller().distilled([refused_when(white), refused_when(black)], refused, allowed)

    assert all(one.body for one in found)
