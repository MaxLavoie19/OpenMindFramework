from openmind.inference.service.clause_constraint import ClauseConstraint
from openmind.inference.service.refusal_learner import REFUSED
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Number
from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.grid import Grid
from openmind.world.model.action import Action
from openmind.world.model.state import State

NAMES = CellNames(("a", "b"), ("2", "1"))


def a_position():
    return State.of(grid=Grid.of([["a thing", None], [None, None]], NAMES), turn="white")


def refused_when(*literals):
    return Clause((Literal(REFUSED, ()), *(one.denied for one in literals)))


def test_a_candidate_is_legal_where_no_learned_constraint_refuses_it():
    staying = refused_when(Literal("origin", (Number(1), Number(1))))

    legal = ClauseConstraint().constraint([staying], "move")

    assert not legal(a_position(), origin="a2", destination="b2")
    assert legal(a_position(), origin="b2", destination="a2")


def test_with_nothing_learned_every_candidate_is_legal():
    """Learning only ever adds or tightens, so before anything is learned the whole space is allowed. That is what
    having no generators means: nothing has to produce a move for it to be possible."""
    legal = ClauseConstraint().constraint([], "move")

    assert legal(a_position(), origin="a2", destination="b2")


def test_a_constraint_reads_the_position_and_not_only_the_parameters():
    """What is on the board is why a move is refused, so a constraint that could only see the parameters could
    never say anything a game cares about."""
    occupied = refused_when(Literal("grid", (Number(1), Number(1), Constant("a thing"))))

    legal = ClauseConstraint().constraint([occupied], "move")

    assert not legal(a_position(), origin="b1", destination="a1")
    assert legal(State.of(grid=Grid.of([[None, None], [None, None]], NAMES), turn="white"), origin="b1", destination="a1")


def test_which_constraint_turned_a_move_away_can_be_asked():
    """A move OMF will not make is the mistake nothing tells it about. Being able to ask which constraint refused
    it is what makes that mistake findable at all."""
    staying = refused_when(Literal("origin", (Number(1), Number(1))))
    bridge = ClauseConstraint()

    found = bridge.refusing([staying], a_position(), Action("move", (("destination", "b2"), ("origin", "a2"))))

    assert found == (staying,)
