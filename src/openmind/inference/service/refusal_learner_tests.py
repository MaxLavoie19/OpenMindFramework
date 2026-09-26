from dataclasses import dataclass

from openmind.inference.model.evidence import Evidence
from openmind.inference.model.example import Example
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.candidate_readings import CandidateReadings
from openmind.inference.service.refusal_learner import REFUSED, RefusalLearner
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Functor, Number, Variable
from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.grid import Grid
from openmind.structure.model.record import Record
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.world.model.state import State

NAMES = CellNames(("a", "b"), ("2", "1"))
CELLS = ("a2", "b2", "a1", "b1")
DOMAINS = {"origin": CELLS, "destination": CELLS}
PLENTY = InferenceBudget(seconds=30.0)


@dataclass(frozen=True, slots=True)
class Piece(Record):
    colour: str
    type: str


@dataclass(frozen=True, slots=True)
class Square(Record):
    colour: str
    piece: Value = None


def a_position(**pieces):
    """A two by two board, empty but where a piece is named. Square colours alternate, as a board's do."""
    colours = {"a2": "white", "b2": "black", "a1": "black", "b1": "white"}
    return State.of(
        grid=Grid.of(
            [
                [Square(colours["a2"], pieces.get("a2")), Square(colours["b2"], pieces.get("b2"))],
                [Square(colours["a1"], pieces.get("a1")), Square(colours["b1"], pieces.get("b1"))],
            ],
            NAMES,
        ),
        turn="white",
    )


def a_move(origin, destination):
    return Action("move", (("destination", destination), ("origin", origin)))


def evidence_where(state, occupied):
    """The game allows moving the one piece there is to any square nothing stands on. The learner is told only
    which moves are listed, never this rule."""
    legal = tuple(a_move(occupied, one) for one in CELLS if one != occupied)
    return Evidence(state, "move", legal)


def a_case(literals, refused, where="one position"):
    return Example(tuple(literals), refused, where)


def test_which_way_a_number_lies_is_a_value_and_so_it_generalises():
    """A reading names what it is about and then says what was read of it, and only the naming places are held to
    agreeing. Which way one number lies from another is a value, as much as what stands on a square is — and
    while it was frozen, a thing going one way and a thing going the other were two rules nothing could join.
    That is why every pawn rule is written twice and a diagonal takes two rules where it should take one."""
    learner = RefusalLearner()
    going = Literal("from nothing", (Constant("x"), Constant("after"), Number(2)))
    clause = Clause((Literal(REFUSED, ()), going.denied))

    found = learner.generalised(
        clause, a_case([Literal("from nothing", (Constant("x"), Constant("before"), Number(2)))], True)
    )

    assert found is not None
    said = [one for one in found.body if one.predicate == "from nothing"]
    assert said, "the reading survived rather than being dropped for want of a partner"
    assert isinstance(said[0].arguments[1], Variable), "the way became a variable"
    assert said[0].arguments[0] == Constant("x"), "the place it names did not"


def test_a_place_a_reading_names_still_has_to_agree():
    """The other half. Two readings naming different places are not two accounts of one thing, and pairing them
    gives a variable standing for either of two unrelated places, which is true of everything."""
    learner = RefusalLearner()
    clause = Clause(
        (Literal(REFUSED, ()), Literal("from nothing", (Constant("x"), Constant("after"), Number(2))).denied)
    )

    found = learner.generalised(
        clause, a_case([Literal("from nothing", (Constant("y"), Constant("after"), Number(2)))], True)
    )

    assert found is None or not [one for one in found.body if one.predicate == "from nothing"]


def test_a_reading_of_two_cases_ties_their_places_together_with_one_variable():
    """The whole reason for generalising upward. Two moves along a row, on different rows, have the destination's
    row equal to the origin's row in each — and the same differing pair of terms always gives the same variable,
    so what comes out says the two rows are equal. Nothing proposed it and nothing searched for it."""
    learner = RefusalLearner()
    clause = Clause(
        (
            Literal(REFUSED, ()),
            Literal("origin", (Number(4), Number(1))).denied,
            Literal("destination", (Number(4), Number(5))).denied,
        )
    )

    wider = learner.generalised(
        clause,
        a_case([Literal("origin", (Number(2), Number(1))), Literal("destination", (Number(2), Number(7)))], True),
    )

    origin = next(one for one in wider.body if one.predicate == "origin")
    destination = next(one for one in wider.body if one.predicate == "destination")
    assert isinstance(origin.arguments[0], Variable)
    assert origin.arguments[0] == destination.arguments[0]
    assert origin.arguments[1] == Number(1)


def test_generalising_goes_inside_a_term_rather_than_giving_up_on_it():
    """A white rook and a white king agree on being a piece and on the colour and differ on the kind. Stopping at
    the top would throw all of that away the moment the two were not identical."""
    learner = RefusalLearner()
    rook = Functor("piece", (Constant("white"), Constant("rook")))
    king = Functor("piece", (Constant("white"), Constant("king")))
    clause = Clause((Literal(REFUSED, ()), Literal("grid", (Number(1), Number(1), rook)).denied))

    wider = learner.generalised(clause, a_case([Literal("grid", (Number(1), Number(1), king))], True))

    held = wider.body[0].arguments[2]
    assert isinstance(held, Functor) and held.name == "piece"
    assert held.arguments[0] == Constant("white")
    assert isinstance(held.arguments[1], Variable)


def test_a_reading_of_nothing_but_lone_variables_is_dropped():
    """It asks only that the case have such a reading, which every case does. It restricts nothing and costs time
    on every question."""
    learner = RefusalLearner()
    clause = Clause((Literal(REFUSED, ()), Literal("grid", (Number(1), Number(1), Constant("a"))).denied))

    wider = learner.generalised(clause, a_case([Literal("grid", (Number(2), Number(2), Constant("b")))], True))

    assert wider is None


def test_what_the_game_allows_is_never_refused():
    """Letting a refused candidate through is a move OMF proposes and the game rejects, which it finds out about
    at once. Refusing an allowed one is a move OMF will never find, and nothing will ever tell it what it missed.
    The two are not priced the same, so one of them is not tolerated at all."""
    readings, learner = CandidateReadings(), RefusalLearner()
    evidence = evidence_where(a_position(a2=Piece("white", "rook")), "a2")

    learned = learner.learn(readings.cases(evidence, DOMAINS), PLENTY)

    assert not learner.scored(learned, evidence, DOMAINS).forbade


def test_everything_the_game_refuses_is_accounted_for():
    readings, learner = CandidateReadings(), RefusalLearner()
    evidence = evidence_where(a_position(a2=Piece("white", "rook")), "a2")

    learned = learner.learn(readings.cases(evidence, DOMAINS), PLENTY)

    assert not learner.scored(learned, evidence, DOMAINS).allowed


def test_what_was_learned_in_one_position_is_widened_by_the_next_rather_than_learned_again():
    """A game met a position at a time offers the same rule over and over in slightly different circumstances. A
    learner starting fresh each time would end with as many constraints as it had seen moves."""
    readings, learner = CandidateReadings(), RefusalLearner()
    first = evidence_where(a_position(a2=Piece("white", "rook")), "a2")
    second = evidence_where(a_position(b1=Piece("white", "rook")), "b1")

    held = learner.learn(readings.cases(first, DOMAINS), PLENTY)
    both = learner.learn(readings.cases(second, DOMAINS), PLENTY, starting=held)

    assert learner.scored(both, second, DOMAINS).settled
    assert len(both) <= 2 * len(held)


def test_a_constraint_that_refuses_a_legal_move_is_repaired_from_what_it_must_keep_refusing():
    """Dropping it outright would bring back everything it rightly refused, and the next position would learn it
    all again. It is given more to say instead, since a constraint that says more covers less."""
    learner = RefusalLearner()
    clause = Clause((Literal(REFUSED, ()), Literal("turn", (Constant("white"),)).denied))
    keeping = [a_case([Literal("turn", (Constant("white"),)), Literal("blocked", (Constant(True),))], True)]
    without = [a_case([Literal("turn", (Constant("white"),)), Literal("blocked", (Constant(False),))], False)]

    repaired = learner.repaired(clause, keeping, without)

    assert Literal("blocked", (Constant(True),)) in repaired.body
    assert learner.covers(repaired, keeping[0])
    assert not learner.covers(repaired, without[0])


def test_mending_works_where_the_kept_cases_share_nothing():
    """**The fault that made mending dead code, said as a test.** It used to take only conditions *every* kept
    case carries, because then adding one loses none of them. Across boards they share almost nothing: measured
    over ninety-five repairs in a day's runs, one was mended and ninety-four were not, and every one of those
    was regrown from nothing instead — so most of the constraint set was being rebuilt each position while
    everything claimed it was carried forward.

    Here the two kept cases have no reading in common but the one the constraint already says. Asking what they
    share gives nothing; asking what tells them from the move that must be released gives an answer."""
    learner = RefusalLearner()
    clause = Clause((Literal(REFUSED, ()), Literal("turn", (Constant("white"),)).denied))
    keeping = [
        a_case([Literal("turn", (Constant("white"),)), Literal("blocked", (Constant(True),))], True),
        a_case([Literal("turn", (Constant("white"),)), Literal("elsewhere", (Constant(9),))], True),
    ]
    without = [a_case([Literal("turn", (Constant("white"),)), Literal("blocked", (Constant(False),))], False)]

    repaired = learner.repaired(clause, keeping, without)

    assert repaired is not None, "the old way found nothing to say and gave up"
    assert not learner.covers(repaired, without[0]), "it releases the move the game allows"
    assert any(learner.covers(repaired, one) for one in keeping), "and goes on refusing something"


def test_mending_may_keep_less_than_all_of_what_it_refused():
    """**Losing some is the point, because the alternative was losing every bit of it.** A condition that tells
    the kept cases from the move to release will not usually hold of all of them, so what it no longer refuses
    goes back to being unaccounted for — which is exactly where regrowing left it, having thrown the whole
    constraint away first."""
    learner = RefusalLearner()
    clause = Clause((Literal(REFUSED, ()), Literal("turn", (Constant("white"),)).denied))
    keeping = [
        a_case([Literal("turn", (Constant("white"),)), Literal("blocked", (Constant(True),))], True),
        a_case([Literal("turn", (Constant("white"),)), Literal("elsewhere", (Constant(9),))], True),
    ]
    without = [a_case([Literal("turn", (Constant("white"),)), Literal("blocked", (Constant(False),))], False)]

    repaired = learner.repaired(clause, keeping, without)

    assert sum(1 for one in keeping if learner.covers(repaired, one)) < len(keeping)


def test_a_constraint_no_reading_can_tell_apart_is_not_repaired():
    """Where the candidates it must keep refusing and the move it must release read exactly alike, the constraint
    was wrong rather than merely too broad. None is the honest answer, and the caller drops it."""
    learner = RefusalLearner()
    clause = Clause((Literal(REFUSED, ()), Literal("turn", (Constant("white"),)).denied))
    alike = [Literal("turn", (Constant("white"),))]

    assert learner.repaired(clause, [a_case(alike, True)], [a_case(alike, False)]) is None


def test_a_candidate_is_legal_where_no_constraint_covers_it_and_nothing_else():
    learner = RefusalLearner()
    refusing = Clause((Literal(REFUSED, ()), Literal("turn", (Constant("black"),)).denied))

    assert learner.refuses([refusing], a_case([Literal("turn", (Constant("black"),))], True))
    assert not learner.refuses([refusing], a_case([Literal("turn", (Constant("white"),))], False))
    assert not learner.refuses([], a_case([Literal("turn", (Constant("black"),))], True))


def test_a_constraint_can_say_what_stands_where_the_candidate_points():
    """The thing that must not be lost. A rule about how anything moves is a rule about what is on the squares it
    names, so if nothing a constraint can be built from reaches the board, no such rule exists to be found.

    It is said through the reading that ties a place to the candidate, not through the sixty-four the position
    carries about squares the candidate never mentions. Those separate nothing — every candidate in a position
    has all of them alike — while making every clause built from a case as long as the board."""
    readings, learner = CandidateReadings(), RefusalLearner()
    evidence = evidence_where(a_position(a2=Piece("white", "rook")), "a2")

    learned = learner.learn(readings.cases(evidence, DOMAINS), PLENTY)

    assert any(one.predicate == "holds" for clause in learned for one in clause.body)
    assert all(len(clause.body) <= 4 for clause in learned), "asked by size, the shortest body that works wins"
