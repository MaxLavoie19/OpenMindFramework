from dataclasses import dataclass

from openmind.inference.model.evidence import Evidence
from openmind.inference.model.example import Example
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.candidate_readings import CandidateReadings
from openmind.inference.service.hypothetical import ALLOWED_AFTER, TAKEN_AFTER
from openmind.inference.service.refusal_learner import REFUSED, RefusalLearner
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Functor, Number, Variable
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


def a_reading(name, value):
    return Literal(name, (Constant(value),))


def reaching_further():
    """Cases where one condition can be said safely but says almost nothing, and two conditions say a great deal
    while neither of them alone may be said at all.

    `narrow` holds of one refused case and of nothing the game allows, so a search stopping at one condition
    takes it. `wide` and `paired` each hold of a legal move, so neither survives the guard on its own; together
    they hold of five refused cases and of nothing legal. That shape is not contrived — it is what a run's
    constraints were measured leaving standing, where `a rook` and `both offsets away from nothing` each turn
    away something legal and the two together turn away nothing."""
    refused = [
        a_case([a_reading("narrow", "here"), a_reading("wide", "yes"), a_reading("paired", "yes")], True),
        *(a_case([a_reading("wide", "yes"), a_reading("paired", "yes")], True) for _ in range(4)),
    ]
    allowed = [
        a_case([a_reading("wide", "yes"), a_reading("paired", "no")], False),
        a_case([a_reading("wide", "no"), a_reading("paired", "yes")], False),
    ]
    return [*refused, *allowed]


def test_a_search_stopping_at_the_first_size_that_works_misses_what_two_conditions_say():
    """The baseline the change is against, kept so the difference is visible rather than asserted. Told to look
    no further than the first size that works, the learner takes the one condition that says almost nothing."""
    learner = RefusalLearner(further=0)

    learned = learner.learn(reaching_further(), PLENTY)

    said = [one.predicate for clause in learned for one in clause.body]
    assert "narrow" in said, "it took the one condition it could say safely"


def test_two_conditions_neither_of_which_may_be_said_alone_are_reached_and_preferred():
    """What the search is for. Looking one size past the first that works finds the pair, and pricing coverage by
    what a body costs to say prefers it — five cases for two conditions against one case for one."""
    learner = RefusalLearner()

    learned = learner.learn(reaching_further(), PLENTY)

    first = learned[0]
    assert {one.predicate for one in first.body} == {"wide", "paired"}
    assert sum(1 for one in reaching_further() if one.holds and learner.covers(first, one)) == 5


def test_a_longer_body_is_not_taken_merely_for_refusing_more():
    """The other half of the price, and the reason coverage alone will not do. A body is reached past the first
    working size only where its shorter parts turn away a legal move, and those are exactly the bodies that
    refuse most — so a search choosing by coverage alone takes the longest thing it is offered every time."""
    learner = RefusalLearner()
    one_term = Clause((Literal(REFUSED, ()), a_reading("wide", "yes").denied))
    two_term = Clause((Literal(REFUSED, ()), a_reading("wide", "yes").denied, a_reading("paired", "yes").denied))

    assert learner.cost(two_term) > learner.cost(one_term)
    assert learner.cost(one_term) == 2.0, "one condition, plus what having a constraint at all costs"
    assert learner.cost(two_term) == 5.0, "two conditions squared, plus the same"


def test_a_body_whose_shorter_part_already_passed_is_never_put_to_the_guard():
    """Conditions only narrow, so a body containing a part that passed refuses no more than that part and costs
    more to say. Asking the guard about it is work that cannot change the answer."""
    learner = RefusalLearner()
    wide, paired = a_reading("wide", "yes").denied, a_reading("paired", "yes").denied
    slipped_alone = {frozenset([wide])}

    assert not learner.worth_trying(frozenset([wide, paired]), slipped_alone), "paired passed on its own"
    assert learner.worth_trying(frozenset([wide, paired]), {frozenset([wide]), frozenset([paired])})


def test_every_body_is_worth_trying_where_nothing_shorter_was_recorded():
    """At the first size there is nothing shorter to have slipped, and the prune must not turn into a filter
    that lets nothing through."""
    learner = RefusalLearner()

    assert learner.worth_trying(frozenset([a_reading("wide", "yes").denied]), None)


def test_the_prune_does_not_change_what_is_learned():
    """The point of it. It skips bodies that could not have won, so the constraint that comes out is the one
    that came out before — the same five cases for the same two conditions."""
    learner = RefusalLearner()

    learned = learner.learn(reaching_further(), PLENTY)

    first = learned[0]
    assert {one.predicate for one in first.body} == {"wide", "paired"}
    assert sum(1 for one in reaching_further() if one.holds and learner.covers(first, one)) == 5


def test_a_reading_every_candidate_carries_is_not_offered_as_a_condition() -> None:
    """It cannot tell one candidate here from another, so the only thing it can do for a body is narrow the
    constraint to positions like this one — which is how a constraint escapes the guard by saying where it
    was fitted rather than why a move is refused."""
    learner = RefusalLearner()
    here = Literal("halfmove clock", (Constant(0),))
    mine = Literal("x", (Constant(1),))
    yours = Literal("x", (Constant(2),))
    examples = (
        Example((here, mine), True, None),
        Example((here, yours), False, None),
    )

    everywhere = learner._everywhere(examples)

    assert here in everywhere, "both candidates carry it"
    assert mine not in everywhere and yours not in everywhere
    assert here not in learner.offered(examples[0], everywhere)
    assert mine in learner.offered(examples[0], everywhere)


def test_leaving_them_out_never_leaves_nothing() -> None:
    """A body of nothing is not an improvement on a body that says where it was fitted."""
    learner = RefusalLearner()
    only = Literal("halfmove clock", (Constant(0),))
    examples = (Example((only,), True, None), Example((only,), False, None))

    everywhere = learner._everywhere(examples)

    assert only in everywhere
    assert learner.offered(examples[0], everywhere) == learner.offered(examples[0])


def test_one_candidate_shares_nothing_because_there_is_nothing_to_tell_apart() -> None:
    learner = RefusalLearner()
    one = Example((Literal("x", (Constant(1),)),), True, None)

    assert learner._everywhere((one,)) == frozenset()
    assert learner._everywhere(()) == frozenset()


class ACountingHypothetical:
    """A stand-in that answers the dear question and says how often it was asked.

    What it answers is not the point — how many times it is reached is. Answering one for real means drawing a
    position and sweeping every candidate of it, so the count is the cost."""

    def __init__(self, covering=()):
        self.asked = 0
        self._covering = set(covering)

    def below(self, among):
        return tuple(one for one in among if not any(
            held.predicate in (ALLOWED_AFTER, TAKEN_AFTER) for held in one.body
        ))

    def taken(self, example, whose, what, refuses):
        self.asked += 1
        return example in self._covering


def refusing(name):
    """A constraint refusing the cases that read `name`."""
    return Clause((Literal(REFUSED, ()), Literal(name, (), negated=True)))


def asking_after():
    """A constraint asking about the board a move leads to, which is the dear kind."""
    return Clause((Literal(REFUSED, ()), Literal(TAKEN_AFTER, (Constant("white"), Constant("king")), negated=True)))


def case(*names, holds=True):
    return Example(tuple(Literal(one, ()) for one in names), holds)


def test_a_constraint_asking_about_the_board_after_is_asked_only_of_what_the_others_leave():
    """**Because answering one is a whole position swept, and the rest is a lookup.** Measured on a real board:
    forty plain constraints accounted for 14,363 of 14,400 cases in seven seconds, and the one asking about the
    board after was being put to all 14,400 — fifty-five minutes a position, where thirty-seven would do.

    `_matching` already refuses to ask such a rule at all, in these words. Nothing had met it here because the
    learner cannot build one: only a set written by hand contains one."""
    cases = [case("a"), case("b"), case("c"), case("d")]
    hypothetical = ACountingHypothetical()
    learner = RefusalLearner(CandidateReadings(), hypothetical=hypothetical)

    learner.coverage((refusing("a"), refusing("b"), asking_after()), cases)

    assert hypothetical.asked == 2, "the two nothing else accounted for, not all four"


def test_what_is_still_unaccounted_for_is_the_same_either_way():
    """The union is what learning reads, and it must not move: a case the dear rule covers is accounted for
    whether or not a cheaper rule covers it too."""
    cases = [case("a"), case("b"), case("c"), case("d")]
    hypothetical = ACountingHypothetical(covering=(cases[2],))
    learner = RefusalLearner(CandidateReadings(), hypothetical=hypothetical)

    held = learner.coverage((refusing("a"), refusing("b"), asking_after()), cases)

    assert frozenset().union(*held.values()) == frozenset({0, 1, 2}), "d is covered by nothing"


def test_every_other_constraint_keeps_the_whole_of_what_it_covers():
    """Only the dear one is narrowed. A plain constraint's set is what it has always been, asked in no
    particular order, so nothing that reads those sets depends on which was asked first."""
    cases = [case("a"), case("a", "b"), case("b"), case("c")]
    learner = RefusalLearner(CandidateReadings(), hypothetical=ACountingHypothetical())

    held = learner.coverage((refusing("a"), refusing("b"), asking_after()), cases)

    assert held[refusing("a")] == frozenset({0, 1})
    assert held[refusing("b")] == frozenset({1, 2}), "the overlap is credited to both, as before"


def test_a_set_with_nothing_asking_about_the_board_after_is_untouched():
    """Every run that learns its own rules is this one, because a learner cannot build a rule that asks about a
    board that does not exist yet."""
    cases = [case("a"), case("b")]
    learner = RefusalLearner(CandidateReadings())

    held = learner.coverage((refusing("a"), refusing("b")), cases)

    assert held[refusing("a")] == frozenset({0})
    assert held[refusing("b")] == frozenset({1})
