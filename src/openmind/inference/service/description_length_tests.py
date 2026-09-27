from math import inf, isclose

from openmind.inference.constant.refusal_constant import REFUSED
from openmind.inference.service.description_length import DescriptionLength
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Number


def refused_when(*literals):
    return Clause((Literal(REFUSED, ()), *(one.denied for one in literals)))


def a_condition(number):
    return Literal("x", (Number(number),))


def a_clause(conditions):
    return refused_when(*(a_condition(one) for one in range(conditions)))


def test_the_marginal_cost_of_a_further_condition_falls_rather_than_rises():
    """**The fault in what we had, said as a test.** Under `1 + length²` each further condition cost more than
    the last — five bits for the third, nine for the fifth. Under a code it costs less, because there are fewer
    ways left to choose it.

    The direction is not a detail. A price that rises with length prefers short constraints, and a short
    constraint in refusal semantics is an over-general one: it turns away moves the game allows in positions
    nobody has looked at yet, which is the error nothing ever reports."""
    held = DescriptionLength(conditions=200)
    marginal = [held.saying(a_clause(size + 1)) - held.saying(a_clause(size)) for size in range(1, 8)]

    assert marginal == sorted(marginal, reverse=True), marginal
    assert marginal[0] > marginal[-1]


def test_a_set_is_not_charged_for_an_order_it_does_not_have():
    """K constraints could have been written down in K! orders and they all mean the same thing. Charging for
    the order charges for something that carries nothing — about twenty-two bits at ten constraints, which is
    the difference between a set paying for itself and not."""
    held = DescriptionLength(conditions=200)
    clauses = [a_clause(2) for _ in range(10)]

    apart = held.counting(10) + sum(held.saying(one) for one in clauses)

    assert held.theory(clauses) < apart
    assert isclose(apart - held.theory(clauses), 21.79, abs_tol=0.01)


def test_naming_a_legal_move_gets_cheaper_the_more_the_constraints_refuse():
    """**The term nothing had at all, and the reason a set's size can now be argued about.** The constraints are
    never shown a corpus. What they are shown is that these candidates are legal, and the tighter they are, the
    fewer bits it takes to say which."""
    held = DescriptionLength()

    assert held.naming(survivors=14400, legal=1) > held.naming(survivors=40, legal=1)
    assert isclose(held.naming(survivors=14400, legal=1), 13.81, abs_tol=0.01)
    assert isclose(held.naming(survivors=40, legal=1), 5.32, abs_tol=0.01)


def test_refusing_a_legal_move_is_four_hundred_times_dearer_than_letting_an_illegal_one_through():
    """**The asymmetry this project has always asserted, arriving out of the encoding rather than a weight.**

    One more illegal candidate surviving moves the naming cost from log₂40 to log₂41 — under four hundredths of
    a bit, because the game rejects such a move at once and OMF learns. Refusing a legal one cannot be priced at
    all: there is then nothing left to name it with.

    Nothing here was tuned to produce that, which is why it can be believed in a position nobody has seen."""
    held = DescriptionLength()

    letting_through = held.naming(41, 1) - held.naming(40, 1)

    assert letting_through < 0.04
    assert held.naming(survivors=0, legal=1) == inf, "a refused legal move is a contradiction, not a price"


def test_a_number_nobody_bounded_is_said_without_inventing_a_bound():
    """How many constraints a game needs is not something to assume. Rissanen's code says a whole number without
    one — and costs more for bigger numbers, which is the pressure against a pile of rules."""
    held = DescriptionLength()

    assert held.counting(1) < held.counting(10) < held.counting(1000)
    assert held.counting(0) == held.counting(1), "nothing and one thing are both said at once"


def test_the_whole_length_is_what_is_said_plus_what_is_left_to_say():
    """Two terms, and the trade between them is the thing that was missing. More constraints cost more to state
    and less to live with; the point at which that stops paying is where a set should stop growing."""
    held = DescriptionLength(conditions=200)
    tighter, looser = [a_clause(2) for _ in range(4)], [a_clause(2)]

    assert held.whole(tighter, [(40, 20)]) < held.whole(looser, [(14400, 20)])
    assert held.whole((), [(14400, 20)]) == held.theory(()) + held.naming(14400, 20)
