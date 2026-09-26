from openmind.inference.model.hypothesis import Hypothesis
from openmind.inference.service.hypothesis_table import HypothesisTable
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Number

REFUSED = "refused"


def refused_when(*literals):
    return Clause((Literal(REFUSED, ()), *(one.denied for one in literals)))


def standing(what):
    return Literal("holds", (Constant("self row"), Constant("self column"), Constant("grid"), Constant(what)))


def going(how_far):
    return Literal("x", (Number(how_far),))


def a_hypothesis(clause, refusing=(0,), slips=False):
    return Hypothesis(clause, frozenset(refusing), slips)


def test_the_same_conditions_in_another_order_are_one_hypothesis():
    """A conjunction is a set. Two workers reaching the same body from two different cases have not found two
    things, and a table that thinks they have is a table whose whole purpose — not doing the same work twice —
    fails exactly where the work is most likely to be repeated."""
    table = HypothesisTable()
    one, other = refused_when(standing("nothing"), going(0)), refused_when(going(0), standing("nothing"))

    assert table.tell([a_hypothesis(one)]) == 1
    assert table.tell([a_hypothesis(other)]) == 0
    assert len(table) == 1
    assert table.knows(other)


def test_it_counts_the_work_the_sharing_saved():
    """The number that says whether any of this was worth doing. If nearly every offer is new, the workers are
    not treading on each other and the table is only overhead."""
    table = HypothesisTable()
    held = a_hypothesis(refused_when(going(0)))

    table.tell([held, held, held])

    assert table.tried == 3
    assert table.spared == 2


def test_only_what_refuses_something_and_turns_nothing_away_is_offered_for_choosing():
    """**Filtering here is what keeps choosing a set possible at all.** Offered every body ever tried, picking
    the best set is a search over subsets of everything; offered the few that refuse something and break
    nothing, it is a small problem with an exact answer."""
    table = HypothesisTable()
    table.tell([
        a_hypothesis(refused_when(going(0)), refusing=(0, 1)),
        a_hypothesis(refused_when(going(1)), refusing=(), slips=False),
        a_hypothesis(refused_when(going(2)), refusing=(0,), slips=True),
    ])

    assert [one.clause for one in table.useful()] == [refused_when(going(0))]


def test_what_is_offered_comes_briefest_first():
    """Size is the thing being minimised, so the order is the answer's order."""
    table = HypothesisTable()
    table.tell([
        a_hypothesis(refused_when(standing("nothing"), going(0)), refusing=(0,)),
        a_hypothesis(refused_when(going(0)), refusing=(0,)),
    ])

    assert [one.size for one in table.useful()] == [1, 2]


def test_a_failure_is_remembered_so_nobody_tries_it_again():
    """**This is the only thing a failed hypothesis is good for, and it is worth a lot.** A body that turns away
    a legal move prunes nothing — every subset of it turns that move away too, and a search by increasing size
    has already been through every subset. What it saves is the next case reaching the same body and paying for
    the same test."""
    table = HypothesisTable()
    slipped = refused_when(going(0))

    table.tell([a_hypothesis(slipped, refusing=(0,), slips=True)])

    assert table.knows(slipped)
    assert table.asked(slipped).slips
    assert not table.useful()


def test_what_is_given_up_when_it_is_full_is_the_failures_and_never_the_rules():
    """Forgetting a failure costs the one test that finds it again. Forgetting something worth choosing costs a
    rule, because whatever proposed it has moved on to another case and will not propose it a second time."""
    table = HypothesisTable(most=2)
    keeping = a_hypothesis(refused_when(standing("nothing")), refusing=(0,))

    table.tell([keeping])
    table.tell([a_hypothesis(refused_when(going(number)), refusing=(), slips=True) for number in range(6)])

    assert len(table) == 2
    assert table.knows(refused_when(standing("nothing"))), "the one worth keeping survived"
    assert not table.knows(refused_when(going(0))), "the oldest failure went first"


def test_holding_more_rules_than_asked_for_is_said_rather_than_obeyed():
    """A budget is a budget over what may be thrown away, and there is nothing here it is right to throw away.
    Quietly dropping rules to hit a number would lose the run's work to a setting."""
    table = HypothesisTable(most=1)

    table.tell([a_hypothesis(refused_when(going(number)), refusing=(number,)) for number in range(4)])

    assert len(table) == 4


def test_which_of_them_account_for_a_given_case():
    """What a greedy covering asks: of the rules worth having, which reach this case that nothing yet explains."""
    table = HypothesisTable()
    table.tell([
        a_hypothesis(refused_when(going(0)), refusing=(0, 1)),
        a_hypothesis(refused_when(going(1)), refusing=(2,)),
    ])

    assert [one.clause for one in table.covering(1)] == [refused_when(going(0))]
