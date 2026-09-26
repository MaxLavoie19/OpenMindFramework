from openmind.inference.constant.refusal_constant import REFUSED
from openmind.inference.model.hypothesis import Hypothesis
from openmind.inference.service.constraint_selector import ConstraintSelector
from openmind.inference.service.description_length import DescriptionLength
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Number


def a_clause(conditions):
    return Clause(
        (Literal(REFUSED, ()), *(Literal("x", (Number(one),)).denied for one in range(conditions)))
    )


def a_rule(conditions, refusing):
    return Hypothesis(a_clause(conditions), frozenset(refusing), False)


def a_selector(conditions=200):
    return ConstraintSelector(DescriptionLength(conditions=conditions))


def test_a_constraint_accounting_for_a_great_deal_is_kept():
    """The plain case. One rule refusing most of what the game refuses saves far more in pointing out the legal
    moves than it costs to say."""
    offered = [a_rule(1, range(900))]

    kept, refusing = a_selector().selected(offered, among=[1000], legal=[20])

    assert len(kept) == 1
    assert len(refusing) == 900


def test_a_constraint_accounting_for_almost_nothing_is_declined():
    """**What could not be said before.** Accounting for everything was a condition rather than a preference, so
    a rule explaining one candidate out of a thousand was kept for the same reason as a rule explaining nine
    hundred. Here it has to pay, and it does not."""
    offered = [a_rule(4, [7])]

    kept, refusing = a_selector().selected(offered, among=[1000], legal=[20])

    assert kept == ()
    assert refusing == frozenset()


def test_what_is_declined_is_reported_as_left_over_and_not_as_a_failure():
    """A set that deliberately explains less is honest only if it says what it stopped explaining. Counted among
    what it *let through*, it would read as the learner having got worse — the same mistake as the one this is
    here to fix, pointing the other way."""
    held = a_selector()
    offered = [a_rule(1, range(900))]

    _, refusing = held.selected(offered, among=[1000], legal=[20])

    assert held.left_over(refusing, among=[1000], legal=[20]) == 80


def test_the_same_candidates_accounted_for_more_simply_wins():
    """Where two ways of saying it refuse exactly the same things, nothing in the data can tell them apart, so
    what decides is what they cost to say. Five narrow rules or the one they are all instances of."""
    held = a_selector()
    pile = [a_rule(2, range(one * 100, one * 100 + 100)) for one in range(5)]
    alone = a_rule(2, range(500))

    assert held._code.theory([one.clause for one in pile]) > held._code.theory([alone.clause])  # noqa: SLF001


def offered_over(positions):
    """A broad rule and a narrow one, over that many boards of a hundred candidates with two legal moves each.

    The narrow rule is sized so it pays for itself across six boards and not across one — which is the whole
    behaviour, and a synthetic case is the only way to put a crossover somewhere a test can point at it."""
    broad = [place for board in range(positions) for place in range(board * 100, board * 100 + 50)]
    narrow = [place for board in range(positions) for place in range(board * 100 + 50, board * 100 + 84)]
    return [a_rule(1, broad), a_rule(2, narrow)], [100] * positions, [2] * positions


def test_more_positions_pay_for_more_constraints():
    """**The answer to how many, and why it was never a number anyone should have guessed.** A constraint is
    stated once and pays off in every position, so the same rule that does not earn its keep against one board
    earns it against six. The count is a fact about how much has been seen, and it rises.

    Measured on real positions rather than these: two constraints pay at the first board, three by the second,
    eight by the sixth — and seven more are declined that the arrangement before this would have kept.

    **Against almost nothing it buys nothing, and is right to.** Two legal moves are two moves' worth of
    evidence, and no rule is worth stating to account for them. A criterion that bought rules anyway would be
    reporting a theory it had not earned."""
    held = a_selector()

    alone, _ = held.selected(*offered_over(1))
    across, _ = held.selected(*offered_over(6))

    assert len(alone) == 0, "against two legal moves, no rule is worth what it costs to say"
    assert len(across) == 2, "across six boards the same two rules pay"


def test_nothing_that_turns_a_legal_move_away_is_ever_offered_or_kept():
    """It is not priced and never will be. A code would have nothing left to name such a move with, and the
    older reason stands: it is a move OMF will never make and will never hear it could have made."""
    slipping = Hypothesis(a_clause(1), frozenset(range(900)), True)

    assert not slipping.useful
