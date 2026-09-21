from openmind.inference.service.covering_learner import Covering
from openmind.inference.service.rule_reasoner import RuleReasoner


def a_rule(*conditions):
    return Covering(tuple(conditions), 0, 0)


def test_a_rule_asking_less_allows_everything_a_rule_asking_more_allows():
    sliding = a_rule(("things between source and target", "==", 0))
    straight = a_rule(("things between source and target", "==", 0), ("source and target share a row or a column", "==", True))

    assert RuleReasoner().entails(sliding, straight)
    assert not RuleReasoner().entails(straight, sliding)


def test_a_looser_bound_is_implied_by_a_tighter_one():
    reasoner = RuleReasoner()

    assert reasoner.implies(("steps from source to target", "<=", 2), ("steps from source to target", "<=", 4))
    assert not reasoner.implies(("steps from source to target", "<=", 4), ("steps from source to target", "<=", 2))
    assert reasoner.implies(("steps from source to target", "==", 2), ("steps from source to target", "<=", 4))
    assert reasoner.implies(("rows apart, source and target", "==", 3), ("rows apart, source and target", ">=", 1))


def test_conditions_about_different_readings_say_nothing_about_one_another():
    reasoner = RuleReasoner()

    assert not reasoner.implies(("rows apart, source and target", "==", 1), ("columns apart, source and target", "==", 1))


def test_what_a_queen_may_do_takes_in_what_a_rook_and_a_bishop_may_do():
    """Concluded from the rules, with no position looked at: a queen is worth at least a rook, always."""
    clear = ("things between source and target", "==", 0)
    rook = [a_rule(clear, ("source and target share a row or a column", "==", True))]
    bishop = [a_rule(clear, ("source and target are on a diagonal", "==", True))]
    queen = [
        a_rule(clear, ("source and target share a row or a column", "==", True)),
        a_rule(clear, ("source and target are on a diagonal", "==", True)),
    ]
    reasoner = RuleReasoner()

    assert reasoner.within(queen, rook)
    assert reasoner.within(queen, bishop)
    assert not reasoner.within(rook, bishop)
    assert reasoner.ordered({"queen": queen, "rook": rook, "bishop": bishop}) == (("queen", "rook"), ("queen", "bishop"))


def test_a_king_stepping_one_square_is_taken_in_by_a_queen_that_steps_any_distance():
    king = [a_rule(("steps from source to target", "==", 1))]
    queen = [a_rule(("steps from source to target", ">=", 1))]

    assert RuleReasoner().within(queen, king)
    assert not RuleReasoner().within(king, queen)
