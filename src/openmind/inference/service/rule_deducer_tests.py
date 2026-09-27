from openmind.inference.service.rule_deducer import Deduced, RuleDeducer
from openmind.world.model.action import Action


def move(**readings):
    """One candidate as its readings, with whether the game allowed it."""
    return {name: value for name, value in readings.items() if name != "legal"}, readings["legal"]


def a_game():
    """A made-up game where a move is legal exactly when it goes one place and lands on nothing.

    Nothing here is chess. Two readings decide it and a third is noise, which is what the deducer has to sort
    out without being told which is which."""
    return [
        move(distance=1, lands_on="nothing", colour="dark", legal=True),
        move(distance=1, lands_on="nothing", colour="light", legal=True),
        move(distance=1, lands_on="a piece", colour="dark", legal=False),
        move(distance=2, lands_on="nothing", colour="light", legal=False),
        move(distance=3, lands_on="a piece", colour="dark", legal=False),
    ]


def test_what_every_legal_action_has_in_common_is_deduced_from_nothing_but_which_were_legal():
    found = RuleDeducer().deduce(a_game())

    said = {one.readable for one in found}
    assert "lands_on == 'nothing'" in said
    assert "distance == 1" in said, "one value among the legal ones, so it is an equality and not a bound"


def test_a_condition_every_candidate_satisfies_is_true_and_useless_and_is_left_out():
    """It explains nothing: it rules out none of the ones the game refused."""
    found = RuleDeducer().deduce(a_game())

    assert all(one.excluded > 0 for one in found)
    assert not any("colour" in one.readable and one.condition[1] == "in" and len(one.condition[2]) == 2
                   for one in found), "a reading taking both values of everything rules nothing out"


def test_nothing_can_be_deduced_where_no_action_was_legal():
    """A rule can only be found where there are legal actions to find it in."""
    assert RuleDeducer().deduce([move(distance=2, lands_on="a piece", legal=False)]) == ()


def test_two_readings_that_were_always_the_same_are_a_condition_between_them():
    """What says a piece moved is the mover's own: a condition between two readings rather than a reading and a
    value."""
    examples = [
        move(mover="white", owner="white", legal=True),
        move(mover="black", owner="black", legal=True),
        move(mover="white", owner="black", legal=False),
    ]

    found = RuleDeducer().deduce(examples)

    assert any(one.condition == ("mover", "== reading", "owner") for one in found)


def test_two_readings_that_were_never_the_same_are_a_condition_too():
    """What makes an action legal is as often two readings differing as two agreeing."""
    examples = [
        move(mover="white", target="black", legal=True),
        move(mover="black", target="white", legal=True),
        move(mover="white", target="white", legal=False),
    ]

    found = RuleDeducer().deduce(examples)

    assert any(one.condition == ("mover", "!= reading", "target") for one in found)


def test_readings_of_different_kinds_are_never_compared():
    """Python holds that False is 0, so a reading saying whether two cells line up would otherwise match one
    counting what stands between them, and the nonsense would survive every test."""
    examples = [
        move(lines_up=False, between=0, legal=True),
        move(lines_up=False, between=0, legal=True),
        move(lines_up=True, between=1, legal=False),
    ]

    found = RuleDeducer().deduce(examples)

    assert not any(one.condition[1] in ("== reading", "!= reading") for one in found)


def test_a_condition_holding_only_for_actions_of_one_kind_is_found_within_that_kind():
    """What a knight may do says nothing about a bishop."""
    examples = [
        move(piece="knight", distance=2, legal=True),
        move(piece="bishop", distance=5, legal=True),
        move(piece="knight", distance=5, legal=False),
    ]

    found = RuleDeducer().deduce(examples, given=("piece",))

    within = [one for one in found if one.given]
    assert within, "something was found to hold only within a kind"
    assert any(one.readable.startswith("where piece == 'knight'") for one in within)


def test_the_rules_chosen_together_refuse_what_the_game_refuses():
    deducer = RuleDeducer()
    examples = a_game()

    chosen = deducer.conjunction(deducer.deduce(examples), examples)

    allowed, missed, wrong = deducer.scored(chosen, examples)
    assert (allowed, missed, wrong) == (2, 0, 0)


def test_a_rule_costing_no_legal_action_is_taken_over_one_that_costs_some():
    """A rule refusing a tenth of the legal moves to rule out a handful of illegal ones has made OMF a worse
    player, not a better-informed one."""
    deducer = RuleDeducer()
    examples = a_game()

    chosen = deducer.conjunction(deducer.deduce(examples), examples)

    assert all(deducer.allows(chosen, readings) for readings, legal in examples if legal)


def test_what_no_rule_can_refuse_is_reported_rather_than_papered_over():
    """Either a reading is missing, or what refuses them isn't about the action at all."""
    deducer = RuleDeducer()
    examples = [
        move(distance=1, legal=True),
        move(distance=1, legal=False),
    ]

    deducer.conjunction(deducer.deduce(examples), examples)

    assert len(deducer.unexplained) == 1, "the two read alike and one was refused"


def test_both_ways_of_being_wrong_are_counted():
    """Rules that only ever had to agree with legal actions can refuse none of them and still be useless — they
    let everything else through too."""
    deducer = RuleDeducer()
    examples = a_game()

    allowed, missed, wrong = deducer.scored((), examples)

    assert (allowed, missed, wrong) == (2, 0, 3), "no rules at all lets every illegal one through"


def test_a_rule_a_legal_action_contradicts_is_refuted():
    """One legal action that doesn't satisfy it is enough, which is why the positions worth building are the
    ones where it might not hold."""
    deducer = RuleDeducer()
    rule = Deduced(("distance", "==", 1), excluded=3, covered=5)

    standing, surprises = deducer.refuted([rule], [move(distance=2, legal=True)])

    assert standing == ()
    assert len(surprises) == 1


def test_a_rule_nothing_contradicts_stands_and_counts_what_it_has_now_covered():
    deducer = RuleDeducer()
    rule = Deduced(("distance", "==", 1), excluded=3, covered=5)

    standing, surprises = deducer.refuted([rule], [move(distance=1, legal=True), move(distance=1, legal=True)])

    assert surprises == ()
    assert standing[0].covered == 7


def test_a_rule_that_says_nothing_about_these_actions_is_neither_broken_nor_credited():
    """A rule holding only where something else does says nothing about a candidate that something else doesn't
    hold of."""
    deducer = RuleDeducer()
    rule = Deduced(("distance", "==", 2), excluded=1, covered=5, given=(("piece", "==", "knight"),))

    standing, surprises = deducer.refuted([rule], [move(piece="bishop", distance=5, legal=True)])

    assert surprises == ()
    assert standing[0].covered == 5, "no bishop move counted toward a knight rule"


def test_an_action_of_a_kind_never_considered_is_a_discovery_rather_than_a_refutation():
    """Castling, a pawn becoming a queen — no candidate can refute it, because none was ever offered."""
    deducer = RuleDeducer()

    found = deducer.unforeseen([Action("move", (("to", 1),))], [Action("castle", (("side", "king"),))], seen=1000)

    assert [one.name for one in found] == ["castle"]
    assert "of a kind never considered" in deducer.surprises[0].rule


def test_an_action_carrying_a_value_never_seen_is_said_to_be_that_rather_than_a_new_kind():
    deducer = RuleDeducer()

    deducer.unforeseen([Action("move", (("to", 1),))], [Action("move", (("to", 9),))], seen=1000)

    assert "carrying a value never seen" in deducer.surprises[0].rule


def test_an_action_already_among_the_candidates_is_no_discovery():
    deducer = RuleDeducer()
    known = Action("move", (("to", 1),))

    assert deducer.unforeseen([known], [known]) == ()
    assert deducer.surprises == ()
