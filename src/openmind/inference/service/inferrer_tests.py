from openmind.inference.model.fact import Fact
from openmind.inference.service.covering_learner import Covering
from openmind.inference.service.inferrer import (
    ACTS_ONCE,
    AT_ONCE,
    CANNOT_ALL_ESCAPE,
    CAN_BRING_ABOUT,
    REACHES,
    TAKES_IN,
    THREATENS,
    WORTH_AT_LEAST,
    Inferrer,
)

CLEAR = ("things between source and target", "==", 0)
STRAIGHT = ("source and target share a row or a column", "==", True)
DIAGONAL = ("source and target are on a diagonal", "==", True)


def a_rule(*conditions):
    return Covering(tuple(conditions), 0, 0)


def chess_pieces():
    return {
        "queen": [a_rule(CLEAR, STRAIGHT), a_rule(CLEAR, DIAGONAL)],
        "rook": [a_rule(CLEAR, STRAIGHT)],
        "bishop": [a_rule(CLEAR, DIAGONAL)],
        "knight": [
            a_rule(("rows apart, source and target", "==", 1), ("columns apart, source and target", "==", 2)),
            a_rule(("rows apart, source and target", "==", 2), ("columns apart, source and target", "==", 1)),
        ],
        "king": [a_rule(("steps from source to target", "==", 1))],
    }


def said(facts, kind):
    return {one.about: one.held for one in facts if one.kind == kind}


def test_how_far_a_rule_reaches_is_read_off_it():
    facts = Inferrer().chain(chess_pieces(), (8, 8))
    reaches = said(facts, REACHES)

    assert reaches[("rook",)] == 14.0
    assert reaches[("bishop",)] == 8.75
    assert reaches[("queen",)] == 22.75
    assert reaches[("knight",)] == 5.25
    assert round(reaches[("king",)], 2) == 6.56


def test_what_one_thing_takes_in_is_read_off_the_rules():
    facts = Inferrer().chain(chess_pieces(), (8, 8))
    takes = said(facts, TAKES_IN)

    assert ("queen", "rook") in takes
    assert ("queen", "bishop") in takes
    assert ("rook", "bishop") not in takes


def test_a_worth_follows_from_a_reach_and_is_carried_by_what_takes_it_in():
    """The chain: a rook reaches fourteen squares, so a rook is worth at least fourteen; a queen takes in a rook,
    so a queen is worth at least fourteen too — concluded from two conclusions and no rule."""
    facts = Inferrer().chain(chess_pieces(), (8, 8))
    worth = said(facts, WORTH_AT_LEAST)

    assert worth[("rook",)] == 14.0
    assert worth[("queen",)] >= 14.0
    assert round(worth[("king",)], 2) == 6.56


def test_every_fact_says_what_it_rests_on():
    facts = Inferrer().chain(chess_pieces(), (8, 8))
    carried = [one for one in facts if one.kind == WORTH_AT_LEAST and one.from_facts]

    assert carried
    assert all(one.rests_on for one in carried)


def test_what_is_already_known_is_not_concluded_again():
    known = (Fact(WORTH_AT_LEAST, ("rook",), 100.0),)

    facts = Inferrer().chain(chess_pieces(), (8, 8), known)
    worth = [one.held for one in facts if one.kind == WORTH_AT_LEAST and one.about == ("rook",)]

    assert worth == [100.0]


def taking_consequences():
    """What a move does, as a game would declare it: it removes what stands where it lands, when that is another
    player's."""
    from openmind.predictor.model.consequence import Consequence
    from openmind.predictor.model.drawn import Column, Row

    taking = Covering((("color at target is another player", "==", True),), 0, 0)
    return (Consequence("Removed", "piece", (Row("source"), Column("target")), when=(taking,)),)


def test_what_an_action_brings_about_is_concluded_from_the_rules_that_allow_it():
    facts = Inferrer().chain(chess_pieces(), (8, 8), doing=taking_consequences())
    brought = {one.about for one in facts if one.kind == CAN_BRING_ABOUT}

    assert ("knight", "Removed", "piece") in brought
    assert ("queen", "Removed", "piece") in brought


def test_a_threat_is_worth_what_the_thing_threatened_is_worth():
    """Two conclusions held together: a knight can remove what it lands on, and a queen is worth 22.75."""
    facts = Inferrer().chain(chess_pieces(), (8, 8), doing=taking_consequences())
    threats = {one.about: one.held for one in facts if one.kind == THREATENS}

    assert threats[("knight", "queen")] == 22.75
    assert threats[("knight", "rook")] == 14.0


def test_what_a_thing_puts_at_stake_at_once_is_read_off_its_rule():
    """A knight bears on everything its rule reaches from where it stands, all at the same time."""
    facts = Inferrer().chain(chess_pieces(), (8, 8), doing=taking_consequences())
    at_once = {one.about: one.held for one in facts if one.kind == AT_ONCE}

    assert at_once[("knight",)] > 1
    assert at_once[("king",)] > 1


def test_a_fork_follows_from_that_and_from_a_player_acting_once():
    """Neither fact alone says anything: a thing bearing on several, and a turn answering one, together say one of
    them cannot be got out of the way."""
    known = (Fact(ACTS_ONCE, ()),)

    without = Inferrer().chain(chess_pieces(), (8, 8), doing=taking_consequences())
    with_it = Inferrer().chain(chess_pieces(), (8, 8), known, doing=taking_consequences())

    assert not [one for one in without if one.kind == CANNOT_ALL_ESCAPE]
    assert {one.about for one in with_it if one.kind == CANNOT_ALL_ESCAPE} >= {("knight",), ("queen",)}
