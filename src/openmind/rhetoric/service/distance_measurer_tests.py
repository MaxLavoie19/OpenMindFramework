import pytest

from openmind.rhetoric.constant.rhetoric_constant import AUDIENCE, IDENTITY
from openmind.rhetoric.model.ethos import Ethos
from openmind.rhetoric.model.pathos import Pathos
from openmind.rhetoric.model.position import Position
from openmind.rhetoric.model.speaker import Speaker
from openmind.rhetoric.service.distance_measurer import DistanceMeasurer

FISH = "is his fish disgusting"
PRICE = "is it dear"


def a_speaker(effective=(), projective=(), perceived=()) -> Speaker:
    return Speaker(
        "a speaker",
        Ethos(tuple(effective)),
        tuple((member, Ethos(tuple(positions))) for member, positions in projective),
        tuple((member, Pathos(tuple(positions))) for member, positions in perceived),
    )


def test_what_a_speaker_hides_is_the_distance_from_who_they_are_to_who_they_show():
    """Identity distance is the speaker's own position minus the one they project: a cook who thinks the fish
    disgusting and says it is fine is a whole two apart from themselves."""
    speaker = a_speaker(
        effective=[Position(FISH, 1.0, importance=1.0)],
        projective=[("a guest", [Position(FISH, -1.0)])],
    )

    (distance,) = DistanceMeasurer().distances(speaker)

    assert distance.kind == IDENTITY
    assert distance.value == pytest.approx(2.0)


def test_how_much_a_gap_matters_is_how_wide_it_is_times_how_much_the_question_does():
    """A speaker hiding something they care nothing about is not in the position of one hiding what defines
    them, and a measure that could not tell them apart would say they were."""
    indifferent = a_speaker(
        effective=[Position(FISH, 1.0, importance=0.1)], projective=[("a guest", [Position(FISH, -1.0)])]
    )
    committed = a_speaker(
        effective=[Position(FISH, 1.0, importance=1.0)], projective=[("a guest", [Position(FISH, -1.0)])]
    )
    measurer = DistanceMeasurer()

    assert measurer.distances(indifferent)[0].problematicity == pytest.approx(0.2)
    assert measurer.distances(committed)[0].problematicity == pytest.approx(2.0)


def test_a_position_that_does_not_say_how_much_it_matters_matters_fully():
    """A position heard in a message carries no importance — it stays with whoever holds it — and treating
    that as *matters not at all* would make every overheard disagreement unproblematic."""
    speaker = a_speaker(
        effective=[Position(FISH, 1.0)], projective=[("a guest", [Position(FISH, -1.0)])]
    )

    assert DistanceMeasurer().distances(speaker)[0].problematicity == pytest.approx(2.0)


def test_the_distance_to_the_audience_is_measured_from_what_was_shown_them():
    """The audience gap is between what the speaker projected and what they take the member to hold — not
    between the member and who the speaker really is, which the member has never seen."""
    speaker = a_speaker(
        effective=[Position(FISH, 1.0, importance=1.0)],
        projective=[("a guest", [Position(FISH, 0.0)])],
        perceived=[("a guest", [Position(FISH, -1.0, importance=1.0)])],
    )

    identity, audience = DistanceMeasurer().distances(speaker)

    assert identity.value == pytest.approx(1.0)
    assert audience.kind == AUDIENCE
    assert audience.value == pytest.approx(1.0)


def test_how_much_an_audience_gap_matters_is_the_members_own_importance():
    """Whose question it is decides whose importance counts: a speaker unbothered by something the listener
    cares about is in trouble, and the measure has to say so."""
    speaker = a_speaker(
        effective=[Position(FISH, 0.0, importance=0.0)],
        projective=[("a guest", [Position(FISH, 1.0, importance=0.0)])],
        perceived=[("a guest", [Position(FISH, -1.0, importance=1.0)])],
    )

    audience = [one for one in DistanceMeasurer().distances(speaker) if one.kind == AUDIENCE][0]

    assert audience.problematicity == pytest.approx(2.0)


def test_a_question_only_one_side_answers_is_not_a_distance():
    """Two positions are needed for a gap; one alone is a thing said into silence, and subtracting from
    nothing would invent an answer the other side never gave."""
    speaker = a_speaker(
        effective=[Position(FISH, 1.0), Position(PRICE, 1.0)],
        projective=[("a guest", [Position(FISH, 1.0)])],
    )

    found = DistanceMeasurer().distances(speaker)

    assert [one.question for one in found] == [FISH]


def test_a_member_the_speaker_has_no_reading_of_still_has_an_identity_distance():
    """What a speaker hides is about the speaker, so it holds whether or not they have any idea who they are
    talking to."""
    speaker = a_speaker(
        effective=[Position(FISH, 1.0)], projective=[("a stranger", [Position(FISH, -1.0)])]
    )

    found = DistanceMeasurer().distances(speaker)

    assert len(found) == 1 and found[0].kind == IDENTITY


def test_each_member_is_measured_apart_and_in_the_order_they_were_given():
    """A speaker shows themselves differently to different people, which is the whole point of a projective
    ethos, so the distances cannot be pooled."""
    speaker = a_speaker(
        effective=[Position(FISH, 1.0)],
        projective=[("a guest", [Position(FISH, 1.0)]), ("a critic", [Position(FISH, -1.0)])],
    )

    found = DistanceMeasurer().distances(speaker)

    assert [one.member for one in found] == ["a guest", "a critic"]
    assert found[0].value == pytest.approx(0.0)
    assert found[1].value == pytest.approx(2.0)


def test_a_speaker_saying_nothing_has_no_distances():
    assert DistanceMeasurer().distances(a_speaker()) == ()
