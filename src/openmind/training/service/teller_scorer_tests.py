"""What a heuristic put beside a teller comes to, and what it must never come to."""

import pytest

from openmind.training.service.teller_scorer import TellerScorer
from openmind.world.model.state import State


def a_position(number: int) -> tuple[State, str]:
    """A position that is only a number, because nothing here reads one. The scorer is handed a value per
    position by whoever is being judged, and never looks inside."""
    return State.of(**{"n": number}), "white"


def positions(how_many: int):
    return [a_position(one) for one in range(how_many)]


def valuing(by):
    """A heuristic as this sees one: a position and a player in, a number or nothing out."""
    return lambda node, player: by(node.value("n"))


def test_a_heuristic_ordering_positions_the_teller_s_way_tracks_it_completely():
    """The claim, in its simplest form: agreement is about the ordering and never about the units.

    The heuristic here says ten times what the teller says and shifts it by a hundred, which on raw values is
    a different number every time and on ranks is the same ordering exactly."""
    told = [float(one) for one in range(10)]

    found = TellerScorer().scored(positions(10), told, {"same order": valuing(lambda n: n * 10 + 100)})

    assert found[0].agreed == pytest.approx(1.0)
    assert found[0].decided == 10 and found[0].declined == 0


def test_a_heuristic_ordering_them_backwards_is_not_uninformed_but_wrong():
    """Below nought is worth telling from nought. A heuristic that knows nothing tracks nothing; one that
    orders positions backwards has an opinion and it is the opposite of the teller's."""
    told = [float(one) for one in range(10)]

    found = TellerScorer().scored(positions(10), told, {"backwards": valuing(lambda n: -n)})

    assert found[0].agreed == pytest.approx(-1.0)


def test_a_heuristic_that_says_the_same_of_everything_separates_nothing_rather_than_disagreeing():
    """Not firing is not the same as being wrong, which this project has a standing rule about. One number
    for every position is an opinion that orders nothing, so there is no ordering to agree with."""
    told = [float(one) for one in range(10)]

    found = TellerScorer().scored(positions(10), told, {"flat": valuing(lambda n: 1.0)})

    assert found[0].agreed == 0.0
    assert found[0].decided == 0 and found[0].undecided == 10


def test_positions_a_heuristic_knows_nothing_about_are_set_aside_and_counted():
    """A detector answers a handful and is right about them, which is the thing it is for. Declining is
    counted beside the measure so that coverage is read and never folded in."""
    told = [float(one) for one in range(10)]

    found = TellerScorer().scored(
        positions(10), told, {"only the last three": valuing(lambda n: None if n < 7 else n)}
    )

    assert found[0].agreed == pytest.approx(1.0)
    assert found[0].decided == 3 and found[0].declined == 7
    assert found[0].told == 10, "so three of ten is not read as three of three"


def test_ties_share_the_place_they_span_rather_than_being_put_in_some_order():
    """A heuristic saying the same of two positions has not ranked them, and a scorer that broke the tie by
    whichever came first would read an ordering it was never given.

    Pinned by swapping what the teller says of the two it tied: if the tie were being broken, the swap would
    move the number. It also cannot reach one, and should not — it agreed with every ordering it expressed and
    expressed less ordering than the teller, which is a real difference between them."""
    keeping = {"two pairs": valuing(lambda n: 0.0 if n < 2 else 1.0)}

    found = TellerScorer().scored(positions(4), [1.0, 2.0, 3.0, 4.0], keeping)
    swapped = TellerScorer().scored(positions(4), [2.0, 1.0, 4.0, 3.0], keeping)

    assert found[0].agreed == swapped[0].agreed, "within a tie there is no order to get right or wrong"
    assert 0.0 < found[0].agreed < 1.0


def test_a_value_that_is_not_a_number_is_knowing_nothing_rather_than_a_number():
    """A NaN let through correlates against everything and comes back NaN for the whole heuristic, which
    reads as a measurement rather than as the absence of one."""
    told = [float(one) for one in range(6)]

    found = TellerScorer().scored(
        positions(6), told, {"one blank": valuing(lambda n: float("nan") if n == 3 else n)}
    )

    assert found[0].agreed == pytest.approx(1.0)
    assert found[0].declined == 1


def test_being_handed_more_positions_than_tellings_is_refused_rather_than_zipped():
    """Two lists that must stay in step, and a quiet mismatch would score every heuristic against the wrong
    position from the mismatch onward."""
    with pytest.raises(ValueError):
        TellerScorer().scored(positions(4), [1.0, 2.0], {})


def test_every_heuristic_is_measured_over_the_same_positions_in_one_pass():
    """How fast a heuristic gathers evidence is not bounded by how often it is drawn to play — the same
    reason `AgreementScorer` replays a game for all of them at once."""
    told = [float(one) for one in range(8)]

    found = TellerScorer().scored(
        positions(8),
        told,
        {"with": valuing(lambda n: n), "against": valuing(lambda n: -n), "silent": valuing(lambda n: None)},
    )

    assert {one.holder for one in found} == {"with", "against", "silent"}
    assert [one.told for one in found] == [8, 8, 8]
