from openmind.inference.model.case_index import CaseIndex
from openmind.inference.model.example import Example
from openmind.inference.model.pursuit import Pursuit
from openmind.inference.service.hypothesis_table import HypothesisTable
from openmind.inference.service.refusal_learner import RefusalLearner
from openmind.inference.service.sniper import Sniper
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant


def a_reading(name: str, value: str) -> Literal:
    return Literal(name, (Constant(value),))


def a_case(readings, refused: bool) -> Example:
    return Example(tuple(a_reading(name, value) for name, value in readings), refused, "a position")


class Ticking:
    """A clock that moves a fixed amount every time it is read, so a budget of seconds is a budget of reads."""

    def __init__(self, by: float = 0.0) -> None:
        self._at, self._by = 0.0, by

    def __call__(self) -> float:
        self._at += self._by
        return self._at


def sniping(clock=None) -> Sniper:
    return Sniper(RefusalLearner(), clock=clock or Ticking(), between_clocks=4)


def one_condition_refuses_it():
    """A candidate refused, and legal moves none of which is dark. One condition tells it from them."""
    wanted = a_case((("colour", "dark"), ("way", "sideways"), ("far", "one")), True)
    guard = CaseIndex((
        a_case((("colour", "light"), ("way", "sideways"), ("far", "one")), False),
        a_case((("colour", "light"), ("way", "ahead"), ("far", "two")), False),
    ))
    return wanted, guard


def two_conditions_refuse_it():
    """A candidate refused, and legal moves such that neither of its readings alone may be denied — each
    appears in something legal — while the pair appears in nothing legal."""
    wanted = a_case((("colour", "dark"), ("way", "ahead")), True)
    guard = CaseIndex((
        a_case((("colour", "dark"), ("way", "sideways")), False),
        a_case((("colour", "light"), ("way", "ahead")), False),
    ))
    return wanted, guard


def test_a_candidate_one_condition_accounts_for_is_finished_in_one_session():
    wanted, guard = one_condition_refuses_it()
    sniper = sniping()

    found = sniper.pursue(sniper.started(wanted), guard, HypothesisTable(), (wanted,), seconds=60.0)

    assert found.found is not None
    assert found.exhausted, "nothing slipped at one condition, so no longer body could beat it"
    assert [one.predicate for one in found.found.body] == ["colour"]


def test_a_candidate_needing_two_conditions_is_reached_although_neither_alone_may_be_said():
    """Each of its readings appears in something the game allows, so no one condition survives the guard. The
    pair appears in nothing legal, and no shorter search reaches it."""
    wanted, guard = two_conditions_refuse_it()
    sniper = sniping()

    found = sniper.pursue(sniper.started(wanted), guard, HypothesisTable(), (wanted,), seconds=60.0)

    assert found.found is not None
    assert {one.predicate for one in found.found.body} == {"colour", "way"}
    assert found.exhausted


def test_put_down_and_picked_up_it_reaches_the_same_body_as_in_one_sitting():
    """The whole point of being reentrant. A second a time must not give a different answer from a minute."""
    wanted, guard = two_conditions_refuse_it()
    at_once = sniping().pursue(
        sniping().started(wanted), guard, HypothesisTable(), (wanted,), seconds=60.0
    )

    sniper, table = sniping(clock=Ticking(by=0.1)), HypothesisTable()
    bit_by_bit = sniper.started(wanted)
    for _ in range(20):
        bit_by_bit = sniper.pursue(bit_by_bit, guard, table, (wanted,), seconds=0.3)

    assert bit_by_bit.exhausted
    assert bit_by_bit.found == at_once.found


def test_a_session_carries_on_from_where_the_last_one_stopped_rather_than_starting_again():
    wanted, guard = two_conditions_refuse_it()
    sniper, table = sniping(clock=Ticking(by=0.1)), HypothesisTable()

    first = sniper.pursue(sniper.started(wanted), guard, table, (wanted,), seconds=0.3)
    second = sniper.pursue(first, guard, table, (wanted,), seconds=0.3)

    assert (second.size, second.at) >= (first.size, first.at)
    assert second.seconds > first.seconds


def test_a_candidate_the_readings_cannot_tell_from_the_legal_moves_is_finished_with_nothing():
    """Stronger than running out of time: nothing the vocabulary can say distinguishes it, at any length. That
    is a reading it has not got."""
    wanted = a_case((("colour", "dark"), ("way", "ahead")), True)
    guard = CaseIndex((a_case((("colour", "dark"), ("way", "ahead")), False),))
    sniper = sniping()

    found = sniper.pursue(sniper.started(wanted), guard, HypothesisTable(), (wanted,), seconds=60.0)

    assert found.exhausted
    assert found.found is None


def test_a_pursuit_already_finished_is_not_searched_again():
    wanted, guard = one_condition_refuses_it()
    sniper = sniping()
    done = sniper.pursue(sniper.started(wanted), guard, HypothesisTable(), (wanted,), seconds=60.0)

    again = sniper.pursue(done, guard, HypothesisTable(), (wanted,), seconds=60.0)

    assert again is done


def test_of_two_bodies_that_refuse_it_the_one_refusing_more_of_the_pool_is_taken():
    """A body refusing this candidate and many others over many boards is a rule about the game."""
    wanted = a_case((("colour", "dark"), ("way", "ahead")), True)
    guard = CaseIndex((a_case((("colour", "light"), ("way", "sideways")), False),))
    narrow = a_case((("colour", "light"), ("way", "ahead")), True)
    sniper = sniping()

    found = sniper.pursue(
        sniper.started(wanted), guard, HypothesisTable(), (wanted, narrow, narrow, narrow), seconds=60.0
    )

    assert [one.predicate for one in found.found.body] == ["way"], "way refuses all four; colour refuses one"


def test_the_ranking_says_how_many_of_the_pool_each_body_refuses():
    sniper = sniping()
    wanted, guard = two_conditions_refuse_it()
    pool = (wanted, a_case((("colour", "dark"), ("way", "sideways")), True))
    found = sniper.pursue(sniper.started(wanted), guard, HypothesisTable(), pool, seconds=60.0)

    ranked = sniper.ranked([found.found], pool)

    assert ranked[0][1] == 1, "the pair refuses the candidate and not the other, which goes sideways"


def test_what_one_pursuit_tried_another_finds_already_answered():
    """The table is shared on purpose: what loses for one case is what the next finds already tested."""
    wanted, guard = two_conditions_refuse_it()
    other = a_case((("colour", "dark"), ("way", "ahead")), True)
    sniper, table = sniping(), HypothesisTable()
    sniper.pursue(sniper.started(wanted), guard, table, (wanted,), seconds=60.0)
    held, spared = len(table), table.spared

    sniper.pursue(sniper.started(other), guard, table, (other,), seconds=60.0)

    assert len(table) == held, "it learned of no body the first pursuit had not already tried"
    assert table.spared > spared, "and the table said so rather than testing them again"


def test_a_pursuit_carries_its_readings_so_resuming_means_the_same_places():
    wanted, _ = one_condition_refuses_it()
    sniper = sniping()

    started = sniper.started(wanted)

    assert started.offered, "tied once when it began"
    assert Pursuit(wanted, started.offered).offered == started.offered
