from openmind.doxastic.constant.doxastic_constant import PROVED, TOLD
from openmind.doxastic.model.belief import Belief
from openmind.doxastic.model.claim import Claim
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.record import Record

CLAIM = Claim("the bishop pair wins the ending")


def new_record(source: str, strength: float, supports: bool) -> Record:
    return Record("as it was said", Provenance(source), claim=CLAIM, supports=supports, strength=strength)


def test_a_claim_with_no_record_is_neither_believed_nor_disbelieved_and_wholly_uncertain() -> None:
    belief = Belief(CLAIM)

    assert (belief.belief, belief.disbelief, belief.uncertainty, belief.contested, belief.net) == (0.0, 0.0, 1.0, 0.0, 0.0)


def test_weak_records_add_up_without_ever_reaching_certainty() -> None:
    belief = Belief(CLAIM, tuple(new_record(TOLD, 0.5, True) for _ in range(3)))

    assert belief.belief == 1.0 - 0.5**3
    assert belief.uncertainty == 1.0 - belief.belief


def test_one_conclusive_record_settles_its_side() -> None:
    belief = Belief(CLAIM, (new_record(PROVED, 1.0, True),))

    assert (belief.belief, belief.uncertainty) == (1.0, 0.0)


def test_poor_evidence_for_and_solid_evidence_against_are_kept_apart() -> None:
    belief = Belief(CLAIM, (new_record(TOLD, 0.2, True),), (new_record(PROVED, 0.9, False),))

    assert (round(belief.belief, 10), round(belief.disbelief, 10)) == (0.2, 0.9)
    assert round(belief.net, 10) == -0.7
    assert (belief.uncertainty, round(belief.contested, 10)) == (0.0, 0.2)


def test_each_kind_of_source_shows_what_it_carries_on_each_side() -> None:
    belief = Belief(
        CLAIM,
        (new_record(TOLD, 0.4, True), new_record(TOLD, 0.4, True)),
        (new_record(PROVED, 0.8, False), new_record(TOLD, 0.5, False)),
    )

    assert belief.by_source == (
        (PROVED, 0.0, 0.8),
        (TOLD, 1.0 - 0.6**2, 0.5),
    )


def test_a_strength_outside_zero_to_one_is_held_to_it() -> None:
    assert Belief(CLAIM, (new_record(PROVED, 5.0, True),)).belief == 1.0
    assert Belief(CLAIM, (new_record(TOLD, -1.0, True),)).belief == 0.0
