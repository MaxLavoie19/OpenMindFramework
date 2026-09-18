from openmind.doxastic.constant.doxastic_constant import PROVED, SEEN, TOLD
from openmind.doxastic.model.claim import Claim
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.record import Record
from openmind.doxastic.service.evidence_weigher import EvidenceWeigher

CLAIM = Claim("black holds no spade", about=("black",))


def new_record(source: str, claim: Claim | None, supports: bool | None, strength: float = 1.0) -> Record:
    return Record("as said", Provenance(source), claim=claim, supports=supports, strength=strength)


def test_the_records_bearing_on_the_claim_are_split_by_which_way_they_go() -> None:
    records = (
        new_record(TOLD, CLAIM, True, 0.3),
        new_record(SEEN, CLAIM, False, 0.9),
        new_record(PROVED, CLAIM, True, 0.5),
    )

    belief = EvidenceWeigher().weigh(CLAIM, records)

    assert [record.provenance.source for record in belief.supporting] == [TOLD, PROVED]
    assert [record.provenance.source for record in belief.opposing] == [SEEN]


def test_a_record_that_only_remembers_something_weighs_on_neither_side() -> None:
    records = (new_record(SEEN, None, None), new_record(SEEN, CLAIM, None))

    belief = EvidenceWeigher().weigh(CLAIM, records)

    assert (belief.supporting, belief.opposing, belief.uncertainty) == ((), (), 1.0)


def test_another_claim_s_records_are_left_out_even_where_the_name_is_the_same() -> None:
    elsewhere = Claim("black holds no spade", about=("white",))
    believed = Claim("black holds no spade", about=("black",), holder=("white",))

    belief = EvidenceWeigher().weigh(CLAIM, (new_record(TOLD, elsewhere, True), new_record(TOLD, believed, True)))

    assert (belief.supporting, belief.opposing) == ((), ())


def test_a_claim_is_weighed_whether_or_not_the_records_carry_the_rule_reading_it() -> None:
    from openmind.rbs.model.python_rule import PythonRule

    with_rule = Claim("black holds no spade", PythonRule("here.spades('black') == 0"), about=("black",))

    belief = EvidenceWeigher().weigh(CLAIM, (new_record(SEEN, with_rule, True, 0.8),))

    assert round(belief.belief, 10) == 0.8
