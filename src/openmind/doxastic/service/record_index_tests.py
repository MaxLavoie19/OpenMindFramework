from openmind.doxastic.constant.doxastic_constant import PROVED, TOLD
from openmind.doxastic.model.claim import Claim
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.record import Record
from openmind.doxastic.service.record_index import RecordIndex

CLAIM = Claim("black holds no spade", about=("black",))


def new_record(record_id: str, **fields: object) -> Record:
    provenance = fields.pop("provenance", Provenance(TOLD, "black"))
    return Record("as said", provenance, id=record_id, **fields)  # type: ignore[arg-type]


def test_a_record_is_found_by_subject_by_name_mentioned_and_by_keyword() -> None:
    index = RecordIndex()
    index.add(new_record("000001", subjects=("black",), names=("Spades",), keywords=("bluff",)), 0)
    index.add(new_record("000002", subjects=("white",), keywords=("endgame",)), 10)

    assert index.find(subject="black") == ("000001",)
    assert index.find(name="spades") == ("000001",)
    assert index.find(keyword="endgame") == ("000002",)


def test_a_lookup_naming_several_fields_gives_what_answers_them_all() -> None:
    index = RecordIndex()
    index.add(new_record("000001", subjects=("black",), keywords=("bluff",)), 0)
    index.add(new_record("000002", subjects=("black",), keywords=("endgame",)), 10)

    assert index.find(subject="black", keyword="bluff") == ("000001",)
    assert index.find(subject="black") == ("000001", "000002")
    assert index.find(subject="black", keyword="opening") == ()


def test_records_are_found_by_claim_by_who_told_them_and_by_kind_of_source() -> None:
    index = RecordIndex()
    index.add(new_record("000001", claim=CLAIM, supports=True), 0)
    index.add(new_record("000002", provenance=Provenance(PROVED)), 10)

    assert index.find(claim=CLAIM.key) == ("000001",)
    assert index.find(told="black") == ("000001",)
    assert index.find(source=PROVED) == ("000002",)


def test_a_lookup_naming_nothing_gives_every_record_in_the_order_they_were_remembered() -> None:
    index = RecordIndex()
    index.add(new_record("000002"), 10)
    index.add(new_record("000001"), 0)

    assert index.find() == ("000002", "000001") == index.ids


def test_attending_to_a_word_brings_up_what_it_is_a_subject_a_name_or_a_keyword_of() -> None:
    index = RecordIndex()
    index.add(new_record("000001", subjects=("Kriegspiel",)), 0)
    index.add(new_record("000002", names=("kriegspiel",)), 10)
    index.add(new_record("000003", keywords=("KRIEGSPIEL",)), 20)
    index.add(new_record("000004", subjects=("chess",)), 30)

    assert index.under("kriegspiel") == ("000001", "000002", "000003")


def test_the_index_says_where_a_record_is_and_forgets_it_when_it_is_removed() -> None:
    index = RecordIndex()
    record = new_record("000001", subjects=("black",))
    index.add(record, 42)

    assert (index.place("000001"), len(index)) == (42, 1)

    index.remove(record)

    assert (index.place("000001"), index.find(subject="black"), len(index)) == (None, (), 0)
