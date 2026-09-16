import logging
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from openmind.doxastic.constant.doxastic_constant import COUNTED, PLAYED, PROVED, RELAXED, SEEN, TOLD
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.doxastic.model.claim import Claim
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.record import Record
from openmind.doxastic.service.knowledge_base import KnowledgeBase

pytestmark = pytest.mark.log_level("INFO")

NO_SPADE = Claim("black holds no spade", about=("black",))


def new_base(tmp_path: Path) -> KnowledgeBase:
    return create_knowledge_base("cheat", tmp_path)


def test_a_record_is_given_an_id_and_a_time_and_is_cited_back_word_for_word(tmp_path: Path) -> None:
    base = new_base(tmp_path)

    remembered = base.remember(Record('He said "I have no spades"', Provenance(TOLD, "black")))

    assert remembered.id == "000001"
    assert remembered.provenance.when is not None
    assert base.cite("000001") == remembered
    assert base.cite("000009") is None


def test_records_are_recalled_by_subject_by_name_by_keyword_by_teller_and_by_kind_of_source(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    base.remember(Record("black claimed two spades", Provenance(TOLD, "black"), ("black",), ("spades",), ("claim",)))
    base.remember(Record("white showed the ace", Provenance(SEEN), ("white",), ("ace",), ("card",)))

    assert [record.text for record in base.recall(subject="black")] == ["black claimed two spades"]
    assert [record.text for record in base.recall(name="ace")] == ["white showed the ace"]
    assert [record.text for record in base.recall(keyword="claim")] == ["black claimed two spades"]
    assert [record.text for record in base.recall(told="black")] == ["black claimed two spades"]
    assert [record.text for record in base.recall(source=SEEN)] == ["white showed the ace"]
    assert len(base.recall()) == 2


def test_records_are_recalled_from_one_position_from_a_time_on_and_the_newest_few(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    early = datetime(2026, 9, 15, 10, 0, 0)
    for number in range(4):
        base.remember(
            Record(
                f"ply {number}",
                Provenance(PLAYED, at="8/8/8/8" if number else "start", when=early + timedelta(minutes=number)),
                ("chess",),
            )
        )

    assert [record.text for record in base.recall(subject="chess", at="start")] == ["ply 0"]
    assert [record.text for record in base.recall(since=early + timedelta(minutes=2))] == ["ply 2", "ply 3"]
    assert [record.text for record in base.recall(limit=2)] == ["ply 2", "ply 3"]


def test_the_evidence_for_a_claim_and_the_evidence_against_it_are_weighed_apart(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    base.remember(Record('black said "no spades"', Provenance(TOLD, "black"), claim=NO_SPADE, supports=True, strength=0.3))
    base.remember(Record("black played the spade queen", Provenance(SEEN), claim=NO_SPADE, supports=False, strength=1.0))
    base.remember(Record("the deck was shuffled", Provenance(SEEN)))

    belief = base.weigh(NO_SPADE)

    assert (round(belief.belief, 10), belief.disbelief) == (0.3, 1.0)
    assert [record.provenance.source for record in belief.supporting] == [TOLD]
    assert [(source, round(for_it, 10), round(against, 10)) for source, for_it, against in belief.by_source] == [
        (SEEN, 0.0, 1.0),
        (TOLD, 0.3, 0.0),
    ]


def test_a_claim_another_player_holds_is_weighed_on_its_own(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    theirs = Claim("black holds no spade", about=("black",), holder=("white",))
    base.remember(Record("black said so to white", Provenance(TOLD, "black"), claim=theirs, supports=True, strength=0.6))
    base.remember(Record("black played the spade queen", Provenance(SEEN), claim=NO_SPADE, supports=False))

    assert round(base.weigh(theirs).belief, 10) == 0.6
    assert base.weigh(theirs).disbelief == 0.0
    assert {claim.key for claim in base.claims()} == {NO_SPADE.key, theirs.key}
    assert base.claims(holder=("white",)) == (theirs,)


def test_the_claims_held_both_ways_at_once_come_out_the_most_contested_first(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    settled = Claim("the king is mated")
    base.remember(Record("a proof", Provenance(PROVED), claim=settled, supports=True, strength=1.0))
    base.remember(Record("hearsay", Provenance(TOLD, "black"), claim=NO_SPADE, supports=True, strength=0.7))
    base.remember(Record("a card seen", Provenance(SEEN), claim=NO_SPADE, supports=False, strength=0.6))

    contested = base.contested()

    assert [belief.claim.key for belief in contested] == [NO_SPADE.key]
    assert round(contested[0].contested, 10) == 0.6
    assert base.contested(least=0.9) == ()


def test_a_counted_record_written_again_under_its_id_raises_its_count_without_a_second_record(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    counted = base.remember(Record("options agreed with the winner", Provenance(COUNTED, round=1), count=100))

    base.remember(replace(counted, count=250, provenance=replace(counted.provenance, round=2)))

    assert len(base) == 1
    (record,) = base.recall(source=COUNTED)
    assert (record.count, record.provenance.round) == (250, 2)


def test_what_was_remembered_is_still_there_for_a_base_opened_again_and_ids_carry_on(tmp_path: Path) -> None:
    first = new_base(tmp_path)
    first.remember(Record("black claimed two spades", Provenance(TOLD, "black"), ("black",), keywords=("claim",)))
    first.remember(Record("a proof in a relaxed game", Provenance(RELAXED), ("chess",), claim=NO_SPADE, supports=True, strength=0.4))

    opened = new_base(tmp_path)

    assert len(opened) == 2
    assert [record.text for record in opened.recall(subject="black")] == ["black claimed two spades"]
    assert round(opened.weigh(NO_SPADE).belief, 10) == 0.4
    assert opened.remember(Record("the next one", Provenance(SEEN))).id == "000003"


def test_a_forgotten_record_is_gone_from_the_lookups_the_belief_and_the_base_opened_again(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    kept = base.remember(Record("black played the spade queen", Provenance(SEEN), ("black",), claim=NO_SPADE, supports=False))
    dropped = base.remember(Record("a rumour", Provenance(TOLD, "white"), ("black",), claim=NO_SPADE, supports=True, strength=0.9))

    base.forget(dropped.id)

    assert [record.id for record in base.recall(subject="black")] == [kept.id]
    assert base.weigh(NO_SPADE).belief == 0.0
    assert len(new_base(tmp_path)) == 1


def test_attending_to_a_word_brings_what_is_known_about_it_into_context(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    caplog.set_level(logging.INFO, logger="openmind.doxastic")
    base = new_base(tmp_path)
    base.remember(Record("black claimed two spades", Provenance(TOLD, "black"), ("black",), ("spades",)))
    base.remember(Record("white showed the ace", Provenance(SEEN), ("white",)))
    opened = new_base(tmp_path)

    brought = opened.attend("black", "spades")

    assert brought == 1
    assert any("Attending to black, spades: 1 records, 1 in context" == message for message in caplog.messages)
    assert any("Knowledge of cheat: 2 records remembered" == message for message in caplog.messages)
