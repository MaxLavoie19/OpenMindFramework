import logging
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from openmind.doxastic.constant.doxastic_constant import COUNTED, PLAYED, PROVED, RELAXED, SEEN, TOLD
from openmind.doxastic.constant.rule_kind_constant import CONSTRAINT, MOVE, POSITION
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.doxastic.model.claim import Claim
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.record import Record
from openmind.doxastic.model.rule_record import RuleRecord
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.model.python_rule import PythonRule

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
    assert any("Knowledge of cheat: 2 records remembered, 0 rules declared" == message for message in caplog.messages)


def new_rule(name: str, kind: str = POSITION, source: str = "True", *contexts: tuple[str, float]) -> RuleRecord:
    return RuleRecord(name, kind, PythonRule(source), Provenance(PROVED), contexts or (("cheat", 1.0),))


def test_a_declared_rule_is_given_an_id_and_a_time_and_is_found_again_by_it(tmp_path: Path) -> None:
    base = new_base(tmp_path)

    declared = base.declare(new_rule("a player holding no spade never leads one"))

    assert declared.id == "r000001"
    assert declared.provenance.when is not None
    assert base.rule("r000001") == declared
    assert base.rule("r000009") is None


def test_the_rules_retrieved_are_those_relevant_to_the_context_the_heaviest_first(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    base.declare(new_rule("light here", POSITION, "1", ("cheat", 0.2)))
    base.declare(new_rule("heavy here", POSITION, "2", ("cheat", 0.9)))
    base.declare(new_rule("for another game", POSITION, "3", ("chess", 1.0)))

    assert [rule.name for rule in base.rules("cheat")] == ["heavy here", "light here"]
    assert [rule.name for rule in base.rules("chess")] == ["for another game"]
    assert base.rules("go") == ()


def test_the_same_rule_carries_its_own_weight_in_each_context_it_bears_on(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    base.declare(new_rule("more moves is better placed", POSITION, "len(here.moves(me))", ("cheat", 0.3), ("chess", 0.8)))

    ((in_cheat,), (in_chess,)) = base.rules("cheat"), base.rules("chess")

    assert in_cheat is in_chess
    assert (in_cheat.weight("cheat"), in_chess.weight("chess")) == (0.3, 0.8)


def test_rules_are_retrieved_by_kind_where_a_kind_is_asked_for(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    base.declare(new_rule("a cell is played only when it is empty", CONSTRAINT))
    base.declare(new_rule("more moves is better placed", POSITION))
    base.declare(new_rule("a move taking a piece is worth looking at", MOVE))

    assert [rule.kind for rule in base.rules("cheat", (CONSTRAINT,))] == [CONSTRAINT]
    assert {rule.kind for rule in base.rules("cheat", (POSITION, MOVE))} == {POSITION, MOVE}


def test_declaring_a_rule_again_under_its_id_changes_what_it_weighs(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    declared = base.declare(new_rule("more moves is better placed", POSITION, "1", ("cheat", 0.3)))

    base.declare(replace(declared, contexts=(("cheat", 0.9),)))

    ((only,),) = (base.rules("cheat"),)
    assert (only.id, only.weight("cheat")) == (declared.id, 0.9)


def test_the_rules_declared_come_back_when_the_base_is_opened_again(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    caplog.set_level(logging.INFO, logger="openmind.doxastic")
    base = new_base(tmp_path)
    base.declare(new_rule("a cell is played only when it is empty", CONSTRAINT))
    base.declare(new_rule("more moves is better placed", POSITION))

    opened = new_base(tmp_path)

    assert [rule.name for rule in opened.rules("cheat")] == [
        "a cell is played only when it is empty",
        "more moves is better placed",
    ]
    assert opened.declare(new_rule("the next one")).id == "r000003"
    assert any("Knowledge of cheat: 0 records remembered, 2 rules declared" == message for message in caplog.messages)


def test_an_undeclared_rule_is_retrieved_no_more_here_or_in_the_base_opened_again(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    kept = base.declare(new_rule("kept", POSITION, "1"))
    dropped = base.declare(new_rule("dropped", POSITION, "2"))

    base.undeclare(dropped.id)

    assert [rule.name for rule in base.rules("cheat")] == [kept.name]
    assert [rule.name for rule in new_base(tmp_path).rules("cheat")] == [kept.name]
