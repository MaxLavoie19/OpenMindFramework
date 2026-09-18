from datetime import datetime

from openmind.doxastic.constant.doxastic_constant import COUNTED, TOLD
from openmind.doxastic.mapper.record_json_mapper import RecordJsonMapper
from openmind.doxastic.model.claim import Claim
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.record import Record
from openmind.rbs.model.python_rule import PythonRule


def test_a_record_comes_back_word_for_word_with_everything_it_carried() -> None:
    record = Record(
        'He said "I have no spades", and showed one later',
        Provenance(TOLD, "black", "8/8/8/8", "game-3", 2, 41, datetime(2026, 9, 15, 20, 30, 5)),
        ("black",),
        ("spades",),
        ("bluff", "claim"),
        Claim("black holds no spade", PythonRule("here.spades('black') == 0"), ("black",), ("black",)),
        supports=False,
        strength=0.8,
        count=1,
        id="000007",
    )
    mapper = RecordJsonMapper()

    line = mapper.to_line(record)

    assert "\n" not in line
    assert mapper.from_line(line) == record


def test_a_counted_record_keeps_its_count_and_a_record_without_a_claim_keeps_none() -> None:
    record = Record("options agreed with the winner", Provenance(COUNTED, round=4), count=1200, id="000001")
    mapper = RecordJsonMapper()

    assert mapper.from_line(mapper.to_line(record)) == record


def test_a_claim_whose_rule_is_a_function_names_it_and_comes_back_without_it() -> None:
    def trapped(state: object) -> bool:
        return True

    record = Record("the knight has nowhere to go", Provenance(TOLD, "white"), claim=Claim("trapped", trapped), supports=True, id="000002")
    mapper = RecordJsonMapper()

    line = mapper.to_line(record)
    read = mapper.from_line(line)

    assert "trapped" in line
    assert read.claim is not None and read.claim.rule is None and read.claim.name == "trapped"
