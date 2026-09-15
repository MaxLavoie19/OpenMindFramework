import json
from pathlib import Path

from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_rule import ValueRule
from openmind.rule.model.python_rule import PythonRule
from openmind.training.mapper.signal_library_json_mapper import SignalLibraryJsonMapper
from openmind.training.model.rule_support import RuleSupport
from openmind.training.model.signal import Signal
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_record import SignalRecord
from openmind.training.repository.signal_library_repository import SignalLibraryRepository

LIBRARY = SignalLibrary(
    "chess",
    (
        SignalRecord(Signal("win"), 40, 0),
        SignalRecord(
            Signal("pieces", PythonRule("sum(1 for at in here.color if here.color[at] == me)"), premises=("options",)), 7, 3, 4, 2, 1, 1
        ),
        SignalRecord(Signal("uniform", parts=("win", "pieces")), 10, 2),
    ),
    (RuleSupport(PythonRule("here.mobility(me)"), (("win", 0.4), ("pieces", 0.1))),),
    (("win", ValueBase("chess", 0.1, 0.0, 1.0, (ValueRule(PythonRule("here.mobility(me)"), 0.05),))),),
)


def test_a_library_goes_to_json_with_each_record_s_accuracy_and_reliability_and_comes_back() -> None:
    mapper = SignalLibraryJsonMapper()

    text = mapper.to_json(LIBRARY)

    pieces = json.loads(text)["records"][1]
    assert (pieces["accuracy"], round(pieces["reliability"], 6), pieces["source"]) == (
        0.7,
        0.4,
        "sum(1 for at in here.color if here.color[at] == me)",
    )
    assert mapper.from_json(text) == LIBRARY


def test_the_repository_writes_a_library_and_loads_it_back(tmp_path: Path) -> None:
    repository = SignalLibraryRepository(SignalLibraryJsonMapper())

    path = repository.write(LIBRARY, tmp_path / "signals" / "chess" / "run.json")

    assert repository.load(path) == LIBRARY
