from pathlib import Path

from openmind.doxastic.constant.doxastic_constant import PROVED, TOLD
from openmind.doxastic.constant.rule_kind_constant import CONSTRAINT, POSITION
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.rule_record import RuleRecord
from openmind.doxastic.service.file_rule_store import FileRuleStore
from openmind.rbs.model.python_rule import PythonRule


def new_rule(
    name: str, source: str = "True", kind: str = CONSTRAINT, rule_id: str = "r000001", weight: float = 1.0
) -> RuleRecord:
    return RuleRecord(name, kind, PythonRule(source), Provenance(TOLD), (("tictactoe", weight),), id=rule_id)


def test_a_rule_is_read_back_from_the_place_it_was_written_at(tmp_path: Path) -> None:
    store = FileRuleStore(tmp_path / "rules.jsonl")
    first = store.append(new_rule("a cell is played only when it is empty", "cell[row, col] is None"))
    second = store.append(new_rule("the turn passes", 'turn = "O" if turn == "X" else "X"', rule_id="r000002"))

    assert store.read(first).rule == PythonRule("cell[row, col] is None")
    assert store.read(second).name == "the turn passes"


def test_loading_gives_every_rule_with_its_place_in_the_order_they_were_declared(tmp_path: Path) -> None:
    store = FileRuleStore(tmp_path / "rules.jsonl")
    store.append(new_rule("first", rule_id="r000001"))
    store.append(new_rule("second", rule_id="r000002"))

    loaded = list(store.load())

    assert [rule.name for _, rule in loaded] == ["first", "second"]
    assert [store.read(place).name for place, _ in loaded] == ["first", "second"]


def test_declaring_a_rule_again_keeps_the_newest_weight_and_leaves_the_line_it_was_on(tmp_path: Path) -> None:
    path = tmp_path / "rules.jsonl"
    store = FileRuleStore(path)
    store.append(new_rule("more moves is better placed", "len(here.moves(me))", POSITION, weight=0.4))
    store.append(new_rule("more moves is better placed", "len(here.moves(me))", POSITION, weight=0.9))

    ((_, rule),) = list(store.load())

    assert rule.weight("tictactoe") == 0.9
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_a_forgotten_rule_is_left_out_of_what_loads_and_the_file_says_so(tmp_path: Path) -> None:
    path = tmp_path / "rules.jsonl"
    store = FileRuleStore(path)
    store.append(new_rule("kept", rule_id="r000001"))
    store.append(new_rule("dropped", rule_id="r000002"))

    store.forget("r000002")

    assert [rule.name for _, rule in store.load()] == ["kept"]
    assert '"forgotten":"r000002"' in path.read_text(encoding="utf-8")


def test_loading_a_file_that_was_never_written_gives_nothing(tmp_path: Path) -> None:
    assert list(FileRuleStore(tmp_path / "rules.jsonl").load()) == []


def test_a_rule_keeps_what_it_is_a_rule_about_and_where_it_came_from(tmp_path: Path) -> None:
    store = FileRuleStore(tmp_path / "rules.jsonl")
    declared = RuleRecord(
        "a queen moves like a rook or a bishop",
        CONSTRAINT,
        PythonRule("straight or diagonal"),
        Provenance(PROVED, game="0031"),
        (("chess", 1.0),),
        action="move",
        parameter=None,
        id="r000007",
    )
    place = store.append(declared)

    assert store.read(place) == declared
