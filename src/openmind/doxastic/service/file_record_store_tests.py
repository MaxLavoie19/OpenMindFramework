from pathlib import Path

from openmind.doxastic.constant.doxastic_constant import COUNTED, SEEN, TOLD
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.record import Record
from openmind.doxastic.service.file_record_store import FileRecordStore


def new_record(text: str, source: str = TOLD, record_id: str = "000001", count: int = 1) -> Record:
    return Record(text, Provenance(source), id=record_id, count=count)


def test_a_record_is_read_back_word_for_word_from_the_place_it_was_written_at(tmp_path: Path) -> None:
    store = FileRecordStore(tmp_path / "records.jsonl")
    first = store.append(new_record("the first, said plainly", record_id="000001"))
    second = store.append(new_record("the second, with a comma, and a \"quote\"", SEEN, "000002"))

    assert store.read(first).text == "the first, said plainly"
    assert store.read(second).text == 'the second, with a comma, and a "quote"'


def test_loading_gives_every_record_with_its_place_in_the_order_they_were_remembered(tmp_path: Path) -> None:
    store = FileRecordStore(tmp_path / "records.jsonl")
    store.append(new_record("first", record_id="000001"))
    store.append(new_record("second", record_id="000002"))

    loaded = list(store.load())

    assert [record.text for _, record in loaded] == ["first", "second"]
    assert [store.read(place).text for place, _ in loaded] == ["first", "second"]


def test_writing_a_record_again_keeps_the_newest_of_it_and_leaves_the_line_it_was_on(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    store = FileRecordStore(path)
    store.append(new_record("options agreed with the winner", COUNTED, "000001", 100))
    store.append(new_record("options agreed with the winner", COUNTED, "000001", 250))

    ((_, record),) = list(store.load())

    assert record.count == 250
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_a_forgotten_record_is_left_out_of_what_loads_and_the_file_says_so(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    store = FileRecordStore(path)
    store.append(new_record("kept", record_id="000001"))
    store.append(new_record("dropped", record_id="000002"))

    store.forget("000002")

    assert [record.id for _, record in store.load()] == ["000001"]
    assert path.read_text(encoding="utf-8").splitlines()[-1] == '{"forgotten":"000002"}'


def test_a_store_whose_file_was_never_written_loads_nothing(tmp_path: Path) -> None:
    assert list(FileRecordStore(tmp_path / "nothing.jsonl").load()) == []
