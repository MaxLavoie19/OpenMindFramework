from pathlib import Path

from openmind.knowledge.service.file_store import FileStore


def test_the_last_entry_written_under_an_id_wins_in_the_order_ids_were_first_written(tmp_path: Path) -> None:
    store = FileStore(tmp_path / "beliefs.jsonl")
    store.append({"id": "b000001", "value": 3})
    store.append({"id": "b000002", "value": "x"})
    store.append({"id": "b000001", "value": 4})

    assert list(store.load()) == [{"id": "b000001", "value": 4}, {"id": "b000002", "value": "x"}]


def test_a_forgotten_entry_is_left_out_and_the_file_keeps_every_line(tmp_path: Path) -> None:
    path = tmp_path / "rules.jsonl"
    store = FileStore(path)
    store.append({"id": "r000001"})
    store.append({"id": "r000002"})

    store.forget("r000001")

    assert list(store.load()) == [{"id": "r000002"}]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_a_store_without_a_file_yet_loads_nothing(tmp_path: Path) -> None:
    assert list(FileStore(tmp_path / "none.jsonl").load()) == []
