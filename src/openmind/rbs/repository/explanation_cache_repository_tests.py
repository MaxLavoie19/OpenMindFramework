from pathlib import Path

from openmind.rbs.repository.explanation_cache_repository import ExplanationCacheRepository


def test_entries_are_saved_by_source_and_loaded_back(tmp_path: Path) -> None:
    repository = ExplanationCacheRepository()
    path = repository.path(tmp_path, "tictactoe/fourinarow", "qwen3:8b")
    entries = {"here.mobility(me)": ("the number of moves I could make", "How many moves I have.")}

    assert repository.load(path) == {}
    assert repository.save(entries, path) == tmp_path / "tictactoe_fourinarow" / "qwen3_8b.json"
    assert repository.load(path) == entries
