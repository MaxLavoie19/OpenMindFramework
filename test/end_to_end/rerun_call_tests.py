import pickle
from pathlib import Path

import pytest

from openmind.entrypoint.rerun_call import main


def test_a_pickled_call_runs_again_and_shows_the_lines_holding_memory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    diagnosis = tmp_path / "2026-09-15_10-00-00-worker-1.txt"
    diagnosis.write_text("A diagnosis\n", encoding="utf-8")
    with diagnosis.with_suffix(".pickle").open("wb") as file:
        pickle.dump((pow, (2, 5)), file)

    main([str(diagnosis), "--lines", "3", "--log-directory", str(tmp_path / "log")])

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("The call ended after ") and "traced memory peaked at " in lines[0]
    assert lines[1].split() == ["bytes", "blocks", "line"]
    assert len(lines) <= 5
    assert list((tmp_path / "log").glob("*.log"))


def test_a_path_without_a_pickled_call_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([str(tmp_path / "missing.txt")])
