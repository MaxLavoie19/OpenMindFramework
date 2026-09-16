import json
from datetime import datetime
from pathlib import Path

import pytest

from openmind.entrypoint.train_values import main
from openmind.rbs.mapper.value_base_json_mapper import ValueBaseJsonMapper
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.repository.value_base_repository import ValueBaseRepository
from openmind.testing.service.log_reader import said
from openmind.training.mapper.signal_library_json_mapper import SignalLibraryJsonMapper
from openmind.training.repository.signal_library_repository import SignalLibraryRepository

pytestmark = pytest.mark.log_level("INFO")

SMALL = (
    *("--games", "2", "--held-out-games", "1", "--iterations", "10", "--seed", "1"),
    *("--seconds", "300", "--memory", "1", "--candidates", "1000", "--prices", "0.1,0.01", "--max-steps", "100"),
    *("--rollout-actions", "1", "--evaluation-games", "2", "--workers", "1"),
)


def directories(tmp_path: Path) -> tuple[str, ...]:
    return (
        *("--log-directory", str(tmp_path / "log"), "--values-directory", str(tmp_path / "values")),
        *("--report-directory", str(tmp_path / "training")),
    )


def test_with_the_signals_target_every_round_saves_the_signal_library_and_the_report_holds_the_arms(tmp_path: Path) -> None:
    signals = tmp_path / "signals"
    main(
        [
            "tictactoe",
            *("--rounds", "2", "--target", "signals", "--arms", "2", "--signals-directory", str(signals)),
            *SMALL,
            *directories(tmp_path),
        ]
    )

    (library_file,) = (signals / "tictactoe").glob("*.json")
    library = SignalLibraryRepository(SignalLibraryJsonMapper()).load(library_file)
    (report_file,) = (tmp_path / "training" / "tictactoe").glob("*.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert library.domain == "tictactoe" and any(record.signal.name == "win" for record in library.records)
    assert report["settings"]["signals"] == {"arms": 2, "horizon": 0, "goal_limit": 2}
    assert [arm["name"] for arm in report["rounds"][1]["arms"]][-2:] == ["uniform", "weighted"]

    main(["tictactoe", *("--rounds", "1", "--target", "signals", "--signal-library", str(library_file)), *SMALL, *directories(tmp_path)])


def test_a_signal_library_without_the_signals_target_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--signal-library", str(tmp_path / "missing.json"), *SMALL, *directories(tmp_path)])


def test_train_values_saves_every_round_and_the_report_and_prints_the_rounds(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    main(["tictactoe", "--rounds", "2", *SMALL, *directories(tmp_path)])

    (run,) = (tmp_path / "values" / "tictactoe").iterdir()
    assert sorted(path.name for path in run.iterdir()) == ["round-1.json", "round-1.md", "round-2.json", "round-2.md"]
    assert (run / "round-2.md").read_text(encoding="utf-8").startswith("# tictactoe value rules\n")
    assert ValueBaseRepository(ValueBaseJsonMapper()).load(run / "round-2.json").domain == "tictactoe"
    (report_file,) = (tmp_path / "training" / "tictactoe").glob("*.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert (report["complete"], [item["number"] for item in report["rounds"]]) == (True, [1, 2])
    output = capsys.readouterr().out
    assert output.startswith("Trained tictactoe value rules for 2 of 2 rounds, complete; round 1 started from no value rules\n")
    assert f"\nSaved round 2 values {run / 'round-2.json'}\n" in output
    assert output.endswith(f"Saved training report {report_file}\n")
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    lines = said(log_file)
    assert (
        "INFO  openmind.entrypoint.train_values Training in 1 worker processes, each holding at most 1073741824 bytes; "
        f"memory diagnoses in {tmp_path / 'log' / 'tictactoe' / 'memory'}"
    ) in lines
    assert lines[-1] == f"INFO  openmind.entrypoint.train_values Saved training report {report_file}"


def test_start_rules_of_another_domain_are_rejected(tmp_path: Path) -> None:
    start = ValueBaseRepository(ValueBaseJsonMapper()).save(
        ValueBase("sudoku", 0.0, 0.0, 1.0, ()), tmp_path / "values", datetime(2026, 9, 14, 13, 0, 0)
    )

    with pytest.raises(SystemExit):
        main(["tictactoe", "--start", str(start), *directories(tmp_path)])
