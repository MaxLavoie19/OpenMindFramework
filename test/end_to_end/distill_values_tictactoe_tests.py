import re
from pathlib import Path

import pytest

from openmind.entrypoint.distill_values import main
from openmind.rbs.mapper.value_base_json_mapper import ValueBaseJsonMapper
from openmind.rbs.repository.value_base_repository import ValueBaseRepository

pytestmark = pytest.mark.log_level("INFO")

SMALL = (
    *("--games", "3", "--held-out-games", "2", "--iterations", "20", "--seed", "1"),
    *("--pair-pool", "5", "--solo-limit", "1", "--prices", "0.1,0.01", "--max-steps", "200"),
)


def test_distill_values_prints_and_saves_value_rules(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(
        [
            "tictactoe",
            *SMALL,
            *("--workers", "1", "--log-directory", str(tmp_path / "log"), "--values-directory", str(tmp_path / "values")),
        ]
    )

    (values_file,) = (tmp_path / "values" / "tictactoe").glob("*.json")
    assert ValueBaseRepository(ValueBaseJsonMapper()).load(values_file).domain == "tictactoe"
    output = capsys.readouterr().out
    assert output.startswith("bias ")
    assert re.search(
        r"\nValue rules: \d+ of \d+ candidate terms, chosen at price (0\.1|0\.01); payoffs from 0\.0 to 1\.0\n", output
    )
    assert "\nprice  terms kept  steps  settled  training loss  held-out loss\n" in output
    assert re.search(
        r"\nRows: \d+ for training, \d+ held out, valued at the outcome target; mean absolute error on held-out rows: "
        r"0\.\d+\n",
        output,
    )
    assert output.endswith(f"Saved values {values_file}\n")
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    assert log_file.read_text(encoding="utf-8").splitlines()[-1] == (
        f"INFO  openmind.entrypoint.distill_values Saved values {values_file}"
    )


def test_several_workers_distill_the_same_value_rules(tmp_path: Path) -> None:
    value_bases = []
    for workers in ("1", "2"):
        main(
            [
                "tictactoe",
                *SMALL,
                *("--workers", workers, "--log-directory", str(tmp_path / workers / "log")),
                *("--values-directory", str(tmp_path / workers / "values")),
            ]
        )
        (values_file,) = (tmp_path / workers / "values" / "tictactoe").glob("*.json")
        value_bases.append(ValueBaseRepository(ValueBaseJsonMapper()).load(values_file))

    assert value_bases[0] == value_bases[1]


def test_negative_prices_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--prices", "0.1,-1", "--log-directory", str(tmp_path)])
