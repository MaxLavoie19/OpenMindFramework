from pathlib import Path

from openmind.dashboard.mapper.dashboard_html_mapper import DashboardHtmlMapper
from openmind.dashboard.model.dashboard_snapshot import DashboardSnapshot
from openmind.dashboard.model.log_progress import LogProgress
from openmind.dashboard.model.machine_status import MachineStatus
from openmind.dashboard.model.process_status import ProcessStatus
from openmind.dashboard.model.report_summary import ReportSummary
from openmind.dashboard.model.round_row import RoundRow

GB = 1024**3


def snapshot() -> DashboardSnapshot:
    rounds = (RoundRow(1, 1, 0.6931, 0.0026, (("random", "3 / 14 / 3"),), None, (200, 48, 30, 1, 1), 14400.0),)
    return DashboardSnapshot(
        "chess",
        "2026-09-15 09:30:00",
        ReportSummary(Path("data/training/chess/run.json"), "2026-09-15T08:54:52", False, rounds, (("here.color[1, 5] == other", -0.06),)),
        LogProgress(Path("data/log/train-values/chess/run.log"), 2, 100000, "self-play valuing positions with round 1's rules", 37, 11, 4213, 5, ("INFO value_distiller: <Distilled>",)),
        MachineStatus(
            62 * GB,
            5 * GB,
            8 * GB,
            1 * GB,
            (ProcessStatus(11, "training", 3 * GB, 7260.0, "python -m openmind.entrypoint.train_values chess"), ProcessStatus(12, "worker", 2 * GB, 7200.0, "python")),
            ("earlyoom[1518]: sending SIGTERM to process 9544",),
        ),
    )


def test_the_page_reloads_itself_and_shows_the_progress_the_machine_the_rounds_and_the_rules() -> None:
    page = DashboardHtmlMapper().to_html(snapshot(), 30)

    assert "<meta http-equiv='refresh' content='30'>" in page
    for shown in (
        "2 of 100000",
        "Self-play games this round<b>37</b>",
        "Games against opponents this round<b>11</b>",
        "Moves searched this round<b>4213</b>",
        "Moves deduced this round<b>5</b>",
        "running, 2 h 01 min",
        "5.0 of 62.0 GB (8%)",
        "class='warn'",
        "<th>against random</th>",
        "<td>3 / 14 / 3</td>",
        "<th>pondered</th><th>proven</th><th>seeds</th><th>seeds kept</th><th>seeds in rules</th>",
        "<td>200</td><td>48</td><td>30</td><td>1</td><td>1</td>",
        "<th>time</th>",
        "<summary>What the columns mean</summary>",
        "<li><b>seeds kept</b>: seeds the expression search kept",
        "<td>4 h 00 min</td>",
        "here.color[1, 5] == other",
        "sending SIGTERM to process 9544",
    ):
        assert shown in page
    assert "&lt;Distilled&gt;" in page and "<Distilled>" not in page


def test_a_page_without_report_log_or_process_says_so() -> None:
    empty = DashboardSnapshot("chess", "2026-09-15 09:30:00", None, None, MachineStatus(62 * GB, 40 * GB, 8 * GB, 4 * GB, (), ()))

    page = DashboardHtmlMapper().to_html(empty, 30)

    assert "not running" in page and "No round saved yet." in page and "No training process." in page
