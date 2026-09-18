from dataclasses import replace
from pathlib import Path

from openmind.dashboard.mapper.dashboard_html_mapper import DashboardHtmlMapper
from openmind.dashboard.model.dashboard_snapshot import DashboardSnapshot
from openmind.dashboard.model.game_listing import GameListing
from openmind.dashboard.model.game_view import GameView
from openmind.dashboard.model.log_progress import LogProgress
from openmind.dashboard.model.machine_status import MachineStatus
from openmind.dashboard.model.model_score import ModelScore
from openmind.dashboard.model.process_status import ProcessStatus
from openmind.dashboard.model.report_summary import ReportSummary
from openmind.dashboard.model.round_row import RoundRow

GB = 1024**3


def snapshot() -> DashboardSnapshot:
    rounds = (RoundRow(1, 1, 0.6931, 0.0026, (("random", "3 / 14 / 3"),), None, 14400.0),)
    return DashboardSnapshot(
        "chess",
        "2026-09-15 09:30:00",
        ReportSummary(Path("data/training/chess/run.json"), "2026-09-15T08:54:52", False, rounds, (("here.color[1, 5] == other", -0.06),)),
        LogProgress(Path("data/log/train-values/chess/run.log"), 2, 100000, "self-play valuing positions with round 1's rules", 37, 11, 4213, 5, ("INFO value_distiller: <Distilled>",), 30, 7),
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
        "Drawn self-play games this round<b>30</b>",
        "Decisive self-play games this round<b>7</b>",
        "Games against opponents this round<b>11</b>",
        "Moves searched this round<b>4213</b>",
        "Moves deduced this round<b>5</b>",
        "running, 2 h 01 min",
        "5.0 of 62.0 GB (8%)",
        "class='warn'",
        "<th>against random</th>",
        "<td>3 / 14 / 3</td>",
        "<th>time</th>",
        "<summary>What the columns mean</summary>",
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


def test_the_page_shows_every_model_s_games_under_a_header() -> None:
    models = (ModelScore("win", "0123456789abcdef", 20, 5, 13, 2, "2026-09-16 14:52:53"),)
    page = DashboardHtmlMapper().to_html(replace(snapshot(), models=models), 30)

    assert "<h2>Models</h2>" in page
    assert "<th>model</th><th>id</th><th>games</th><th>wins</th><th>draws</th><th>losses</th><th>score</th><th>last game</th>" in page
    assert "<td>win</td><td>0123456789abcdef</td><td>20</td><td>5</td><td>13</td><td>2</td><td>0.575</td><td>2026-09-16 14:52:53</td>" in page


def test_a_page_without_models_has_no_models_table() -> None:
    assert "<h2>Models</h2>" not in DashboardHtmlMapper().to_html(snapshot(), 30)


def test_the_page_shows_the_latest_decisive_game_with_its_record_and_buttons_to_step_through_its_moves() -> None:
    game = GameView(
        "arms game 212",
        "2026-09-16 23:41:00",
        (("white", "win"), ("black", "weighted")),
        (1.0, 0.0),
        "checkmate",
        '[Result "1-0"] 1. f3 e5 1-0',
        ("f2f3", "e7e5"),
        ("<svg>start</svg>", "<svg>f3</svg>", "<svg>e5</svg>"),
        True,
        "000012",
        "000009",
        None,
    )

    page = DashboardHtmlMapper().to_html(replace(snapshot(), latest_game=game, decisive_games=5), 30)

    assert "<h2>Latest decisive game</h2>" in page
    assert "arms game 212, ended 2026-09-16 23:41:00: win (white) 1, weighted (black) 0 by checkmate" in page
    assert "<div id='position'><svg>start</svg></div>" in page
    assert "<pre class='record'>[Result &quot;1-0&quot;] 1. f3 e5 1-0</pre>" in page
    assert all(f"<button id='{name}'" in page for name in ("first", "previous", "next", "last"))
    assert '"moves": ["f2f3", "e7e5"]' in page and "<\\/svg>" in page
    assert "<a href='/games'>All decisive games (5)</a>" in page

    single = DashboardHtmlMapper().game_page("chess", game, 30)
    assert "<a href='/game/000009'>Previous decisive game</a>" in single and "Next decisive game" not in single
    assert "<div id='position'><svg>start</svg></div>" in single


def test_a_page_without_a_decisive_game_has_no_game_section() -> None:
    assert "Latest decisive game" not in DashboardHtmlMapper().to_html(snapshot(), 30)


def test_the_list_of_decisive_games_links_each_to_its_page_under_a_header() -> None:
    games = (
        GameListing("000012", "arms game 212", "2026-09-16 23:41:00", (("white", "losing"), ("black", "fork")), (1.0, 0.0), "checkmate", 48),
        GameListing("000009", "arms game 207", "2026-09-16 23:30:00", (("white", "fork"), ("black", "losing")), (0.0, 1.0), None, 30),
    )

    page = DashboardHtmlMapper().games_page("chess", games, 30)

    assert "<h2>Decisive games (2)</h2>" in page
    assert "<th>game</th><th>ended</th><th>white</th><th>black</th><th>payoffs</th><th>ending</th><th>plies</th>" in page
    assert "<td><a href='/game/000012'>arms game 212</a></td><td>2026-09-16 23:41:00</td><td>losing</td><td>fork</td><td>1 0</td><td>checkmate</td><td>48</td>" in page
    assert "<td>none</td><td>30</td>" in page
    assert "No decisive game yet." in DashboardHtmlMapper().games_page("chess", (), 30)
