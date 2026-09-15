import html
from collections.abc import Sequence

from openmind.dashboard.model.dashboard_snapshot import DashboardSnapshot

GIGABYTE = 1024**3

STYLE = """
body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem 1.25rem; background: #f7f7f5; color: #222; }
h1 { font-size: 1.3rem; margin: 0 0 .25rem; } h2 { font-size: 1.05rem; margin: 1.5rem 0 .5rem; }
.muted { color: #666; font-size: .85rem; }
.cards { display: flex; flex-wrap: wrap; gap: .75rem; }
.card { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: .6rem .9rem; min-width: 9rem; }
.card b { display: block; font-size: 1.35rem; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; background: #fff; font-size: .85rem; }
th, td { border: 1px solid #ddd; padding: .3rem .5rem; text-align: left; white-space: nowrap; }
th { background: #eee; }
pre { background: #fff; border: 1px solid #ddd; padding: .6rem; overflow-x: auto; font-size: .8rem; }
.warn { color: #a33; }
.legend { font-size: .85rem; margin: .4rem 0; padding-left: 1.2rem; } .legend b { font-weight: 600; }
"""

#: What each column of the rounds table means, in the table's order.
ROUND_COLUMNS = (
    ("round", "the round's number; each round plays self-play games, fits value rules to them, then plays games against each opponent"),
    ("rules", "how many value rules the round's fit kept: weighted terms that value a position"),
    ("held-out loss", "how far the rules' values are from the targets of the held-out games' positions, as log loss; lower is better, 0.693 is what valuing every position at 0.5 gives"),
    ("held-out error", "the mean absolute difference between the rules' values and those targets"),
    ("against <opponent>", "the round's agent's wins / draws / losses against that opponent"),
    ("against the previous", "the same against the previous round's agent"),
    ("pondered", "positions of the round's training games that the previous rules valued worst, reasoned about by deduction before fitting"),
    ("proven", "pondered positions whose result deduction proved within its budget of plies and seconds; their targets become the proven payoffs, exact instead of estimated"),
    ("seeds", "candidate expressions induced from the proofs: the look-ahead that reaches the proven payoff, and the pattern of what the proven move changed; the expression search tries them before anything else"),
    ("seeds kept", "seeds the expression search kept: on every training row, the fit would give them a weight, their gradient being above the price times their clauses"),
    ("seeds in rules", "kept seeds that the chosen fit actually weighted, so they are among the round's value rules"),
    ("endings deduced", "positions of decisive training games deduced walking back from each game's end, before the positions missed most; a walk stops at the first position not proven"),
    ("endings proven", "of those, the positions proven: exact targets taken from won and lost games, and more seeds"),
    ("time", "how long the round took, as the report recorded it"),
)


class DashboardHtmlMapper:
    """Maps a snapshot to a page that reloads itself: the current round's progress, the machine and the training's
    processes, every finished round, the latest value rules, the latest notable log lines, and earlyoom's latest
    kills."""

    def to_html(self, snapshot: DashboardSnapshot, refresh_seconds: int) -> str:
        domain = html.escape(snapshot.domain)
        parts = [
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            f"<meta http-equiv='refresh' content='{refresh_seconds}'>",
            f"<title>{domain} training</title><style>{STYLE}</style></head><body>",
            f"<h1>{domain} training</h1>",
            f"<div class='muted'>Snapshot {html.escape(snapshot.taken_at)}, reloading every {refresh_seconds} seconds</div>",
            self._progress(snapshot),
            self._machine(snapshot),
            self._rounds(snapshot),
            self._rules(snapshot),
            self._arms(snapshot),
            self._recent(snapshot),
            "</body></html>",
        ]
        return "\n".join(parts)

    def _progress(self, snapshot: DashboardSnapshot) -> str:
        progress, report = snapshot.progress, snapshot.report
        processes = snapshot.machine.processes
        training = next((process for process in processes if process.role == "training"), None)
        workers = sum(1 for process in processes if process.role == "worker")
        running = "running" if training is not None else "not running"
        cards = [
            ("Training", f"{running}, {self._duration(training.seconds)}" if training else running),
            ("Workers", str(workers)),
        ]
        if progress is not None:
            round_text = "starting" if progress.round is None else f"{progress.round} of {progress.rounds}"
            cards.extend(
                (
                    ("Round", round_text),
                    ("Self-play games this round", str(progress.games)),
                    ("Drawn self-play games this round", str(progress.draws)),
                    ("Decisive self-play games this round", str(progress.decisive)),
                    ("Games against opponents this round", str(progress.matches)),
                    ("Moves searched this round", str(progress.searched)),
                    ("Moves deduced this round", str(progress.deduced)),
                )
            )
        if report is not None:
            cards.append(("Rounds saved", f"{len(report.rounds)}{', complete' if report.complete else ''}"))
        note = "" if progress is None else f"<div class='muted'>{html.escape(progress.round_note)} &middot; log {html.escape(str(progress.path))}</div>"
        command = "" if training is None else f"<div class='muted'>{html.escape(training.command)}</div>"
        body = "".join(f"<div class='card'>{html.escape(label)}<b>{html.escape(value)}</b></div>" for label, value in cards)
        return f"<h2>Now</h2><div class='cards'>{body}</div>{note}{command}"

    def _machine(self, snapshot: DashboardSnapshot) -> str:
        machine = snapshot.machine
        available = machine.memory_available / max(1, machine.memory_total)
        swap_free = machine.swap_free / max(1, machine.swap_total)
        warn = " class='warn'" if available < 0.15 else ""
        cards = (
            f"<div class='card'>Memory available<b{warn}>{machine.memory_available / GIGABYTE:.1f} of "
            f"{machine.memory_total / GIGABYTE:.1f} GB ({available:.0%})</b></div>"
            f"<div class='card'>Swap free<b>{machine.swap_free / GIGABYTE:.1f} of {machine.swap_total / GIGABYTE:.1f} GB "
            f"({swap_free:.0%})</b></div>"
            f"<div class='card'>Training processes' memory<b>"
            f"{sum(process.rss_bytes for process in machine.processes) / GIGABYTE:.1f} GB</b></div>"
        )
        rows = [
            (str(process.pid), process.role, f"{process.rss_bytes / GIGABYTE:.2f}", self._duration(process.seconds))
            for process in machine.processes
        ]
        table = self._table(("process", "role", "memory (GB)", "running for"), rows) if rows else "<p>No training process.</p>"
        kills = (
            f"<h2>earlyoom's latest kills</h2><pre>{html.escape(chr(10).join(machine.earlyoom))}</pre>"
            if machine.earlyoom
            else "<h2>earlyoom's latest kills</h2><p>None seen since the dashboard started.</p>"
        )
        return f"<h2>Machine</h2><div class='cards'>{cards}</div><details><summary>Processes</summary>{table}</details>{kills}"

    def _rounds(self, snapshot: DashboardSnapshot) -> str:
        report = snapshot.report
        if report is None or not report.rounds:
            return "<h2>Rounds</h2><p>No round saved yet.</p>"
        opponents = list(dict.fromkeys(opponent for item in report.rounds for opponent, _ in item.baselines))
        header = (
            "round",
            "rules",
            "held-out loss",
            "held-out error",
            *(f"against {opponent}" for opponent in opponents),
            "against the previous",
            "pondered",
            "proven",
            "seeds",
            "seeds kept",
            "seeds in rules",
            "endings deduced",
            "endings proven",
            "time",
        )
        rows = [
            (
                str(item.number),
                str(item.rules),
                "none" if item.held_out_loss is None else f"{item.held_out_loss:.6f}",
                "none" if item.held_out_error is None else f"{item.held_out_error:.4f}",
                *(dict(item.baselines).get(opponent, "none") for opponent in opponents),
                item.against_previous or "none",
                *(("none",) * 7 if item.pondering is None else tuple(str(count) for count in item.pondering)),
                self._duration(item.seconds),
            )
            for item in report.rounds
        ]
        started = f"<div class='muted'>Training started {html.escape(report.created_at)} &middot; report {html.escape(str(report.path))}</div>"
        legend = "".join(f"<li><b>{html.escape(name)}</b>: {html.escape(meaning)}</li>" for name, meaning in ROUND_COLUMNS)
        return f"<h2>Rounds</h2>{started}{self._table(header, rows)}<details open><summary>What the columns mean</summary><ul class='legend'>{legend}</ul></details>"

    def _rules(self, snapshot: DashboardSnapshot) -> str:
        report = snapshot.report
        if report is None or not report.rounds:
            return ""
        if not report.latest_rules:
            return "<h2>Latest round's rules</h2><p>No value rule: every position valued the same.</p>"
        rows = [(f"{weight:+.6g}", term) for term, weight in report.latest_rules]
        return f"<h2>Latest round's rules</h2>{self._table(('weight', 'term'), rows)}"

    def _arms(self, snapshot: DashboardSnapshot) -> str:
        report = snapshot.report
        if report is None or not report.latest_arms:
            return ""
        rows = [
            (
                name,
                str(agreements),
                str(disagreements),
                f"{accuracy:.3f}",
                f"{reliability:.3f}",
                str(games),
                str(wins),
                str(draws),
                str(losses),
                "none" if not games else f"{(wins + draws / 2) / games:.3f}",
            )
            for name, agreements, disagreements, accuracy, reliability, games, wins, draws, losses in report.latest_arms
        ]
        explanation = (
            "<div class='muted'>The signals the latest round followed: winning, the signals with the best records, and "
            "the uniform and weighted aggregations. Agreements are anchors, positions whose coming winner was known, where "
            "the winner read higher; disagreements, where the loser did; counted over every round. Reliability is "
            "2 × accuracy − 1, at least 0. Games, wins, draws and losses are self-play games an agent following the signal "
            "played against another signal's agent; score is points per game, a win 1 and a draw 0.5.</div>"
        )
        header = ("signal", "agreements", "disagreements", "accuracy", "reliability", "games", "wins", "draws", "losses", "score")
        return f"<h2>Latest round's signals</h2>{explanation}{self._table(header, rows)}"

    def _recent(self, snapshot: DashboardSnapshot) -> str:
        if snapshot.progress is None or not snapshot.progress.recent:
            return ""
        return f"<h2>Latest log lines</h2><pre>{html.escape(chr(10).join(snapshot.progress.recent))}</pre>"

    def _table(self, header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
        head = "".join(f"<th>{html.escape(cell)}</th>" for cell in header)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
        return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"

    def _duration(self, seconds: float) -> str:
        """A duration people read at a glance: 3 h 06 min, 24 min 13 s, or 45 s."""
        hours, rest = divmod(int(seconds), 3600)
        minutes, rest = divmod(rest, 60)
        if hours:
            return f"{hours} h {minutes:02d} min"
        if minutes:
            return f"{minutes} min {rest:02d} s"
        return f"{rest} s"
