import html
import json
from collections.abc import Sequence

from openmind.dashboard.constant.dashboard_constant import RUN, TRAINING, WORKER
from openmind.dashboard.mapper.svg_chart_mapper import SvgChartMapper
from openmind.dashboard.model.dashboard_snapshot import DashboardSnapshot
from openmind.dashboard.model.game_listing import GameListing
from openmind.dashboard.model.heuristic_standing import HeuristicStanding
from openmind.dashboard.model.constraint_learning import ConstraintLearning
from openmind.dashboard.model.game_view import GameView

GIGABYTE = 1024**3

#: Where the page following the runs working out a game's constraints is served, and where one run of them is,
#: with the run's own name after it.
CONSTRAINTS_PATH = "/constraints"

STYLE = """
body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem 1.25rem; background: #f7f7f5; color: #222; }
h1 { font-size: 1.3rem; margin: 0 0 .25rem; } h2 { font-size: 1.05rem; margin: 1.5rem 0 .5rem; }
.muted { color: #666; font-size: .85rem; }
nav { display: flex; flex-wrap: wrap; gap: .5rem; margin: 0 0 1rem; }
nav a { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: .3rem .7rem;
        text-decoration: none; color: #222; font-size: .9rem; }
nav a:hover { border-color: #999; }
.cards { display: flex; flex-wrap: wrap; gap: .75rem; }
.card { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: .6rem .9rem; min-width: 9rem; }
.card b { display: block; font-size: 1.35rem; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; background: #fff; font-size: .85rem; }
th, td { border: 1px solid #ddd; padding: .3rem .5rem; text-align: left; white-space: nowrap; }
th { background: #eee; }
pre { background: #fff; border: 1px solid #ddd; padding: .6rem; overflow-x: auto; font-size: .8rem; }
.warn { color: #a33; }
td.agree { background: #e6f4ea; }
td.mistake { background: #fce8e6; }
td.count { text-align: right; font-variant-numeric: tabular-nums; }
.legend { font-size: .85rem; margin: .4rem 0; padding-left: 1.2rem; } .legend b { font-weight: 600; }
.charts { display: flex; flex-wrap: wrap; gap: .75rem; }
.chart { margin: 0; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: .5rem .6rem; width: 30rem; max-width: 100%; }
.chart figcaption { font-size: .85rem; font-weight: 600; margin-bottom: .2rem; }
.chart svg { width: 100%; height: auto; }
.chart .legend { font-weight: 400; padding: 0; margin-left: .5rem; }
.chart .key { margin-right: .6rem; white-space: nowrap; }
.chart .key i { display: inline-block; width: .6rem; height: .6rem; margin-right: .25rem; border-radius: 2px; }
.chart text { font-size: 10px; fill: #666; }
.game { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }
.board { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: .6rem; width: 26rem; max-width: 100%; }
.board svg { width: 100%; height: auto; display: block; }
.board pre { margin: 0; border: 0; }
/* Columns and not a flex row, because a wrapped flex line is as tall as its tallest card: one heuristic
   holding eighteen rules stretched its whole row and left the three beside it standing over a void. Columns
   flow each card into the shortest one, which is the masonry this wants and needs no script. The count comes
   from the width so a laptop gets two and a wide screen four, rather than four cards squeezed to fit
   whatever is there. */
.heuristics { columns: 24rem auto; column-gap: 1.5rem; }
.judged { break-inside: avoid; margin: 0 0 1.5rem; }
.judged h3 { margin-top: 0; }
table.sortable th { cursor: pointer; user-select: none; }
table.sortable th:hover { background: #e0e0e0; }
table.sortable th.by::after { content: ' \25BC'; font-size: .7em; }
table.sortable th.by.up::after { content: ' \25B2'; }
.judged table { width: 100%; table-layout: fixed; }
/* The rule wraps and the ratings do not. A rule reads `max(here.payoff[other], here.color[8, 4] == other)`
   and cannot fit a card's width on one line, so every table was scrolled sideways and the last signal's
   column was cut off the page -- the column somebody opens this to compare. */
.judged td:first-child, .judged th:first-child { white-space: normal; word-break: break-word; }
.steps { display: flex; gap: .4rem; align-items: center; margin-top: .5rem; flex-wrap: wrap; }
.steps button { font-size: 1rem; padding: .2rem .6rem; }
.record { flex: 1 1 18rem; min-width: 0; white-space: pre-wrap; word-break: break-word; }
"""

#: Steps through the latest decisive game: buttons and the arrow keys move between positions, and the position shown is
#: kept for that game in the browser, so the page reloading itself comes back to it.
#: Clicking a heading orders the table by that column. Read as a number where every cell in it is one, and
#: as text otherwise, so "worth" sorts by size and "heuristic" by name. A record like "5-8-6" is read by its
#: first number and then its last, which is wins first and losses to break a tie -- as a string it would put
#: "10-2-3" above "2-0-0", which is the ordering nobody wants and the one a plain sort gives.
SORT_SCRIPT = """
<script>
const number = (text) => {
  const parts = String(text).trim().match(/^(-?\d+(?:\.\d+)?)-(\d+)-(\d+)$/);
  if (parts) return [Number(parts[1]), -Number(parts[3])];
  const one = Number(String(text).replace(/[+,]/g, '').trim());
  return Number.isFinite(one) && String(text).trim() !== '' ? [one, 0] : null;
};
document.querySelectorAll('table.sortable').forEach((table) => {
  const body = table.tBodies[0];
  table.querySelectorAll('thead th').forEach((heading, column) => {
    heading.onclick = () => {
      const up = heading.classList.contains('by') && !heading.classList.contains('up');
      table.querySelectorAll('thead th').forEach((one) => one.classList.remove('by', 'up'));
      heading.classList.add('by');
      if (up) heading.classList.add('up');
      const rows = Array.from(body.rows);
      const read = (row) => {
        const cell = row.cells[column];
        return cell ? cell.textContent : '';
      };
      const numeric = rows.every((row) => read(row).trim() === '' || number(read(row)) !== null);
      rows.sort((first, second) => {
        const a = read(first), b = read(second);
        if (numeric) {
          const x = number(a) || [-Infinity, 0], y = number(b) || [-Infinity, 0];
          return (y[0] - x[0]) || (y[1] - x[1]);
        }
        return a.localeCompare(b);
      });
      if (up) rows.reverse();
      rows.forEach((row) => body.appendChild(row));
    };
  });
});
</script>
"""

GAME_SCRIPT = """
(() => {
  const game = JSON.parse(document.getElementById('game-data').textContent);
  const position = document.getElementById('position'), caption = document.getElementById('caption');
  const last = game.pictures.length - 1, stored = 'openmind-game';
  // A game with no positions keeps what the page already says — that it cannot be replayed — rather than
  // being handed index zero of nothing, which reads as the word "undefined" where the board should be.
  if (last < 0) { caption.textContent = `${game.moves.length} moves, no positions`; return; }
  let at = 0;
  try { const kept = JSON.parse(sessionStorage.getItem(stored) || 'null'); if (kept && kept.key === game.key) at = Math.min(kept.at, last); } catch (e) {}
  const show = (next) => {
    at = Math.max(0, Math.min(last, next));
    if (game.pictured) { position.innerHTML = game.pictures[at]; }
    else { const pre = document.createElement('pre'); pre.textContent = game.pictures[at]; position.replaceChildren(pre); }
    caption.textContent = at === 0 ? `start, ${last} moves` : `move ${at} of ${last}: ${game.moves[at - 1]}`;
    try { sessionStorage.setItem(stored, JSON.stringify({key: game.key, at})); } catch (e) {}
  };
  document.getElementById('first').onclick = () => show(0);
  document.getElementById('previous').onclick = () => show(at - 1);
  document.getElementById('next').onclick = () => show(at + 1);
  document.getElementById('last').onclick = () => show(last);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft') show(at - 1);
    if (event.key === 'ArrowRight') show(at + 1);
  });
  show(at);
})();
"""

class DashboardHtmlMapper:
    """Maps a snapshot to a page that reloads itself: the current round's progress, the machine and the training's
    processes, every finished round, the latest value rules, the latest notable log lines, and earlyoom's latest
    kills."""

    def __init__(self, svg_chart_mapper: SvgChartMapper | None = None) -> None:
        self._charts = SvgChartMapper() if svg_chart_mapper is None else svg_chart_mapper

    def to_html(self, snapshot: DashboardSnapshot, refresh_seconds: int) -> str:
        domain = html.escape(snapshot.domain)
        parts = [
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            f"<meta http-equiv='refresh' content='{refresh_seconds}'>",
            f"<title>{domain} training</title><style>{STYLE}</style></head><body>",
            f"<h1>{domain} training</h1>",
            # This page builds its own shell rather than going through `_page`, so it has to be given the
            # navigation too — which is exactly how it came to be the one page without any.
            self._nav(),
            f"<div class='muted'>Snapshot {html.escape(snapshot.taken_at)}, reloading every {refresh_seconds} seconds</div>",
            self._progress(snapshot),
            self._machine(snapshot),
            self._plots(snapshot),
            self._latest_game(snapshot),
            self._models(snapshot),
            self._recent(snapshot),
            "</body></html>",
        ]
        return "\n".join(parts)

    def _progress(self, snapshot: DashboardSnapshot) -> str:
        progress = snapshot.progress
        processes = snapshot.machine.processes
        training = next((process for process in processes if process.role == TRAINING), None)
        workers = sum(1 for process in processes if process.role == WORKER)
        # A run that said which process it is counts as much as the one entrypoint the old string matched.
        # Without this the page read "not running" while four learners had been going for a day and a half.
        runs = tuple(process for process in processes if process.role == RUN)
        eldest = training or (max(runs, key=lambda one: one.seconds) if runs else None)
        running = "running" if eldest is not None else "not running"
        cards = [
            ("Learning", f"{running}, {self._duration(eldest.seconds)}" if eldest else running),
            ("Runs", str(len(runs) + (1 if training is not None else 0))),
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
            f"<div class='card'>Learning processes' memory<b>"
            f"{sum(process.rss_bytes for process in machine.processes) / GIGABYTE:.1f} GB</b></div>"
        )
        rows = [
            (
                str(process.pid),
                process.run or "",
                process.role,
                f"{process.rss_bytes / GIGABYTE:.2f}",
                self._duration(process.seconds),
            )
            for process in machine.processes
        ]
        table = (
            self._table(("process", "run", "role", "memory (GB)", "running for"), rows)
            if rows
            else "<p>Nothing learning. A run says which process is its own; one that does not is not seen here.</p>"
        )
        kills = (
            f"<h2>earlyoom's latest kills</h2><pre>{html.escape(chr(10).join(machine.earlyoom))}</pre>"
            if machine.earlyoom
            else "<h2>earlyoom's latest kills</h2><p>None seen since the dashboard started.</p>"
        )
        return f"<h2>Machine</h2><div class='cards'>{cards}</div><details><summary>Processes</summary>{table}</details>{kills}"

    def _plots(self, snapshot: DashboardSnapshot) -> str:
        """The training's games round by round, drawn: how they ended and how long they were. The round being played counts the games it has finished so far, so the plots grow
        while it runs. Nothing to draw before the first game ends."""
        played = snapshot.played
        if not played:
            return ""
        charts = self._game_plots(played)
        if not charts:
            return ""
        return f"<h2>Round by round</h2><div class='charts'>{''.join(charts)}</div>"

    def _game_plots(self, played: Sequence[object]) -> list[str]:
        if not played:
            return []
        columns = [f"round {games.number}" for games in played]  # type: ignore[attr-defined]
        charts = [
            self._charts.stacked(
                "Games a round: decisive and drawn",
                columns,
                (
                    ("decisive", [games.decisive for games in played]),  # type: ignore[attr-defined]
                    ("drawn", [games.draws for games in played]),  # type: ignore[attr-defined]
                ),
            ),
            self._charts.line(
                "Plies a game: the mean, between the shortest and the longest",
                columns,
                [games.mean_plies for games in played],  # type: ignore[attr-defined]
                [(games.shortest or 0, games.longest or 0) for games in played],  # type: ignore[attr-defined]
            ),
        ]
        endings = sorted({ending for games in played for ending, _ in games.endings})  # type: ignore[attr-defined]
        if endings:
            charts.append(
                self._charts.stacked(
                    "How games ended",
                    columns,
                    [(ending, [dict(games.endings).get(ending, 0) for games in played]) for ending in endings],  # type: ignore[attr-defined]
                )
            )
        return charts

    #: Every page worth going to, and what to call it. One list, so a page added here is reachable from all of
    #: them rather than from whichever one happened to link to it.
    PAGES = (
        ("/", "Training"),
        (CONSTRAINTS_PATH, "What it is learning"),
        ("/heuristics", "Heuristics"),
        ("/games", "Games"),
    )

    def _page(self, title: str, refresh_seconds: int, body: str) -> str:
        """A page of its own, under the title, reloading itself every so many seconds (never at 0).

        Every page carries the same way to every other. Before, a page was reached by whichever other page
        happened to link to it, so the run's learning could only be found by knowing its address."""
        refresh = f"<meta http-equiv='refresh' content='{refresh_seconds}'>" if refresh_seconds > 0 else ""
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"{refresh}<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>"
            f"<h1>{html.escape(title)}</h1>{self._nav()}{body}{SORT_SCRIPT}</body></html>"
        )

    def _nav(self) -> str:
        """The same way to every page, on every page."""
        return "<nav>" + " ".join(
            f"<a href='{where}'>{html.escape(name)}</a>" for where, name in self.PAGES
        ) + "</nav>"

    def heuristics_page(
        self, domain: str, standings: Sequence[HeuristicStanding], refresh_seconds: int
    ) -> str:
        """Every heuristic a run has judged, with what each measure made of it.

        **Each measure is its own column and none is folded into another.** What games paid is the anchor and
        what a teller makes of a position is a claim beside it; a heuristic may do well on one and badly on the
        other, and that is the case somebody opens this page to find. A single blended number would hide it.

        **Coverage is read beside the score, never inside it.** A detector that speaks on a twentieth of the
        decisions and is right every time is the thing this project keeps saying it wants, and it is
        indistinguishable from a useless one if what it says is averaged over the decisions it declined."""
        if not standings:
            return self._page(
                f"{domain} heuristics",
                refresh_seconds,
                "<h2>Heuristics</h2><p class='muted'>Nothing judged yet. A heuristic appears here once a "
                "judging has put it to the decisions of a game that finished.</p>",
            )
        rows = [
            (
                one.name,
                f"{one.worth:+.4f}",
                f"{one.mass:.3f}",
                f"{one.offered:.3f}",
                one.speaks,
                str(one.declined),
                str(one.undecided),
                str(one.judgings),
                "not asked" if one.tracks is None else f"{one.tracks:+.3f}",
                str(one.told),
                f"{one.wins}-{one.draws}-{one.losses}" if one.games else "never played",
                ", ".join(one.vouched) or "nobody",
                "retired" if one.retired else "",
            )
            for one in standings
        ]
        header = (
            "heuristic", "worth", "mass", "ignorance", "decided", "declined", "undecided", "judgings",
            "tracks the teller", "positions", "W-D-L", "vouched by", "",
        )
        explanation = (
            "<div class='muted'>What each heuristic has been measured at, by each measure separately. "
            "<b>worth</b> is the mass it put on what was actually played, less what a heuristic with no "
            "opinion would have put there — above nought it saw something, at nought it knows nothing, below "
            "it is wrong about what wins. <b>decided</b> is how often it had an opinion at all, and it is "
            "beside the score rather than in it: a rule that fires rarely and is right is a rule worth "
            "keeping. <b>tracks the teller</b> is how closely its reading of a position follows a stronger "
            "player's, from -1 to 1 — a claim with measured reliability, never the thing that decides, "
            "because a heuristic that tracks a teller perfectly has learned its blind spots too.</div>"
        )
        return self._page(
            f"{domain} heuristics",
            refresh_seconds,
            f"<h2>Heuristics ({len(standings)})</h2>{explanation}{self._table(header, rows, sortable=True)}"
            f"{self._rated(standings)}",
        )

    def _rated(self, standings: Sequence[HeuristicStanding]) -> str:
        """Each heuristic's rules, with what every signal made of each one.

        **This is the only account of why a rule is in a heuristic.** The table above says who backed the whole
        of it; this says what each signal thought of each part, including the signals that thought little and
        did not pay. Where two disagree about a rule is the thing worth reading — one prizes a term that lines
        up with winning and another a term nothing else already says, and which of them was right about a game
        is settled by what each earns rather than by an argument.

        Only heuristics whose rules anybody rated, because a run with no economy buys nothing and a page of
        empty tables says less than no table at all."""
        found = [one for one in standings if one.rated]
        if not found:
            return ""
        sections = []
        for one in found:
            signals = sorted({name for _, held in one.rated for name, _ in held})
            rows = [
                (rule, *(self._rate(dict(held), name) for name in signals))
                for rule, held in sorted(one.rated)
            ]
            sections.append(
                f"<div class='judged'><h3>{html.escape(one.name)}</h3>"
                f"{self._table(('rule', *signals), rows)}</div>"
            )
        return (
            "<h2>What each signal made of each rule</h2>"
            "<div class='muted'>Every signal's rating of every rule a fit kept, not only the ones it paid "
            "for. A signal rating a rule at nearly nothing says as much about that rule as one that bought "
            "it, and where two disagree is the thing to read. The scales are each signal's own and are never "
            "comparable across columns — only the order within a column means anything.</div>"
            f"<div class='heuristics'>{''.join(sections)}</div>"
        )

    def _rate(self, held: dict, signal: str) -> str:
        """What that signal made of that rule, or nothing where it said nothing about it.

        Two figures, because only the order within a column means anything: a rating of `1.053e-11` carried
        four digits of what is a zero to anybody reading it, and the width of them was a good part of what
        pushed the last signal's column off the page."""
        return "—" if signal not in held else f"{held[signal]:.2g}"

    def games_page(self, domain: str, games: Sequence[GameListing], refresh_seconds: int) -> str:
        """The page listing every remembered game, newest first, each linking to its own page."""
        rows = [
            (
                f"<a href='/game/{html.escape(game.id)}'>{html.escape(game.label)}</a>",
                html.escape(game.ended),
                *(html.escape(model) for _, model in game.players),
                html.escape(" ".join(f"{payoff:g}" for payoff in game.payoffs)),
                html.escape(game.ending or "none"),
                str(game.plies),
            )
            for game in games
        ]
        players = games[0].players if games else ()
        header = ("game", "ended", *(player for player, _ in players), "payoffs", "ending", "plies")
        body = self._table(header, rows, escaped=True) if rows else "<p>No game played yet.</p>"
        return self._page(
            f"{domain} games",
            refresh_seconds,
            f"<div class='muted'><a href='/'>Back to the training</a></div><h2>Games ({len(games)}, "
            f"{sum(1 for one in games if len(set(one.payoffs)) > 1)} decisive)</h2>{body}",
        )

    def game_page(self, domain: str, game: GameView, refresh_seconds: int) -> str:
        """The page showing one game, with links to the games just before and after it."""
        links = [
            "<a href='/games'>All games</a>",
            *(() if game.previous_id is None else (f"<a href='/game/{html.escape(game.previous_id)}'>Previous game</a>",)),
            *(() if game.next_id is None else (f"<a href='/game/{html.escape(game.next_id)}'>Next game</a>",)),
            "<a href='/'>Back to the training</a>",
        ]
        return self._page(f"{domain} {game.label}", refresh_seconds, self._game_section(game, game.label, " · ".join(links)))

    def constraints_page(
        self,
        domain: str,
        learning: ConstraintLearning | None,
        refresh_seconds: int,
        runs: Sequence[ConstraintLearning] = (),
    ) -> str:
        """The page following a run that is working out what a game refuses: the position it is on, how the
        constraints it now holds stand against that position, and the constraints themselves.

        `runs` is every run that has said anything, so the page can put them beside each other before it shows
        one of them in full. Arms are how anything here is settled and they are launched together; a page that
        shows one and names no other hides the only comparison that decides anything."""
        if learning is None:
            return self._page(
                f"{domain} constraints",
                refresh_seconds,
                "<p class='muted'>No run has said anything yet. One writes here after each position it learns "
                "from.</p>",
            )
        cards = "".join(
            f"<div class='card'><span class='muted'>{html.escape(name)}</span><b>{html.escape(value)}</b></div>"
            for name, value in (
                ("position", str(learning.position)),
                ("constraints", str(len(learning.rules))),
                ("readings", str(learning.readings)),
                ("moves the game allows", str(learning.legal)),
                ("candidates", str(learning.candidates)),
                ("seconds", f"{learning.seconds:.1f}"),
                *((("matching the rules", learning.matched),) if learning.matched else ()),
            )
        )
        board = (
            f"<div class='board'>{learning.picture}</div>"
            if learning.picture
            else f"<div class='board'><pre>{html.escape(learning.fen)}</pre></div>"
        )
        return self._page(
            f"{domain} constraints",
            refresh_seconds,
            f"{self._runs(runs, learning)}"
            f"<h2>{html.escape(learning.run) or 'The run'}</h2>"
            f"<p class='muted'>{html.escape(learning.at)} &middot; {html.escape(learning.fen)}</p>"
            f"<div class='cards'>{cards}</div>"
            f"<h2>The position it is on</h2><div class='game'>{board}</div>"
            f"<h2>How it is doing</h2>{self._matrix(learning)}"
            f"<h2>What it refuses</h2>{self._rules(learning)}"
            f"<h2>What a move does</h2>{self._consequences(learning)}"
            f"<h2>What the notation says</h2>{self._notation(learning)}",
        )

    def _runs(self, runs: Sequence[ConstraintLearning], showing: ConstraintLearning) -> str:
        """Every run that has said anything, beside each other, the one being shown marked.

        Empty where there is only one, because a comparison of one run with itself is a row of numbers already
        on the page below it. `let through` is what the arms are usually being compared on: a candidate the
        game refuses and the constraints do not."""
        if len(runs) < 2:
            return ""
        head = (
            "run", "still going", "position", "constraints", "readings", "let through", "wrongly refused",
            "seconds", "last said",
        )
        rows = "".join(
            "<tr>"
            f"<td><a href='{CONSTRAINTS_PATH}/{html.escape(one.run)}'>{html.escape(one.run)}</a>"
            f"{' &larr;' if one.run == showing.run else ''}</td>"
            f"<td>{self._going(one)}</td>"
            f"<td class='count'>{one.position}</td>"
            f"<td class='count'>{len(one.rules)}</td>"
            f"<td class='count'>{one.readings}</td>"
            f"<td class='count'>{one.let_through}</td>"
            f"<td class='count'>{one.wrongly_refused}</td>"
            f"<td class='count'>{one.seconds:.1f}</td>"
            f"<td>{html.escape(one.at)}</td>"
            "</tr>"
            for one in runs
        )
        headings = "".join(f"<th>{html.escape(one)}</th>" for one in head)
        return (
            "<h2>The runs</h2><div class='scroll'>"
            f"<table><tr>{headings}</tr>{rows}</table></div>"
            "<p class='muted'>Each run writes its own snapshot; they are launched together so that an arm with "
            "a change can be read against one without it. Whether one is still going is its own process being "
            "looked for, not how long ago it last spoke.</p>"
        )

    def _going(self, learning: ConstraintLearning) -> str:
        """Whether that run is still going, in words rather than a mark, and honest about not knowing.

        A run that named no process is not known to be running, which is not the same as stopped: an older run
        wrote no process and saying it had stopped would be a claim nothing here can make."""
        if learning.running:
            return "<b>yes</b>"
        return "<span class='muted'>not said</span>" if not learning.pid else "no"

    def _consequences(self, learning: ConstraintLearning) -> str:
        """What the predictor has worked out a move does, drawn from the action rather than from the position it
        left.

        A consequence naming a square outright — rather than the row of where the move started and the column of
        where it lands — is one the sightings have not yet narrowed, and says the evidence is still thin rather
        than that the move is about that square."""
        if not learning.consequences:
            return "<p class='muted'>Nothing yet. It takes two moves that differ before a drawing narrows.</p>"
        return "".join(f"<pre>{html.escape(one)}</pre>" for one in learning.consequences)

    def _notation(self, learning: ConstraintLearning) -> str:
        """What the game's notation is made of and what its pieces say.

        The sorts and the shapes come from the strings alone, knowing nothing of what any move did; the
        couplings are measured against what happened. Keeping those apart is what lets the two check each other
        rather than agree by construction.

        A part nothing says anything about is the half the rules have to supply — in chess that is where a move
        *starts*, left out precisely because the rules make it recoverable."""
        if not (learning.sorts or learning.shapes or learning.couplings):
            return (
                "<p class='muted'>Nothing yet. A piece of notation is measured once it has turned up in enough "
                "moves to be more than an accident.</p>"
            )
        sorts = ", ".join(html.escape(one) for one in learning.sorts) or "none"
        shapes = "".join(f"<pre>{html.escape(one)}</pre>" for one in learning.shapes[:12])
        couplings = (
            "".join(f"<pre>{html.escape(one)}</pre>" for one in learning.couplings[:20])
            or "<p class='muted'>No piece of it says anything yet.</p>"
        )
        return (
            f"<p><span class='muted'>sorts of character, found by the company they keep &middot; </span>"
            f"<b>{sorts}</b></p>"
            f"<details><summary>{len(learning.shapes)} shapes it comes in</summary>{shapes}</details>"
            f"<p class='muted'>what each place of each shape says, firmest first</p>{couplings}"
        )

    def _matrix(self, learning: ConstraintLearning) -> str:
        """Agreement and disagreement, counted over every candidate the solver could propose.

        Agreement is green and the two mistakes are red, so which corner is which is seen before it is read. The
        two mistakes are still named underneath, because they are not the same mistake and the colour says only
        that both are wrong. Letting a refused candidate through is a move OMF would offer and
        the game would reject, and it finds out at once. Refusing a move the game allows is a move OMF will never
        make, and nothing will ever tell it what it missed."""
        rows = [
            ("the game refuses", (learning.rightly_refused, "agree"), (learning.let_through, "mistake")),
            ("the game allows", (learning.wrongly_refused, "mistake"), (learning.rightly_allowed, "agree")),
        ]
        head = "".join(
            f"<th>{html.escape(one)}</th>"
            for one in ("", "the constraints refuse", "the constraints allow")
        )
        body = "".join(
            f"<tr><td>{html.escape(said)}</td>"
            + "".join(f"<td class='count {kind}'>{count}</td>" for count, kind in cells)
            + "</tr>"
            for said, *cells in rows
        )
        return (
            f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
            + "<ul class='legend'>"
            f"<li><b>{learning.let_through}</b> let through: a move OMF would offer and the game would reject, "
            "which it hears about at once.</li>"
            f"<li class='warn'><b>{learning.wrongly_refused}</b> wrongly refused: a move OMF will never make, "
            "and nothing will ever tell it what it missed.</li></ul>"
        )

    def _rules(self, learning: ConstraintLearning) -> str:
        """Every constraint it now holds, longest first, since a constraint still carrying a whole position is
        what distilling has not yet got to."""
        if not learning.rules:
            return "<p class='muted'>None yet.</p>"
        return "".join(f"<pre>{html.escape(one)}</pre>" for one in learning.rules)

    def missing_page(self, domain: str) -> str:
        return self._page(f"{domain} decisive games", 0, "<p>No such decisive game. <a href='/games'>All decisive games</a></p>")

    def _latest_game(self, snapshot: DashboardSnapshot) -> str:
        game = snapshot.latest_game
        if game is None:
            return ""
        link = f"<a href='/games'>All games ({snapshot.decisive_games} decisive)</a>"
        return self._game_section(game, "Latest game", link)

    def _game_section(self, game: GameView, heading: str, links: str) -> str:
        players = ", ".join(f"{html.escape(model)} ({html.escape(player)}) {payoff:g}" for (player, model), payoff in zip(game.players, game.payoffs, strict=True))
        ending = "" if game.ending is None else f" by {html.escape(game.ending)}"
        # A game with no positions is a game that was played and cannot be replayed — its moves were written
        # in a shape the declared rules do not take, or several players acted at once and the summary does not
        # say who did what. What it paid and what was played are still known, and saying so is better than a
        # page that fails: the reader learns which games the rules can be put to and which they cannot.
        if not game.pictures:
            first = "<pre>This game cannot be replayed from the declared rules, so its positions are not shown.</pre>"
        elif game.pictured:
            first = game.pictures[0]
        else:
            first = f"<pre>{html.escape(game.pictures[0])}</pre>"
        data = json.dumps({"key": f"{game.label} {game.ended}", "moves": list(game.moves), "pictures": list(game.pictures), "pictured": game.pictured})
        record = "" if game.record is None else f"<pre class='record'>{html.escape(game.record)}</pre>"
        return (
            f"<h2>{html.escape(heading)}</h2><div class='muted'>{html.escape(game.label)}, ended {html.escape(game.ended)}: "
            f"{players}{ending}</div><div class='muted'>{links}</div><div class='game'><div class='board'><div id='position'>{first}</div>"
            "<div class='steps'><button id='first' title='first position'>⏮</button><button id='previous' title='previous move'>◀</button>"
            "<button id='next' title='next move'>▶</button><button id='last' title='last move'>⏭</button>"
            f"<span id='caption' class='muted'>start, {len(game.moves)} moves</span></div></div>{record}</div>"
            f"{self._heuristics(game)}"
            f"<script id='game-data' type='application/json'>{data.replace('</', '<\\/')}</script><script>{GAME_SCRIPT}</script>"
        )

    def _heuristics(self, game: GameView) -> str:
        """What each side judged with, rule by rule with its weight.

        A name says which heuristic won and nothing about why. The weight on a rule is what the thing that rule
        reads is worth to it, so these read as what each side believed a position was made of — which is the
        thing to argue with when one of them keeps winning.

        **Each side is one box, and it has to be said in the markup.** A heading, a caption and a table laid
        beside each other are three things to a row of boxes, not one — so two sides came out as six items
        strung across the page, each heading beside somebody else's table. Nothing was wrong with the numbers
        and the page was unreadable."""
        if not game.heuristics:
            return ""
        sections = []
        for player, named, rules in game.heuristics:
            heading = f"<h3>{html.escape(player)}: {html.escape(named)}</h3>"
            if not rules:
                sections.append(
                    f"<div class='judged'>{heading}<p class='muted'>No rules to read: it judged with nothing.</p></div>"
                )
                continue
            body = self._table(
                ("rule", "weight"),
                [(html.escape(name), f"{weight:+.6g}") for name, weight in rules],
                escaped=True,
            )
            sections.append(
                f"<div class='judged'>{heading}<p class='muted'>{len(rules)} rules, heaviest first</p>{body}</div>"
            )
        return f"<h2>What each side judged with</h2><div class='heuristics'>{''.join(sections)}</div>"

    def _models(self, snapshot: DashboardSnapshot) -> str:
        if not snapshot.models:
            return ""
        rows = [
            (
                model.name,
                model.id,
                str(model.games),
                str(model.wins),
                str(model.draws),
                str(model.losses),
                "none" if model.score is None else f"{model.score:.3f}",
                model.last_game,
            )
            for model in snapshot.models
        ]
        explanation = (
            "<div class='muted'>Every model that has played, as the knowledge base remembers it, saved as each game ended. "
            "A model is its settings and rules word for word; the id comes from that text, so a model whose rules changed "
            "has a new id. Games count the sides a model played; score is points per game, a win 1 and a draw 0.5. "
            "The latest to play first.</div>"
        )
        header = ("model", "id", "games", "wins", "draws", "losses", "score", "last game")
        return f"<h2>Models</h2>{explanation}{self._table(header, rows, sortable=True)}"

    def _recent(self, snapshot: DashboardSnapshot) -> str:
        if snapshot.progress is None or not snapshot.progress.recent:
            return ""
        return f"<h2>Latest log lines</h2><pre>{html.escape(chr(10).join(snapshot.progress.recent))}</pre>"

    def _table(
        self, header: Sequence[str], rows: Sequence[Sequence[str]], escaped: bool = False, sortable: bool = False
    ) -> str:
        """A table under its header; `escaped` cells are already HTML, such as links, and go in as they are.

        A `sortable` table's headings can be clicked to order by that column. The server's order stands until
        somebody asks for another, so a page is read top-down as it was written and rearranged only on
        purpose."""
        head = "".join(f"<th>{html.escape(cell)}</th>" for cell in header)
        cell_text = (lambda cell: cell) if escaped else html.escape
        body = "".join("<tr>" + "".join(f"<td>{cell_text(cell)}</td>" for cell in row) + "</tr>" for row in rows)
        classes = "table sortable" if sortable else "table"
        return (
            f"<div class='scroll'><table class='{classes}'><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
        )

    def _duration(self, seconds: float) -> str:
        """A duration people read at a glance: 3 h 06 min, 24 min 13 s, or 45 s."""
        hours, rest = divmod(int(seconds), 3600)
        minutes, rest = divmod(rest, 60)
        if hours:
            return f"{hours} h {minutes:02d} min"
        if minutes:
            return f"{minutes} min {rest:02d} s"
        return f"{rest} s"
