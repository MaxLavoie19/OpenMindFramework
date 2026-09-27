import html
import json
from collections.abc import Sequence

from openmind.dashboard.constant.dashboard_constant import RUN, TRAINING, WORKER
from openmind.dashboard.mapper.svg_chart_mapper import SvgChartMapper
from openmind.dashboard.model.dashboard_snapshot import DashboardSnapshot
from openmind.dashboard.model.game_listing import GameListing
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
.heuristics { display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: flex-start; }
.heuristics > div, .heuristics table { max-width: 40rem; }
.steps { display: flex; gap: .4rem; align-items: center; margin-top: .5rem; flex-wrap: wrap; }
.steps button { font-size: 1rem; padding: .2rem .6rem; }
.record { flex: 1 1 18rem; min-width: 0; white-space: pre-wrap; word-break: break-word; }
"""

#: Steps through the latest decisive game: buttons and the arrow keys move between positions, and the position shown is
#: kept for that game in the browser, so the page reloading itself comes back to it.
GAME_SCRIPT = """
(() => {
  const game = JSON.parse(document.getElementById('game-data').textContent);
  const position = document.getElementById('position'), caption = document.getElementById('caption');
  const last = game.pictures.length - 1, stored = 'openmind-game';
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
    PAGES = (("/", "Training"), (CONSTRAINTS_PATH, "What it is learning"), ("/games", "Games"))

    def _page(self, title: str, refresh_seconds: int, body: str) -> str:
        """A page of its own, under the title, reloading itself every so many seconds (never at 0).

        Every page carries the same way to every other. Before, a page was reached by whichever other page
        happened to link to it, so the run's learning could only be found by knowing its address."""
        refresh = f"<meta http-equiv='refresh' content='{refresh_seconds}'>" if refresh_seconds > 0 else ""
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"{refresh}<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>"
            f"<h1>{html.escape(title)}</h1>{self._nav()}{body}</body></html>"
        )

    def _nav(self) -> str:
        """The same way to every page, on every page."""
        return "<nav>" + " ".join(
            f"<a href='{where}'>{html.escape(name)}</a>" for where, name in self.PAGES
        ) + "</nav>"

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
        first = game.pictures[0] if game.pictured else f"<pre>{html.escape(game.pictures[0])}</pre>"
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
        thing to argue with when one of them keeps winning."""
        if not game.heuristics:
            return ""
        sections = []
        for player, named, rules in game.heuristics:
            heading = f"<h3>{html.escape(player)}: {html.escape(named)}</h3>"
            if not rules:
                sections.append(f"{heading}<p class='muted'>No rules to read: it judged with nothing.</p>")
                continue
            body = self._table(
                ("rule", "weight"),
                [(html.escape(name), f"{weight:+.6g}") for name, weight in rules],
                escaped=True,
            )
            sections.append(f"{heading}<p class='muted'>{len(rules)} rules, heaviest first</p>{body}")
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
        return f"<h2>Models</h2>{explanation}{self._table(header, rows)}"

    def _recent(self, snapshot: DashboardSnapshot) -> str:
        if snapshot.progress is None or not snapshot.progress.recent:
            return ""
        return f"<h2>Latest log lines</h2><pre>{html.escape(chr(10).join(snapshot.progress.recent))}</pre>"

    def _table(self, header: Sequence[str], rows: Sequence[Sequence[str]], escaped: bool = False) -> str:
        """A table under its header; `escaped` cells are already HTML, such as links, and go in as they are."""
        head = "".join(f"<th>{html.escape(cell)}</th>" for cell in header)
        cell_text = (lambda cell: cell) if escaped else html.escape
        body = "".join("<tr>" + "".join(f"<td>{cell_text(cell)}</td>" for cell in row) + "</tr>" for row in rows)
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
