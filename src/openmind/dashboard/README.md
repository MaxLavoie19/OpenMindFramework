# dashboard

## Purpose

Follows a value training while it runs, on the machine it runs on: a page that reloads itself with the current round's
progress, the machine's memory and the training's processes, every finished round, the latest value rules, the latest
notable log lines, and earlyoom's latest kills. It only reads what trainings already write: their reports, their logs,
the process file system and the system log.

## Content

| File | What it is |
|---|---|
| `model/dashboard_settings.py` | `DashboardSettings(domain, report_directory, log_directory, syslog, proc=Path("/proc"))`: where to read a domain's training from |
| `model/log_progress.py` | `LogProgress(path, round, rounds, round_note, games, matches, searched, deduced, recent, draws=0, decisive=0)`: where a training is, from its log, with how many of the round's self-play games were drawn and how many decisive |
| `model/round_row.py` | `RoundRow(number, rules, held_out_loss, held_out_error, baselines, against_previous, seconds)`: a finished round |
| `model/report_summary.py` | `ReportSummary(path, created_at, complete, rounds, latest_rules)`: a training report |
| `model/process_status.py` | `ProcessStatus(pid, role, rss_bytes, seconds, command)`: a training process: the loop script, the training, or a worker |
| `model/machine_status.py` | `MachineStatus(memory_total, memory_available, swap_total, swap_free, processes, earlyoom)` |
| `model/round_games.py` | `RoundGames(number, games, decisive, draws, plies, shortest, longest, endings)`: what a round's self-play games came to, with `mean_plies` and `decisive_share`; the round being played holds the games finished so far |
| `model/dashboard_snapshot.py` | `DashboardSnapshot(domain, taken_at, report, progress, machine, played)`: everything the page shows at one moment, `played` being each round's games |
| `constant/dashboard_constant.py` | Default port (8765) and reload (30 seconds); how many recent lines (15) and earlyoom kills (10) the page shows; the loggers followed; how training processes are recognized |
| `service/incremental_line_reader.py` | `IncrementalLineReader.new_lines(path)`: the complete lines written since the previous call; a file that got shorter or was replaced is read from its start |
| `service/log_progress_reader.py` | `LogProgressReader.progress(directory)` and `.played()`: the newest log's round, the round's self-play games and how many were drawn or decisive, games against opponents, searched and deduced moves, and the latest notable lines |
| `model/model_score.py` | `ModelScore(name, id, games, wins, draws, losses, last_game)`: what a model's games came to; `score` is points per game |
| `service/model_score_reader.py` | `ModelScoreReader.scores(directory, domain)`: every model the knowledge base remembers with its games, the latest to play first |
| `model/game_view.py` | `GameView(label, ended, players, payoffs, ending, record, moves, pictures, pictured, id='', previous_id=None, next_id=None)`: a game as a page shows it, one picture per position, with its record id and its neighbours' |
| `model/game_listing.py` | `GameListing(id, label, ended, players, payoffs, ending, plies)`: a decisive game as the list shows it, without its positions |
| `service/game_browser.py` | `GameBrowser(game_factory=create_game)`: `decisive(directory, domain)` lists every decisive game the knowledge base remembers, newest first, without replaying any; `game(directory, domain, id)` replays and draws the one under that record id, position by position with the domain's picture rule or laid out as text, with the ids of the decisive games just before and after it, None for no such game; `latest(directory, domain)` draws the newest; the game last drawn is kept |
| `service/report_reader.py` | `ReportReader.summary(directory)`: the newest report's rounds and latest rules |
| `service/machine_reader.py` | `MachineReader.status(proc, syslog)`: memory and swap, the training's processes, earlyoom's latest kills |
| `service/dashboard_service.py` | `DashboardService.snapshot(settings)`: a snapshot from the three readers, which keep their places between snapshots |
| `mapper/svg_chart_mapper.py` | `SvgChartMapper.stacked(title, columns, parts)` and `.line(title, columns, values, band)`: charts as inline SVG, no library and nothing fetched, since the page is served on a tailnet and left open for days |
| `mapper/dashboard_html_mapper.py` | `DashboardHtmlMapper.to_html(snapshot, refresh_seconds)`: the page |
| `factory/dashboard_factory.py` | `create_dashboard_service()` |

## How it reads

- **Progress.** The newest `*.log` under `<log directory>/<domain>/` is read once, then only what's written after, so a
  log of hundreds of megabytes costs one read. A line `Round <k> of <n>: ...` from the training loop starts a round:
  the self-play games (`Self-play game ...` lines ending with the payoffs, so a game's record line doesn't count it
  twice), drawn among them when every payoff is the same and decisive otherwise, games against opponents (`Game with
  seeds ... finished ...`), searched moves (`Searching ...`) and deduced moves (`Deduced ...`) are counted from there. Workers log each game as soon as it
  ends, so the counts grow during a round; searches grow even while every game is still being played. The latest
  15 INFO and WARNING lines of the training's main loggers (training loop, distiller, ending walker, expression search,
  value generator, entrypoint, task runner) are kept. A newer log starts over.
- **Plots.** The same streamed log gives each round its own tally — games, decisive, drawn, the plies they took with the
  shortest and the longest, and how each ended where the domain says — kept per round rather than only for the round in
  progress, so the page draws them round by round and the current round grows as its games finish. A log line is read
  whether or not it carries the time it was produced at, so logs written before lines were timed still read.
- **Rounds.** The newest `*.json` report under `<report directory>/<domain>/`, as `TrainingReportJsonMapper` writes it,
  read whole at every snapshot: reports are small. A round handed over before its games shows none against each
  opponent.
- **Models.** The domain's knowledge base under `data/knowledge/<domain>/` (the settings' `knowledge_directory`), read
  whole at every snapshot: every model remembered, with the sides it played, its wins, draws and losses as `GameMemory`
  remembered them at the end of each game, its score and its latest game, the latest to play first. No table without a
  knowledge base.
- **Latest decisive game.** The same knowledge base's latest game whose payoffs differ, replayed from its moves and
  outcome seed, every position drawn by the domain's picture rule (a chess board with the last move highlighted) or
  laid out as text. The page shows who played which side, the result and why the game ended, the board with buttons
  for the first, previous, next and last position (the left and right arrow keys step too), and the record, such as the
  PGN. Every picture travels in the page, so stepping asks the dashboard for nothing; the position shown is kept for
  that game in the browser's session, so the page reloading itself comes back to it. A chess game's boards weigh about
  31 KB each: a 140-move game makes a page of about 4.5 MB, sent at every reload. The section links to every decisive
  game.
- **Decisive games.** `/games` lists every decisive game, newest first, under a header: the game, when it ended, the
  model each player played, the payoffs, why it ended and its plies, each game linking to its own page. `/game/<id>`,
  the id being the game's record id in the knowledge base, shows that game as the training's page does, with links to
  the decisive games just before and after it and to the list; an id with no decisive game gets a 404 page. A game
  page carries only that game's boards, and the list none.
- **Machine.** `meminfo` gives memory and swap; every process's command line, parent, start and resident memory give the
  training's processes: the loop script (`bash ...continue_training...`), the training entrypoint, and the workers
  whose parent is a training. The system log is read a piece at a time for earlyoom's `sending SIG...` lines, the
  latest 10 kept: earlyoom is what killed trainings on maxime-cinamon.

## Usage

```bash
.venv/bin/openmind-dashboard chess          # on the training machine, from the project that trains
```

It listens on the machine's Tailscale IPv4 address, port 8765, so only the tailnet reaches it: open
`http://<machine>:8765` (see `entrypoint/README.md`).

## Logs

Logger `openmind.entrypoint.dashboard`, in `data/log/dashboard/<YYYY-MM-DD_HH-MM-SS>.log`:

- `INFO Serving the <domain> training on http://<host>:<port>`
- `ERROR The snapshot failed`, with the traceback; the page then says so
- `DEBUG <client> <request>`, one line per request

## Notes

- Tests: `mapper/svg_chart_mapper_tests.py`, `service/incremental_line_reader_tests.py`, `service/log_progress_reader_tests.py`,
  `service/report_reader_tests.py`, `service/machine_reader_tests.py`, `mapper/dashboard_html_mapper_tests.py`.
