# Handoff: the OMF bootstrap loop

Rewritten 2026-09-28, replacing an earlier version that was wrong in several load-bearing ways. Everything
here is either running, on disk, or measured, and where a number is quoted the run that produced it is named.
Where a thing is unproven it says so.

**Read this sceptically.** The version it replaces argued for its departures from the specification instead of
implementing them, and stated as fact several things that were not. The corrections are listed at the foot so
the same ground is not walked twice.

## What the project is

OpenMindFramework (OMF) is a game-agnostic learner that induces a game's rules from play. Chess is the test
bed and is **treated as unknown**: OMF is not told how a rook moves, and nothing in the framework may know a
board, a turn, an opponent or a piece. The measure of success is whether the learning works, not whether the
chess is good.

Two repositories, and the framework never imports the game:

| repository | branch | what it holds |
|---|---|---|
| `/home/maxime/Documents/OpenMindFramework` | `statement-domain` | the framework; ships without chess |
| `/home/maxime/Documents/OpenMindChess` | `induced-games` | the chess adapter; imports OMF and python-chess |

Long runs happen on `maxime-cinamon` over Tailscale SSH, from `/mnt/shared`. The machine has 24 logical cores
and 62 GB, and carries other long jobs — at the time of writing `find_heuristics.py` had been running for two
days with six workers and load sat at 21 of 24. Budget accordingly, and check what else is running before
blaming a slow run on the code.

## The specification

This is the bootstrap loop as the project owner described it. **It is the requirement.**

1. **Start with only what the integrator implemented** — for chess, the moves and domains per piece, the grid,
   and the link to the engine.
2. **Ponder briefly** to produce candidate heuristics. Expect a mix of arbitrary signals and sensible ones:
   covering more squares, holding more pieces, pieces that cover more squares, removing the opponent's
   material, perhaps a fork.
3. **Self-play using those heuristics.** One is drawn for white and a *different* one for black, at the start
   of each game, by Upper Confidence Bound and roulette selection.
4. **The engine is the authority on legality.** Each turn the agent receives a board position and a list of
   legal moves. Constraints are for analysis and for sharing the model of the game — they are not consulted
   when choosing a move.
5. **While playing, learn** the encoder, the decoder, the constraints, and the predictor.
6. **At the end of a game, ponder for a few minutes.** Analyse which heuristics predicted the outcome and
   which correlate with each other. Generate new candidates by relaxing constraints, changing the weights of
   rules in a weighted rule-based system, and training a neural network.
7. **Accumulate** — constraints, heuristic scores, and deeper facts about positions: piece values, defended
   pieces, trades, and patterns of the kind the Chess Intelligence Agent detects.
8. **At the same time, learn from watching other players** and extracting the decisions they follow.

### Scoring rules, as settled

- **A heuristic is the rule-based system**, or a network, or anything that produces a score per move. A rule
  (a fork detector, say) is a *term inside* one. A set of one rule is still a set, which is how a disjoint
  rule gets tested.
- **Unfinished games are skipped.** They paid nobody.
- **A draw is a draw** — win one, draw a half, loss none. This is the scale `AccuracyScorer.played` already
  uses.
- **Ties are ignored** for training purposes.
- **The score is the probability mass** the heuristic put on the move that was actually played, weighted by
  what the game paid the player who played it. Ratings become chances by softmax.
- **Not firing is not being wrong.** Coverage — how often a heuristic had an opinion — is reported beside the
  score and never folded into it. A rule that fires in three per cent of positions and is right is a rule
  worth keeping; folding coverage into the score destroys exactly those.
- **A rule's worth inside a set is its fitted weight given the other terms**, which only applies to models
  made of rules. The outer score works for any model family, which is what lets a network and a rule set be
  compared on one number.

## Where the loop stands against the specification

| step | state |
|---|---|
| 1 start from the integrator's own | done, unchanged |
| 2 ponder for candidates | done, now in the players |
| 3 self-play drawing two *different* heuristics per game | **done** — `ModelDrawer`, drawn in `_Players._for` |
| 4 the engine is the authority | done, unchanged; constraints are never consulted to move |
| 5 learn encoder, decoder, constraints, predictor while playing | done, unchanged |
| 6 ponder at the end of each game | **done** — each worker plays a game then thinks |
| 6 …which heuristics correlate with each other | **not built**; see open question 41 |
| 6 …train a neural network | **not built.** No torch anywhere. Its own component, deliberately deferred |
| 7 accumulate | done; nothing is ever retired, which is open question 41 |
| 8 learn from watching other players | done — `--watching`, `--watched-games` |

## What changed, and why

Each item is a commit on the two branches above.

**The crash.** A run died twenty-two minutes in with `KeyError: "Unknown state model: 'piece'"`, immediately
after its first pondering succeeded. `chess_decisions.decisions` built each `Decided` out of the *learner's*
vocabulary — a grid of pieces and a grid of square colours — and attached the *declared* game to it, whose
rules read `piece` and `color`. The moves had the same fault, said in the induced notation that a rater cannot
step through. Both crossings already existed and were already used on the same data elsewhere
(`create_chess_state`, `acted`). This is why verification line 3 below had never been seen: not because the
loop had not reached it, but because reaching it killed the run.

**The drawer.** `ModelDrawer` existed, was tested nowhere, and **could not be imported at all** — it reached
for `ModelRecord` in a module that does not exist. The suite stayed green because nothing imported it. It now
imports, has tests beside it, has a factory, and is wired.

**Who plays each game.** The players were handed the newest few heuristics once a pondering, and how many was
`--contests`, which defaulted to **zero** — so the hand-off returned nothing and the players played at random
for an entire run with every other flag set correctly. The draw now happens in `_Players._for`, where
`TaskRunner.stream` asks for a game's arguments as a worker takes it, which is the one place in the parent
that runs at the start of every game. `--contests` is gone.

**Where pondering happens.** It ran in the learner every N learning positions, which had nothing to do with
games. It now runs in the player worker: play a game, think about it, hand back what was settled. A game and a
think is simply a long game, and the learner is in another process meanwhile. `--pondering` is a switch rather
than a count.

**What a worker thinks over.** Not the one game it just played — that gives one ending, one valued row, and
settles nothing. It thinks over the last `--pondering-games` finished games it has played. What it settles
crosses back as rules and weights and the learner adopts it (`ValueGenerator.adopt`), because a worker that
opened the run's store would be a second writer to it.

**Resuming.** `held` started empty at every launch, so a restart kept the record and relearned from zero. The
constraints an earlier run believed are now read back from the induced game's own ruleset, with their ids, so
a rule already in the store is revised rather than declared twice.

## Measured findings worth keeping

These cost real time to establish. Do not re-derive them.

- **Pondering over walked positions values nothing.** `PositionGatherer` walks at random; measured, *"valued 0
  of 20 positions — 0 paid out, 0 proved"*, relaxations included. A random walk never reaches an ending, so
  nothing pays out and nothing is within proving distance. This is why `ponder()` accepts positions.
- **Positions given are positions used.** `HeuristicPonderer` gathers its own positions only where it is
  handed none, so `settings.positions` stops being a budget the moment a caller supplies any. Handing it a
  four-hundred-ply game whole is four hundred deductions at a second apiece before any fitting starts. The
  caller must cap.
- **One game is one payoff and settles nothing.** Measured, relaxations off: one ending settles nothing in
  29.8 seconds; two endings settle six value rules in 32.6. With relaxations on, a single game runs for
  minutes, because the relaxation search is what pondering falls back on when the payoff route pays nothing.
  Several endings is what made pondering cheap when the learner did it over everybody's games at once.
- **A thing is never valued by what its rules admit.** The heuristics learned are per-square rather than
  material, and the fix that suggests itself — ask the rules how much each thing can do and seed the search
  with that — is the one move this project must not make. Three services had made it, and all three are gone;
  see the ruling at the head of `doc/open-questions.md`. The term meaning "what a knight is worth" is the
  two-condition pattern `piece[i]=='knight' and color[i]==me`, which `ExpressionGenerator` reaches only as a
  later generation, so reaching it is a search-cost problem to be solved by search, not by supplying the
  answer. If the games do not bear a piece value out, OMF does not have one.
- **A chess position has two representations.** `ChessPositions.state` (learning) and the declared game's
  state (`piece`, `color`). They are not interchangeable; `create_chess_state(fen)` is the crossing. This
  blocked pondering, then the dashboard board, then killed a run. Assume any new code crossing between the
  learner and the declared game has this bug until a test says otherwise.
- **The induced move notation collapses promotions.** `move(self, x, y)` cannot say which piece a pawn became,
  so the adapter picks a queen. 65 of 380 converted games replay only as far as their first non-queen
  promotion. Games written since use the declared notation and do not have this.
- **`HypothesisTester` checked its deadline between bodies, not inside one.** A condition can be cheap or
  ruinous — "could this be taken after the move" draws every candidate of the resulting position — so one body
  over 14,398 cases is roughly 207 million board drawings. Five seconds of budget bought 25 minutes. It now
  reads the clock every 64 cases and abandons the body, returning nothing rather than a partial count, because
  what a body refuses is what it is priced on.
- **Of the four learners, only the constraints hunt their own surprises.** The scouts rank positions by how
  many of the game's own moves the constraints refuse — the too-tight error, which is silent, where too-loose
  is exposed by every legal-move list. The predictor learns from whatever sightings arrive. The notation
  measures its misses every position and discards them. `_Scouts` accepts only one kind of wrongness.
- **An agreement score is against whatever produced the move.** Where that was a search, the heuristic is
  being asked to have been a search. The literature's answer is to label against the search's own values
  rather than its final move, which needs a visit distribution on `PlayedGame` that does not exist. Not urgent
  while play is random.

## How to verify the loop end to end

```
.venv/bin/pytest        # framework: 1584 pass in ~70s
.venv/bin/pytest        # chess: 71 pass in ~7 min, from the repo root
```

Then a run, watching its log for, in order:

1. `Learning from <source>` cycling through all four position sources with a running tally.
2. `Pondered N played positions and kept M value rules` with M above zero — **seen**, 2026-09-28 20:32:19,
   under the learner-side pondering that has since moved into the workers.
3. `<heuristic> expected 0.41 of what happened where knowing nothing would have expected 0.03, over 87
   decisions of 210` — a heuristic judged against a game it did not play. **Never yet seen**; the crash above
   is why. This is the thing to confirm first.
4. Two *different* heuristics named as drawn at the start of a game.
5. `N heuristics to draw from, M of them never yet played` — the reading open question 41 waits on.
6. `Judging N heuristics over M decisions of K games took S seconds` — the cost curve beside it.
7. `Resumed N constraints an earlier run believed`, on a restart against a store that has been used.

## Open questions

`doc/open-questions.md` holds the design questions with their numbers and reasoning. Live ones touching this
work: **26**, **31**, **32**, **33**, **36**, **37**, **40**, and **41** — nothing retires a candidate, and
whether that matters is to be measured over a night rather than capped in advance.

Still unwritten, and named here so it is not lost: whether the induced game should be able to draw its own
games rather than the declared one.

## Where the previous handoff was wrong

Listed because a handoff that cannot be trusted is worse than none, and because each of these would have cost
somebody an afternoon.

| it said | the truth |
|---|---|
| the running job "does not run the heuristic half — the code was written after it launched" | it ran it and pondered successfully, then crashed in judging |
| `ModelDrawer` "is tested nowhere" | it could not be imported at all |
| "All framework tests (1,416) pass" | 1,611 at the time; 1,584 now, the difference being tests deleted with the code they pinned |
| the run launched "at 20:15" | 20:10:04 |
| the pondering clock "needs a decision before the loop matches the specification" | there was no dilemma: a worker plays a game then ponders, and a game-and-a-think is a long game |
| — | `--contests` defaulted to zero and silently disabled specification step 3 entirely |
| — | `/mnt/shared/OpenMindFramework-current` was not a git repository and had drifted from the working tree, so the run was not reproducible from git |
| — | the documented `.venv/bin/pytest` did not collect in the chess repo; only `PYTHONPATH=. pytest` worked |
