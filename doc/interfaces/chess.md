# Interfaces: OpenMindChess on the new game API

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

OpenMindChess is a project of its own (`/home/maxime/Documents/OpenMindChess`) that installs OMF and python-chess and
declares chess into the knowledge base. Nothing of it collects today: its state is the old flat variables, and it
imports `openmind.doxastic`, `openmind.mcts`, `openmind.evaluation` and `openmind.agent.service.random_policy`, all
gone. It is rewritten rather than ported, as agreed.

## What stays as it is

The rules of chess are ordinary Python functions of the project, at a module's top level so a worker started fresh
finds them by name, and python-chess does the chess. That was already the shape, and the new API asks for the same.
What changes is the state they read and write, and which declarer declares them.

## The state: named data models

The old state was 136 flat variables, `piece(1,1)` through `payoff(black)`. The new state is named data models:

| Model | What it is |
|---|---|
| `piece` | a `Grid` of 8 rows by 8 columns, each cell `"pawn"`…`"king"` or `None`; row 1 is rank 8 and column 1 the a file, so a board read row by row is a board as white sees it |
| `color` | a `Grid` of the same shape, each cell `"white"`, `"black"` or `None` |
| `turn` | a scalar: the player to act |
| `castling` | a scalar: the rights as an X-FEN writes them |
| `en_passant` | a scalar: the square where a capture is legal, or `None` |
| `halfmove` | a scalar: the halfmove clock |
| `history` | a scalar: the position keys since the last capture or pawn move, for the repetition rule |
| `payoff` | a `Map` by player: 1, 0.5, 0 once the game is over, `None` until then |

The piece and the colour stay two grids, as decided. A square is written as chess writes it, `piece["e4"]`; its
coordinates are (row, column) as every other OMF grid's are, which is my call, not one you made — rank-first
would be one line in the names.

**The grids carry chess's own names for the squares.** `Grid` takes aliases, so a rule or a heuristic reads
`piece["e4"]` as readily as `piece[4, 5]`, and `piece.ray((1, 1), (0, 1), blocked)` walks a file until something
stands in the way. Inference reads positions through the same grid methods it reads every other game's by.

## Who can act

All players play at once, and the constraints say who acts. The CSP adds the player it is solving for to the state as
a model named `player`, so:

```python
def the_player_s_turn(state: State, **parameters: Value) -> bool:
    """A move is legal while no payoff is set, and only for the player whose turn it is."""
```

and the values rule gives nothing where it isn't that player's turn, so black's moves aren't generated and thrown away
on white's turn:

```python
def legal_moves(state: State) -> Iterable[Value]:
    """Every legal move of the position, as python-chess generates them; none when it isn't the player's turn."""
```

## What is declared

`GameDeclarer` replaces `RuleDeclarer`, with the same rules going in:

```python
declarer.starts_at(create_chess_state(fen))
declarer.played_by(Players(("white", "black"), "payoff"))
declarer.values("move", "uci", legal_moves)
declarer.constraint("move", 1, the_player_s_turn)
declarer.leads_to("move", play_move)
declarer.ending(why_it_ended)
declarer.record(game_record)
declarer.picture(board_picture)
```

`Players` is `(names, payoff)` now: no `to_act`, since OMF doesn't know whose turn it is, and one payoff model rather
than a variable per player.

**The relaxations** — `teleport`, where pieces go anywhere, and `ignoring check` — stay what they were: a variant of
chess declared with `variant_of(chess, leaving=((VALUES, …), (EFFECTS, …)))`, which copies chess's rules less the two
it replaces.

## Two rules with no home

- **`declarer.empty(PIECE, EMPTY)`** told the old system what an empty square holds. A grid's cells say it: nothing to
  declare.
- **`declarer.timeout(out_of_time)`** has no kind any more. Clocks belong to the integrator, as decided at the game
  step: OMF doesn't enforce them. See Open.

## The tests

| Test | What it tests | Proposed |
|---|---|---|
| `chess_factory_tests` | the state a FEN gives, what a move leads to, the game ending | rewritten against the new state |
| `chess_rule_tests` | the rules themselves: keys, records, pictures, moves | rewritten |
| `board_cache_tests` | the cache of python-chess boards by state | rewritten |
| `chess_relaxation_factory_tests` | teleport and ignoring check | rewritten |
| `chess_perft_tests` | perft: the legal moves of known positions to a depth | rewritten — the one that proves chess is right |
| `chess_clock_tests` | a game played on a clock | deleted: it runs `MatchRunner` and the old tree search, both gone |
| `chess_inference_tests` | the expression search over chess positions | deleted for now: the inference step revives it |
| `chess_deduction_tests` | the position deducer | deleted for now: the inference step revives it |
| `play_chess_tests` | `openmind-play` and `openmind-evaluate` end to end | rewritten for `openmind-play`; `evaluate` is gone |

## As coded (2026-09-20)

The project was rewritten rather than ported. Its whole suite passes: 39 tests in 0.3 seconds, perft among them.

- **The state** is the eight models above, with the two grids named through OMF's `CellNames` and populated by
  `Grid.of` from a 2D array laid out as a board looks.
- **Who can act:** the CSP hands the rules the player it is solving for as a state model named `player`.
  `the_player_s_turn` reads it, and `legal_moves` gives nothing for the player whose turn it isn't: at the start,
  white has 20 moves and black none.
- **The clock rules** are kept as `game/service/chess_clock.py`, plain functions of the project for an integrator
  that runs a clock, declared to OMF as nothing. `game_record` writes a time forfeit where the moves didn't end the
  game but the payoffs are set, so nothing was lost with the `flagged` parameter.
- **`history` stays a scalar** of position keys.
- **Perft** matches the published counts through OMF's solver and predictor: 20, 400 and 8902 from the start, 48 and
  2039 from Kiwipete.
- **`openmind-play chess --agent black --planner improvised` plays**, the terminal being the integrator; the
  end-to-end test drives it.

### What OMF gained

Chess couldn't declare where it starts: `starts_at` keeps the state as the repr of its models, and a game's own alias
class can't be read back from that. Maxime's answer was that an integration should be able to populate an OMF data
structure and hand it over, so the gap was OMF's:

- **`CellNames(columns, rows)`** (`structure/model/cell_names.py`): the names a game gives a grid's cells, a cell's
  name being its column's and its row's. It is among the models every rule reads, so a state built from it reads
  back. `GridAliases` stays the port for a game that names its cells some other way.
- **`Grid.of(cells)`**: a grid from its cells written out, nested as they are laid out, in any number of dimensions;
  ragged rows raise ValueError.

### Found while coding

- **A game's context isn't always a name the registry knows.** `chess from <fen>` is declared by whoever wants it and
  read back by its context, since the registry only answers to `chess`.
- **Stale knowledge on disk breaks a rewritten game**: `data/knowledge/chess/` still held the old rules, which no
  longer load. Deleting it was enough, as it was for sudoku at the model step.

## What was left behind

- `chess_clock_tests` covers what a flag means, but nothing plays a game on a clock: that test ran OMF's match runner
  and the old tree search, both gone.
- The inference and deduction tests are not rewritten: they exercise OMF's inference, whose own step hasn't come.
- `openmind-evaluate` is gone, so the end-to-end test drives `openmind-play` alone.

## Open when this was written (both settled)

1. **What becomes of the clock rules.** `out_of_time` says what a flag means in chess — the flagged player loses,
   unless the other couldn't checkmate with the material left — and `game_record` writes a time forfeit. OMF no longer
   has a place to declare either: a clock is the integrator's, and the record rule is called with the actions and the
   payoffs only. Three ways:
   - **keep them as functions of the chess project**, exported for an integrator that runs a clock, but declared to
     OMF as nothing. Chess still knows what a flag means; OMF doesn't have to. *(I lean here)*
   - **make running out of time an action of the game**, which needs the clock in the state, so OMF would simulate
     clocks after all;
   - **drop them**, and let each integrator decide what a flag means.
2. **Whether the history stays a scalar of keys.** The repetition rule needs the positions since the last capture or
   pawn move, and today they are a space-separated string of hashes in one scalar. A `List` model would say what it is
   more plainly, at the cost of a longer state. I lean to keeping the scalar: it is what the rule reads, and nothing
   else looks at it.
