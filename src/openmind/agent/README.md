# agent

## Purpose

The loop: perceive, process, plan, communicate, act, under one budget — and the example games OMF ships with.

**Acting and strategizing run at once.** The planner strategizes while the actor acts on what it has worked out so
far. Both read the same state, so the actor answers what is actually happening and the planner works from where the
game really is. Where the strategy has nothing prepared for the state it is in, the actor waits: that is the surprise
case, and the planner is already strategizing from there.

**OMF doesn't run games; integrators do.** What performs an action is a port an integrator fills — a robot's control
loop, a game client's connection, an API — and what happened comes back through the world OMF reads.

## Content

| File | What it is |
|---|---|
| `model/dispatcher.py` | `Dispatcher`, the port performing what a strategy calls for: `dispatch(action)`, `performing()` |
| `service/actor.py` | `Actor(dispatcher, wait_seconds=0.01)`: acts on a strategy, in a thread of its own with `start`/`stop`; `follow(strategy)`, `act(world)` |
| `service/agent.py` | `Agent(time_manager, planner, actor=None)`: `play(knowledge_base, game, world, guidance, budget)`, `allocate(...)` |
| `service/timekeeper.py` | `Timekeeper(time_source=None)`: each player's clock, and a choice timed |
| `service/game_memory.py` | `GameMemory`: games remembered as direct experiences in the knowledge base |
| `factory/agent_factory.py` | `create_actor(dispatcher)`, `create_agent(planner=None, actor=None)` |

The example games — tic-tac-toe and its variants, sudoku, the prisoner's dilemma, rock paper scissors — live in
`constant/`, `factory/`, `model/`, `mapper/` and `repository/`; see the sections below.

## The loop

1. **Perceive.** The integrator pushes what it saw; the world holds it (see `world/README.md`).
2. **Plan.** The time management policy says what to run with; the planner gives a strategy (see `budget/README.md`
   and `search/README.md`).
3. **Act.** The actor dispatches what the strategy says to play in the state as it stands, and waits where it says
   nothing.

The actor opens no debugger frames: a pause in its thread would hold the game up, and what it did is in the logs.

## Logs

- `openmind.agent.service.actor`: `INFO Dispatched <action>`, `DEBUG Nothing prepared for this state: waiting while
  the planner strategizes`.
- `openmind.agent.service.agent`: `INFO Nothing to play for <player> here`.

## Notes

- Tests: `service/actor_tests.py`, `service/agent_tests.py`, `service/game_memory_tests.py`, and the example games'.

## Domains from installed projects

OpenMind ships without the libraries a problem needs. A problem is programmed in its own project, which installs
OpenMind, writes its rules — free to import any library — and declares them into the knowledge base with a
`GameDeclarer` (see `game/README.md`). It registers the function that declares them in its `pyproject.toml`:

```toml
[project.entry-points."openmind.domains"]
chess = "openmind_chess.game.factory.chess_factory:declare_chess"
```

The function takes the whole game name and a knowledge base and gives back the context it declared. The
`GameRegistry` (see `game/README.md`) looks for the part of the name before `/`, and calls that function with the
whole name (`"chess"`, or a variant such as `"chess/960"`). Every command taking a game
name, `openmind-play` among them, then plays the project's game.

## Tic-tac-toe and its variants

A variant, `TicTacToeVariant(name, width, height, line, gravity)`, describes how a game differs from standard
tic-tac-toe: a grid of `width` columns by `height` rows, `line` marks in a row to win, and, with `gravity`, marks that
fall to the lowest empty cell of the column they are dropped in. The variants are data in
`tictactoe_constant.VARIANTS`, each written as the standard game plus what it changes; the same recipes build them all.

| Variant | Context | Width × height | Line | Gravity |
|---|---|---|---|---|
| `standard` | `tictactoe` | 3 × 3 | 3 | no |
| `fourinarow` | `tictactoe/fourinarow` | 7 × 6 | 4 | yes |
| `gomoku` | `tictactoe/gomoku` | 15 × 15 | 5; a longer line also wins (freestyle) | no |

A variant needs a width and a height of at least 1 and a line from 1 to its longer side; otherwise the recipes raise
`ValueError`. A variant that changes a rule rather than a size, such as misère, would add a field that the recipes
read.

### Players

`Players(("X", "O"), "payoff")`: the `payoff` map holds each player's payoff. Whose turn it is is the game's own
`turn` model, which the constraints read.

### State models

| Model | Values | Initial |
|---|---|---|
| `cell`, a `Grid` of height rows by width columns, `cell[row, col]` with row 1 at the top | `"X"`, `"O"`, or `None` when empty | every cell `None` |
| `turn`, a scalar | `"X"` or `"O"` | `"X"` |
| `payoff`, a `Map` by player | 1 for a win, 0 for a loss, 0.5 each for a draw; `None` until the game ends | `{X: None, O: None}` |

### Constraint rules (solved by the CSP)

Every rule of a variant sees the names of `create_tictactoe_definitions(variant)`, a script run once: `WIDTH`,
`HEIGHT`, `LINE`, `PLAYERS`, `WIN`, `DRAW`, `LOSS`, `other(player)`, and `LINES_THROUGH[row, col]`, every line of
`LINE` cells through a cell that fits the grid, worked out once from `Grid.lines(LINE)` of an empty grid of the
variant's size.

Without gravity, the action `place(row, col)`, with row in 1..height and col in 1..width, is legal when these rules
hold, checked in this order:

```python
turn == player
payoff['X'] is None
payoff['O'] is None
cell[row, col] is None
```

`player` is the player the legal actions are solved for, so the player whose turn it isn't has no action.

With gravity, the action `drop(col)`, with col in 1..width, has the same payoff rules and `cell[1, col] is None`, the
column's top cell.

### Effects rules (run by the predictor)

The action has one branch with probability 1, whose effects are this script:

```python
cell = cell.placed((row, col), turn)
if any(all(cell[where] == turn for where in line) for line in LINES_THROUGH[row, col]):
    payoff = payoff.with_item(turn, WIN).with_item(other(turn), LOSS)
elif all(mark is not None for mark in cell.cells):
    payoff = payoff.with_item('X', DRAW).with_item('O', DRAW)
turn = other(turn)
```

With gravity it first finds the landing row, `row = max(r for r in range(1, HEIGHT + 1) if cell[r, col] is None)`, and
the draw check reads only the top row: `all(cell[1, c] is not None for c in range(1, WIDTH + 1))`.

Only the lines through the landing cell are checked, so a move in 4 in a row checks at most 13 lines rather than all
69.

## Sudoku

Sudoku is solved by the CSP alone, without MCTS or RBS: `openmind-solve sudoku`. The domain `sudoku` holds
`sudoku_constant.PUZZLE`, row by row, with `.` for an empty cell. `create_sudoku_domain(name, grid)` makes a domain of
any other grid; a grid that isn't 81 characters, each `.` or a digit from 1 to 9, raises `ValueError`.

Published collections are read from `data/sudoku/`, one `<collection>.txt` file each (the root README has the
downloads). `openmind-solve` names their puzzles `sudoku/<collection>/<number>`:

```python
from pathlib import Path

from openmind.agent.factory.sudoku_factory import create_sudoku_domain
from openmind.agent.mapper.sudoku_collection_mapper import SudokuCollectionMapper
from openmind.agent.repository.sudoku_puzzle_repository import SudokuPuzzleRepository

repository = SudokuPuzzleRepository(SudokuCollectionMapper())
repository.collections(Path("data/sudoku"))   # ('euler', 'hardest', 'top95')
puzzle = repository.load(Path("data/sudoku"), "top95")[6]
domain = create_sudoku_domain(f"sudoku/{puzzle.collection}/{puzzle.number}", puzzle.grid)
```

### Players

`Players(("solver",), "payoff")`.

### State models

| Model | Values | Initial |
|---|---|---|
| `cell`, a 9 by 9 `Grid`, `cell[row, col]` | a digit 1..9, or `None` when empty | the grid's clue, or `None` |
| `payoff`, a `Map` by player | 1.0 for the solver once the grid is filled; `None` until then | `{solver: None}` |

### Constraint rules (solved by the CSP)

The action `fill` has one parameter per empty cell, named `cell_<row>_<col>` (`cell_1_3`), with domain 1..9. It is
legal when `payoff['solver'] is None` holds and, for each of the 27 rows, columns and boxes (the grid's `rows()`,
`columns()` and `boxes((3, 3))`), an `all_different` rule over its
cells, the parameters of its empty cells and the clues read from the state:
`all_different(cell[1, 1], cell[1, 2], cell_1_3, cell_1_4, cell[1, 5], cell_1_6, cell_1_7, cell_1_8, cell_1_9)`.

### Effects rules (run by the predictor)

`fill` has one branch with probability 1, a script placing every parameter in its cell
(`cell = cell.placed((1, 3), cell_1_3)`, one line per empty cell), then `payoff = payoff.with_item('solver', 1.0)`, the
share of cells filled.

## Repeated prisoner's dilemma

The first non-zero-sum game: each player's payoff is their own total, so both can gain, or both lose. The two players
choose at the same time, which the `standard` and `uncertain` variants play as A choosing, then B, and the
`simultaneous` variant plays with both players acting at once.

| Variant | Context | Rounds | Ending chance after each round | Choices |
|---|---|---|---|---|
| `standard` | `prisonersdilemma` | 10 | 0 | A then B |
| `uncertain` | `prisonersdilemma/uncertain` | no known last round | 0.1 | A then B |
| `simultaneous` | `prisonersdilemma/simultaneous` | 10 | 0 | at once |

In the `simultaneous` variant, there is no turn: both players choose at once;
`choose` is legal for a player when no payoff is set and `chosen[player] is None`; its effects are only
`chosen = chosen.with_item(player, choice)`, and the rules for what the choices together lead to play the round with the script below, from `points =`
on, with `each` for `player`. A choice is kept only until the resolution,
which runs before anyone chooses again.

With a known last round, perfect play defects in every round, whatever the ending chance. Without one, the game can't
be searched exactly, since its states never stop.

### Players

`Players(("A", "B"), "payoff")`.

### State models

| Model | Values | Initial |
|---|---|---|
| `chosen`, a `Map` by player | this round's choice, `"cooperate"` or `"defect"`; `None` until made, and again once the round is played | `{A: None, B: None}` |
| `played`, a `Grid` of one row per round and one column per player, A then B | the choice played in a round; `None` for the round being chosen; a round's row is added when it starts | one row, `None` |
| `round`, a scalar | the round being chosen, from 1 | 1 |
| `score`, a `Map` by player | points so far | `{A: 0, B: 0}` |
| `turn`, a scalar, except in the `simultaneous` variant | `"A"` or `"B"` | `"A"` |
| `payoff`, a `Map` by player | the player's score when the game ends; `None` until then | `{A: None, B: None}` |

### Constraint rules (solved by the CSP)

Every rule sees `create_prisoners_dilemma_definitions(variant)`: `PLAYERS`, `POINTS[choice of A, choice of B]`, the
points of both players for a round, `ROUNDS`, `None` without a known last round, and `other(player)`. The action
`choose(choice)`, with choice `"cooperate"` or `"defect"`, is legal when these rules hold:

```python
turn == player
payoff['A'] is None
payoff['B'] is None
chosen[player] is None
```

### Effects rules (run by the predictor)

| A | B | Points of A | Points of B |
|---|---|---|---|
| cooperate | cooperate | 3 | 3 |
| cooperate | defect | 0 | 5 |
| defect | cooperate | 5 | 0 |
| defect | defect | 1 | 1 |

`choose` has one branch without an ending chance, `ENDING = False`, or two: the game goes on with 1 − the chance and
ends with the chance, `ENDING = True`. Each branch runs this script after setting `ENDING`:

```python
chosen = chosen.with_item(turn, choice)
if all(chosen[player] is not None for player in PLAYERS):
    points = POINTS[chosen[PLAYERS[0]], chosen[PLAYERS[1]]]
    for column, (player, gained) in enumerate(zip(PLAYERS, points), start=1):
        played = played.placed((round, column), chosen[player])
        score = score.with_item(player, score[player] + gained)
        chosen = chosen.with_item(player, None)
    if round == ROUNDS or ENDING:
        for player in PLAYERS:
            payoff = payoff.with_item(player, score[player])
    else:
        round = round + 1
        played = Grid((round, len(PLAYERS)), played.cells + (None,) * len(PLAYERS))
turn = other(turn)
```

The next round's row of `played` is a new grid one row longer. A choice that doesn't finish a round gives the same
state on both branches.

### What B knows of A's choice

OMF runs agents, not the game: an agent is given the position it's in, A's choice included, and nothing hides it.

## Rock paper scissors

The first domain where players act at once: A and B throw a shape together, rock beating scissors, paper beating rock
and scissors beating paper. A win pays 1, a draw 0.5 each, a loss 0. Its only equilibrium is throwing each shape a third
of the time, which the search's regret matching heads toward; an opponent predicted to favour a shape gets the shape
beating it.

### Players

`Players(("A", "B"), "payoff")`; both players act at once.

### State models

| Model | Values | Initial |
|---|---|---|
| `hand`, a `Map` by player | the shape thrown, `"rock"`, `"paper"` or `"scissors"`; `None` until thrown | `{A: None, B: None}` |
| `payoff`, a `Map` by player | 1, 0.5 or 0 once both have thrown; `None` until then | `{A: None, B: None}` |

### Constraint rules (solved by the CSP)

Solved for each player, read as `player`: `throw(shape)`, with shape `"rock"`, `"paper"` or `"scissors"`, is legal
when `hand[player] is None`.

### Effects rules (run by the predictor)

`throw` sets `hand = hand.with_item(player, shape)`. The rule for what the two throws lead to together, run once both have thrown, compares `hand['A']` and `hand['B']`
with `BEATS` and sets both payoffs; no player has an action left, and the game is over.

## Usage

Playing a game with OMF, the caller being the integrator:

```python
from openmind.agent.factory.agent_factory import create_actor, create_agent
from openmind.budget.model.budget import Budget
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.rbs.factory.rbs_factory import create_game
from openmind.search.factory.search_factory import create_minimax
from openmind.search.model.guidance import Guidance
from openmind.world.service.world import World

knowledge_base = create_knowledge_base("tictactoe")
game = create_game("tictactoe", knowledge_base)
world = World(game.start())
agent = create_agent(create_minimax(), create_actor(my_dispatcher))

strategy = agent.play(knowledge_base, game, world, Guidance("X"), Budget(5.0))
strategy.chosen(world.current())        # what it settled on here
```

`openmind-play` is the worked example: it runs the game in the terminal, dispatches what OMF chose, and pushes back
what came of it (see `entrypoint/README.md`).

Playing a game forward through its rules:

```python
actions = game.actions(game.start())
# 9 actions; actions[0] is Action(name='place', parameters=(('col', 1), ('row', 1)))
game.outcomes(game.start(), actions[0])
```

## Notes

- Tests: `service/actor_tests.py`, `service/agent_tests.py`, `service/game_memory_tests.py`,
  `factory/prisoners_dilemma_factory_tests.py`, `factory/rock_paper_scissors_factory_tests.py`,
  `factory/sudoku_factory_tests.py`, `factory/tictactoe_factory_tests.py`,
  `mapper/sudoku_collection_mapper_tests.py`, `repository/sudoku_puzzle_repository_tests.py`.
