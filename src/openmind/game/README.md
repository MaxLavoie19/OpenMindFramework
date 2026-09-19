# game

## Purpose

How a programmer defines a game for OMF, and how OMF finds the games installed. The integrator runs the game; OMF only
simulates it, so there is no game object here: a game is the rules its context's **simulation ruleset** holds, which
the RBS, the CSP and the predictor read back.

OMF knows nothing of turns, phases, priority, draws, abandoning, clocks or boards. They belong to the game, as its own
variables, actions and rules. All players play at the same time, all the time: in a game played in turns, the
constraints leave a player no action outside their turn, reading the player solved for as `player`.

## Content

| File | What it is |
|---|---|
| `service/game_declarer.py` | `GameDeclarer(knowledge_base, context, ruleset="simulation", open=False, rule_caller=None)`: `rule`, `starts_at`, `played_by`, `definitions`, `values`, `constraint`, `constraints`, `leads_to`, `together`, `ending`, `lasts`, `cools_down`, `record`, `picture`, `variant_of`, `done`; `ruleset`, `context` |
| `service/game_registry.py` | `GameRegistry(games=None)`: `names()`, `declare(name, knowledge_base)`; games come from the `openmind.domains` entry points unless given |
| `constant/game_constant.py` | `DOMAIN_ENTRY_POINTS`, `VARIANT_SEPARATOR` |

## Usage

```python
from openmind.game.service.game_declarer import GameDeclarer
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.players import Players

declarer = GameDeclarer(knowledge_base, "tictactoe")
declarer.starts_at(initial_state)
declarer.played_by(Players(("X", "O"), "payoff"))
declarer.values("place", "row", PythonRule("(1, 2, 3)"))
declarer.values("place", "col", PythonRule("(1, 2, 3)"))
declarer.constraints("place", PythonRule("turn == player"), PythonRule("cell[row, col] is None"))
declarer.leads_to("place", PythonRule("cell = cell.placed((row, col), turn)\nturn = 'O' if turn == 'X' else 'X'"))
declarer.done()
```

A project registers a function declaring its game, `declare(name, knowledge_base) -> context`, under the
`openmind.domains` entry points; OMF's example games register the same way in its own `pyproject.toml`.

- **Frozen by default.** The rules and the ruleset are frozen unless declared open (`open=True`, on the declarer for
  the ruleset or on `rule` for one rule). OMF never revises a frozen rule; a frozen ruleset lists only frozen rules, so
  an open rule is refused there with a warning. A heuristic changes a copy instead.
- **Declaring again** leaves the knowledge base as it was: a rule of the same name and kind is declared anew under its
  id.
- **Variants.** `variant_of(game, leaving)` lists the game's rules in the variant's ruleset, less those left, and
  records the link in `Context.inherits`. A rule the variant declares under a name it took from its game becomes its
  own, and the game keeps the original.
- **Payoffs** are the Map `played_by` names, which an end state's effects fill.
- **Durations and cooldowns** (`lasts`, `cools_down`) are declared only, for now.
- `record` and `picture` are encoders of the game, not simulation rules; they move to the codec step.

## Logs

- Logger `openmind.game.service.game_declarer`: `INFO Declared <n> rules of <context> into ruleset <name> (<id>)[, open]`;
  `DEBUG <context> is a variant of <game>: it takes <n> of its rules`.
- Logger `openmind.game.service.game_registry`: `DEBUG Declared <name> as context <context>`.

## Notes

- Tests: `service/game_declarer_tests.py`, `service/game_registry_tests.py`.
