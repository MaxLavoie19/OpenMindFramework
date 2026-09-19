# rbs

## Purpose

The rule-based system: a model family that runs the rules its rulesets hold, OMF's main explainable model. What an RBS
is used for is unrelated to what it is: it can fill any task — the game's simulation, a heuristic, a known agent's
play style — and any other model may fill the same task instead.

An RBS is a **model**: one ruleset's rules, each with its weight there, for a context — `tictactoe`, a variant such as
`tictactoe/fourinarow`, a relaxation, a round of training, an arm. A context without a ruleset for a task takes the one
of a context it inherits from. Stateless services, built once and injected, run the RBSs they are given: the
simulation runs a simulation ruleset's. The heuristics are still run by the temporary `RuleBasedGame` facade, until
their own step.

| Task | What the RBS runs |
|---|---|
| legal actions | the simulation ruleset's `constraint` and `values` rules, solved by the CSP (see `csp/README.md`) |
| what actions lead to | the simulation ruleset's `effects` rules, through a rule predictor (see `predictor/README.md`) |
| a position's value | the position value ruleset's `position` rules, each reading times its weight, summed |
| a move's value | the move value ruleset's `move` rules, the same way |

All players play at the same time, all the time: the players acting in a state are those with a legal action there
(`acting`), and a game played in turns leaves a player no action outside their turn.

A **relaxation** is a game with fewer constraints, so it is another context: its simulation ruleset is an open copy of
the game's, less the constraint it drops. Dropping the constraint that keeps a player to their own turn gives the
relaxation where that player can act now.

Nothing here knows a game. Game projects declare their rules (`game/README.md`); the inference engine and fitting link
heuristics into rulesets.

## Rules are Python

Every rule is Python: a constraint, an effect, a heuristic's term. Each is a `PythonRule`, the source of a Python
expression or of a whole script, **or one of the project's own functions**, given to OMF and called as it is.

A project that writes its game in Python hands OMF functions of the prototypes in `rule/model/rule.py`: a constraint
`(state, **parameters) -> bool`, a parameter's values `(state) -> Iterable[Value]`, effects `(state, **parameters) ->
State`, an ending `(state) -> str | None` and a record `(state, actions) -> str | None`. `RuleCaller` calls either kind,
so nothing else in OMF cares which it is; the rules the inference engine generates stay source. A function must live at
a module's top level, since workers are started fresh and are given a rule by name; `GameDeclarer`, given a rule
caller, rejects a lambda or a function defined inside another as it is declared.

Any Python is allowed — imports, libraries, classes, `eval` — because OpenMind is a framework people use to write the
rules of their own problems, and they must be able to plug in whatever their problem needs. This domain compiles rules
once and runs them against states.

A saved rule is code: loading the knowledge base's rules runs what they contain, so they need the same trust as a
source file.

## Content

| File | What it is |
|---|---|
| `model/rule_based_system.py` | `RuleBasedSystem(context, context_id, ruleset, rules)`: the model, one ruleset's rules with their weights there; `of(*kinds)`, `weight(rule)`, `action_names()`, `values(action)`, `constraints(action)`, `effects(action)`, `definitions(name)` |
| `service/simulation.py` | `Simulation(solver, predictor, rule_caller)`: stateless, built once, runs a simulation ruleset's RBS given with every call: `start`, `players`, `actions` (without a player, the one player acting's), `actions_with_statistics`, `acting`, `acting_player`, `joint_actions`, `outcomes`, `joint_outcomes`, `ended`, `call` |
| `service/rule_based_game.py` | `RuleBasedGame(context, simulation_rbs, heuristics, simulation, rule_caller, consequence_library=None, context_id=None)`: a temporary facade over a game's RBSs for the packages not yet reworked, each dropping it at its own step: the simulation's methods without the RBS argument, `start()` and `players()` read once, and the heuristics' `value(state, player)`, `values(state)`, `rate(state, actions, player=None)`, `explain(state, player)`, `weight(rule)`, `describe()`, each rule weighed in its own ruleset |
| `service/game_relaxer.py` | `GameRelaxer(knowledge_base)`: `relaxations(context)`, each constraint dropped; `relax(context, name)` makes that relaxation a context of its own, its simulation ruleset an open copy of the game's |
| `factory/rbs_factory.py` | `create_simulation()`, built once; `create_rule_based_system(knowledge_base, context, task="simulation")` and `find_rule_based_system(...)`, the RBS of the context's ruleset for the task, or of a context it inherits from; `create_rule_based_game(knowledge_base, context, simulation=None)` and `create_game(name, knowledge_base, registry=None)`, the facades; `context_ruleset(knowledge_base, context_id, task)`; `create_value_generator(workers=1)` |
| `constant/game_record_constant.py` | What a record rule reads: `actions`, `payoffs` |
| `model/heuristic_target.py` | `HeuristicTarget(knowledge_base, context)`: where a producer of heuristic rules links what it fits |
| `constant/consequence_constant.py` | The names heuristics read (`me`, `other`, `action`, `win_chance`, `wins`, `near`, `OUTSIDE`) |
| `service/consequence_library.py` | `ConsequenceLibrary`: what heuristics read besides state variables, worked out with the RBS's own legal moves and outcomes |
| `builder/consequence_library_builder.py` | `ConsequenceLibraryBuilder`: wires the library's state reader, variable name mapper and mechanics |
| `model/position_row.py` | `PositionRow(state, player, target)`: a position valued for a player, and the payoff its value is fitted to |
| `model/value_settings.py` | `ValueSettings(prices, max_steps, tolerance, seconds, memory_bytes, candidates=None)` |
| `model/sparse_fit.py` | `SparseFit(weights, bias, steps, settled)`: weights fitted at a price; a weight of 0 drops its term |
| `model/value_fit.py` | `ValueFit(price, terms_kept, steps, settled, training_loss, held_out_loss)`: one price of a sweep |
| `model/value_generation_result.py` | `ValueGenerationResult(context, rules, fits, chosen, candidates, strengths=())`: the position rules declared at the chosen price, and each term's weight in the chosen fit on standardized values, largest first |
| `constant/value_constant.py` | Default value settings: prices 0.1, 0.03, 0.01, 0.003 and 0.001, 1,000 steps, tolerance 1e-6; the search budget's defaults are in `inference/constant/inference_constant.py` |
| `service/term_evaluator.py` | `TermEvaluator`: a term's values on position rows as numbers; several terms at once in the task runner's workers, the rows split in slices; `aggregate_columns(rbs, rows, parts)` folds an aggregate from its readings instead of running its source |
| `service/reading_cache.py` | `ReadingCache`: what a reading gives at every index of a base, kept per position, player and reading, so every candidate sharing a reading reads it once |
| `service/sparse_fitter.py` | `SparseFitter`: a linear fit of the payoffs themselves, with an L1 price on its weights, optionally multiplied per weight by a cost, by accelerated proximal gradient |
| `service/value_generator.py` | `ValueGenerator`: searches expressions with the inference engine, fits them at every price, and declares the fit best on held-out rows as `position` rules; `generate_for_targets(...)` does it for several targets at once |
| `builder/value_generator_builder.py` | `ValueGeneratorBuilder`: sets how many worker processes terms are evaluated in and the memory each holds at most, and wires the generator |

## The kinds of rules

A rule's kind, recorded with it in the knowledge base (see `knowledge/README.md`), says what it is a rule about:

| Kind | What the rule gives | Declared by |
|---|---|---|
| `initial` | where the game starts | the game project |
| `players` | who plays, and the Map holding the payoffs | the game project |
| `definitions` | the names a context's rules (or its effects) all see | the game project |
| `constraint` | whether an action with these parameter values is legal | the game project |
| `values` | the values a parameter of an action can take | the game project |
| `effects` | what an action leads to, with the chance it happens | the game project |
| `duration`, `cooldown` | how long an action takes, and how long before it is available again (declared only for now) | the game project |
| `ending`, `record`, `picture` | why a game ended, its record, a picture | the game project |
| `position` | a term of a position's value for `me` | fitting, the inference engine |
| `move` | a term of a move's value for the player acting | fitting, the inference engine |

A rule carries a **weight per ruleset**, on the ruleset's link to it (see `knowledge/README.md`). Weights are linear
for now: a rule reading x adds weight × x.

## What a rule sees

- **State models.** A scalar is its value: `turn`. A grid, a list or a map is itself: `cell[2, 3]`, `payoff["X"]`
  (see `structure/README.md`).
- **The player solved for**, as `player`, when legal actions are solved for one player.
- **Action parameters**, by name: `row`, `col`. A parameter can't share its name with a state variable.
- **Definitions.** A context's definitions script runs once; every name it leaves — constants, functions, imported
  modules — is visible to its rules.
- **`all_different(*values)`**, true when no two values are equal. The CSP turns a constraint that is a single
  `all_different` call over parameters and parameter-free values into an all-different group.

## How rules are compiled

| Way | Compiled with | Written as | Run by |
|---|---|---|---|
| value | `compile_value(rule, parameters, definitions)` | an expression, or a script that `return`s | `value(compiled, state, parameters, names)`: the result |
| effects | `compile_effects(rule, definitions)` | a script | `apply(compiled, state, parameters)`: the next state |
| definitions | `compile_definitions(rule)` | a script | once per compiled definitions, when a rule seeing them first runs |

- **Value rules** become a function whose arguments are the given parameters the rule reads, in the given order, so a
  caller passes only those; `CompiledRule.arguments` is that list, which the CSP uses as the constraint's scope. A
  parameter it reads that isn't given raises `KeyError`. Its globals are the definitions' names and the state's
  variables, built once per state and kept; a value rule must not change them. `names`, optional, adds names of the
  caller's own, such as a heuristic's `win_chance`; a name a state variable already has raises `ValueError`. The kept
  names are per state and per `names` mapping, by identity, so a caller passes the same mapping for the same state.
  They are kept until the process's memory guard clears them (see `parallel/README.md`).
- **Effects rules** run as a module in a fresh copy of those names plus every parameter. What the script leaves in the
  state's variables is the next state: `turn = other(turn)` or `cell[row, col] = turn`. Other names it assigns are its
  own. An index added under a base the state has (`played[11, 'A'] = 'defect'`) is a new variable, after the state's
  own, in the order the script added it; an index a variable name can't read back the same (`'1'`, `'a,b'`, `True`,
  `1.5`) raises `ValueError`. A new base can't be added this way, and a misspelt plain variable is just a new local
  name.
- A syntax error names the rule and the line; a runtime error's traceback shows the rule's own source.

## What heuristics read

A `position` or `move` rule is a value rule. Besides the state's variables, it reads:

- `me` and `other`: the player the position is valued for, or the player to act for a move, and the next player;
- `action` and the move's parameters, for a `move` rule;
- `win_chance(action)`: the probability that the action ends the game in a win for the player taking it;
- `wins(player, action=None)`: the summed win chances of the actions `player` can take, none outside their turn, now or,
  expected over its outcomes, after `action`;
- `near(action, *offset)`: the value, now, of the variable at that index offset from the indexed variable the action
  sets, or `OUTSIDE`;
- `here`: the position as the inference engine's view, which reads variables as attributes and looks ahead with the
  game's actions (`here.best(me, lambda v1: ...)`; see `inference/README.md`).

A win is an outcome with no legal action left in which the player's payoff is higher than every other player's. The
consequence library works these out with the RBS's own legal moves and outcomes, and caches them until the process's
memory guard clears the cache.

## How heuristics judge

`value(state, player)` is the sum, over the context's `position` rules, of each rule's reading times its weight in the
context. A rule giving `None` is blank, what it reads not being there at that moment, and adds nothing. A rule raising
`KeyError`, `NameError`, `TypeError`, `AttributeError`, `ValueError` or an arithmetic error, or giving something other
than a finite number, is left out. A context with no `position` rule, or none that could be read, values nothing
(`None`), and a search plays that rollout on. `values(state)` gives every player's value; `explain(state, player)` each
rule with what it added. `rate(state, actions)` does the same with the `move` rules, one rating per action.

The value is in the game's own payoff units, so a finished game's payoffs and a heuristic's value compare directly.

## How position rules are fitted

`ValueGenerator.generate(rbs, training, held_out, settings, target, seeds=())` takes position rows, each a position,
the player it is valued for and the payoff to fit; `training/README.md` says where rows come from. Seeds, expressions
given to start from, are tried before the leaves. The fitted rules are declared open and linked, at their weights, into
the position value ruleset of the target's context, which usually inherits the game, so the result is a game that can
be played.

1. **Nothing to fit.** When every training payoff is the same, nothing is fitted and nothing is declared.
2. **Search.** `ExpressionSearch` (see `inference/README.md`) grows expressions of the positions and of what the game's
   actions make of them, for `settings.seconds`, within `settings.memory_bytes` and trying at most
   `settings.candidates` candidates, at the middle price of the sweep. Its kept expressions, read as rules on `here`,
   are the candidate terms.
3. **Fits.** Columns are standardized on the training rows, a column with blanks scaled without centering and its
   blanks read as 0; held-out columns take the training rows' scaling. For each price, from the highest down, starting
   from the previous price's weights, `SparseFitter` minimizes the **mean squared error of the payoffs themselves** plus
   price × the sum of each weight's absolute value times its term's clauses times the share of training rows where the
   term isn't blank, the bias unpriced. It takes accelerated proximal gradient steps (FISTA) of
   `rows / ‖columns with a bias column‖²`; each step shrinks every weight toward 0 and lands it on 0 when it would
   cross, so the terms that don't pay their price drop out. A fit stops after `max_steps`, or once no weight moves by
   more than `tolerance` × the largest of 1 and the largest weight.
4. **Choice.** The fit with the lowest mean squared error on the held-out rows is kept, or on the training rows without
   held-out rows; ties go to the fewest terms. Its nonzero weights, converted back to their terms' own units, are
   declared as `position` rules, and its bias as a constant `position` rule reading `1.0`.

## Usage

A game, from its declared rules:

```python
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.rbs.factory.rbs_factory import create_game

knowledge_base = create_knowledge_base("tictactoe")
rbs = create_game("tictactoe", knowledge_base)   # declares tic-tac-toe's rules where they aren't there yet

state = rbs.start()
actions = rbs.actions(state)                     # 9 placements
state = rbs.outcomes(state, actions[0]).outcomes[0][0]
```

A relaxation:

```python
from openmind.rbs.factory.rbs_factory import create_rule_based_game
from openmind.rbs.service.game_relaxer import GameRelaxer

relaxer = GameRelaxer(knowledge_base)
relaxer.relaxations("tictactoe")                 # each constraint dropped
relaxed = create_rule_based_game(knowledge_base, relaxer.relax("tictactoe", "tictactoe without place is legal, 1"))
relaxed.acting(relaxed.start())                  # (0, 1): without the turn constraint, both players can act
```

Compiling and running a rule by itself:

```python
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.structure.model.grid import Grid
from openmind.world.model.state import State

compiler, runner = RuleCompiler(), RuleRunner(StateNamespaceMapper())
state = State.of(cell=Grid((1, 2), ("X", None)), turn="O")
empty = compiler.compile_value(PythonRule("cell[row, col] is None"), ("row", "col"))
runner.value(empty, state, {"row": 1, "col": 2})   # True
```

## Logs

- `openmind.rbs.service.game_relaxer`: `INFO Relaxed <context> into <relaxation>: <n> rules`.
- `openmind.rbs.service.simulation`: `WARNING The <context> <kind> rule raised`, when a game's ending, record or
  picture rule raises; the logs survive it.
- `openmind.rbs.service.value_generator`:
  - `INFO Every training payoff is <payoff>: nothing to fit`
  - `INFO <c> candidate terms after <g> generations of search (<why it stopped>), looking up to <plies> actions ahead`
  - `INFO Price <price>: <k> of <c> terms kept in <steps> steps, settled|not settled; training loss <loss>, held-out loss <loss>`
  - `INFO Chose price <price>: <r> position rules and a constant of <bias>`
  - `DEBUG <weight> × <term>`, one line per rule

The RBS's roles don't log: they run inside searches.

## Notes

- The compiler and the runner don't log; the services using them log their decisions, quoting rules by their source.
- Compiled code and namespaces don't travel between processes: a pickled `RuleCompiler`, `RuleRunner` or
  `StateNamespaceMapper` arrives without its compiled rules, namespaces or layouts and builds them again (see
  `parallel/README.md`). An RBS travels whole, its rules with it.
- Tests: `service/rule_based_game_tests.py`, `service/simulation_tests.py`, `service/game_relaxer_tests.py`, `factory/rbs_factory_tests.py`,
  `service/consequence_library_tests.py`, `service/sparse_fitter_tests.py`, `service/term_evaluator_tests.py`,
  `service/value_generator_tests.py`, `service/reading_cache_tests.py`, `builder/value_generator_builder_tests.py`.
