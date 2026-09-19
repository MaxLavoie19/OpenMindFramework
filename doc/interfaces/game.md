# Interfaces: game, csp, predictor

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

## Decided so far (2026-09-19)

- **The integrator runs the game.** OMF only simulates it, so `game` has no runtime: it is the programmer's declaration
  API and the registry of games.
- **The simulation is how the SDMCTS expands a node:** the CSP gives the legal actions, the heuristics choose between
  them, and the predictor gives the possible outcomes from the state and the set of actions.
- **Rulesets.** A ruleset belongs to a context, and a rule belongs to the rulesets that list it. A rule can be in
  several rulesets. A ruleset has an id, a name, its context, its task, tags, whether it is open, and its rules; it may
  grow, such as with its role.
- **Weights** live on the link between a ruleset and a rule. Every link carries one, used or not. They are linear for
  now: a rule whose value is x contributes a·x.
- **Open and frozen.** Rules and rulesets can each be open or frozen. A frozen ruleset has only frozen rules. A
  heuristic changes copies of frozen rules or rulesets, or has an open ruleset of its own.
- **Who acts** is declared by the game's context, through its variables and constraints. When it isn't an agent's
  turn, it can only plan.
- **Payoffs** come from the predictor: an end state declares a payoff per player.
- **Draws and abandoning** are rules the game declares. OMF recognizes them by the effect of performing them.
- **Who acts in complex game states:** the constraints determine the legal actions for each player in each phase.
- **All players play at the same time, all the time.** In a game played in turns, a player has no options outside their
  turn. OMF knows nothing of turns or phases; the game's constraints say so.
- **Fitted heuristic terms** are linked into their ruleset by their producer.
- **Durations and cooldowns** are declared only in this step. The CSP and the predictor use them later.
- **Continuous ranges** in the CSP come with the tactic step.
- **The predictor** is a port, plus the model running a ruleset's effects rules.

## Where things sit

Two moves keep the packages in dependency order, with no cycles:
- **`RuleCaller` and `RuleCompiler` move from `rbs` to `rule`.** Running a rule on a state isn't the RBS's own: the CSP
  and the predictor's rule model run rules too, and today they import `rbs` for it.
- **`RuleDeclarer` leaves `rbs`.** The RBS reads rules and never declares them. The knowledge base keeps rules and
  rulesets; `game` declares a game's.

```
structure → world → rule → knowledge → epistemology → game → csp → predictor → rbs → …
```

## knowledge: rules and rulesets

`RuleRecord` loses `contexts`: a rule belongs to a context only through the rulesets that list it, and its weight is on
the link.

```python
@dataclass(frozen=True, slots=True)
class RuleRecord:
    name: str
    kind: str                      # constraint, values, effects, ending, duration, cooldown, … (constants)
    rule: Rule
    source: Source
    action: str | None = None
    parameter: str | None = None
    probability: float = 1.0
    open: bool = False
    tags: Tags = ()
    id: str = ""                   # rule-<uuid4>


@dataclass(frozen=True, slots=True)
class RulesetLink:
    rule: str                      # a rule id
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class Ruleset:
    name: str                      # such as "simulation", "position value", "move value", "charge"
    context: str                   # a context id
    task: str                      # the task it is a model of (constants): simulation, position value, move value, …
    source: Source                 # declared by an application, copied, inferred, fitted
    links: tuple[RulesetLink, ...] = ()
    open: bool = False
    tags: Tags = ()
    id: str = ""                   # ruleset-<uuid4>
```

`KnowledgeBase` gains:

```python
def ruleset(self, ruleset: Ruleset) -> Ruleset: ...
    # keeps it and gives it back with its id; refuses, with a WARNING, to link an open rule into a frozen ruleset
def ruleset_by_id(self, ruleset_id: str) -> Ruleset | None: ...
def ruleset_named(self, context_id: str, name: str) -> Ruleset | None: ...
def rulesets(self, context_id: str, task: str | None = None, tags: Tags = ()) -> tuple[Ruleset, ...]: ...
def ruleset_rules(self, ruleset_id: str, kinds: tuple[str, ...] = ()) -> tuple[tuple[RuleRecord, float], ...]: ...
    # the rules it lists, with their weights, in the order they were linked
def link(self, ruleset_id: str, rule_id: str, weight: float = 1.0) -> Ruleset: ...
def unlink(self, ruleset_id: str, rule_id: str) -> Ruleset: ...
def frozen_ruleset(self, ruleset: Ruleset) -> bool: ...
    # declared by an application and not open, like a rule
def copy_ruleset(self, ruleset_id: str, name: str, context_id: str | None = None) -> Ruleset: ...
    # an open copy, same links and weights, sourced as a copy of the original; same context unless one is given
def revise_in(self, ruleset_id: str, rule_id: str, revision: RuleRecord) -> RuleRecord: ...
    # in an open ruleset: an open rule is revised in place; a frozen one is copied, the copy revised and relinked here
    # only, and every other ruleset keeps the original
```

`rules(context_id, kinds, tags)` and `rule_contexts()` go: rules are found through rulesets. `revise` stays, and still
refuses a frozen rule with a WARNING.

## game: the declaration API

`GameDeclarer` replaces `RuleDeclarer`. It declares a game's rules into its context's simulation ruleset, frozen unless
declared open.

```python
class GameDeclarer:
    def __init__(self, knowledge_base: KnowledgeBase, context: str, ruleset: str = SIMULATION,
                 open: bool = False, rule_caller: RuleCaller | None = None) -> None: ...

    def rule(self, name: str, kind: str, rule: Rule, action: str | None = None, parameter: str | None = None,
             probability: float = 1.0, open: bool = False) -> RuleRecord: ...
    def starts_at(self, state: State) -> RuleRecord: ...
    def played_by(self, players: Players) -> RuleRecord: ...
    def definitions(self, rule: PythonRule, effects: bool = False) -> RuleRecord: ...
    def values(self, action: str, parameter: str, rule: Rule) -> RuleRecord: ...
    def constraints(self, action: str, *rules: Rule) -> tuple[RuleRecord, ...]: ...
    def leads_to(self, action: str, rule: Rule, probability: float = 1.0, number: int | None = None) -> RuleRecord: ...
    def together(self, rule: Rule, probability: float = 1.0, number: int | None = None) -> RuleRecord: ...
    def ending(self, rule: Rule) -> RuleRecord: ...
    def lasts(self, action: str, rule: Rule) -> RuleRecord: ...       # seconds the action takes, from state and parameters
    def cools_down(self, action: str, rule: Rule) -> RuleRecord: ...  # seconds before the action is available again
    def record(self, rule: Rule) -> RuleRecord: ...                   # until the codec step, see below
    def picture(self, rule: Rule) -> RuleRecord: ...                  # until the codec step, see below
    def variant_of(self, game: str, leaving: Collection[tuple[str, str]] = ()) -> Ruleset: ...
        # this context's simulation ruleset lists the game's rules except those left; Context.inherits records it
    def done(self) -> str: ...
```

**No structure that isn't generic.** OMF knows nothing of turns, phases, priority, draws, abandoning, clocks or boards:
they belong to the game the integrator implements, and OMF doesn't enforce them. So the declarer has nothing specific
to them:
- `empty(base, value)` goes: what an empty cell holds is the game's own value, which it passes to the grid's `moved`
  and `removed`. The inference readings using it (`mechanics.py`) are revisited at the inference step.
- `timeout(rule)` goes: a clock is one of the game's variables, and running out of time is an effects or ending rule
  over it.
- `record` and `picture` are encoders (text and image) of the game, not rules of its simulation. They stay as they are
  until the codec step, which moves them there.
- A draw offer or abandoning is an ordinary action, recognized by the effect of performing it.

Payoffs need no method of their own: an end state carries the payoff Map `played_by` names, which the effects rules
fill.

**Who acts.** The players allowed to act are a variable part of the state, read by the constraints:
- In chess, `turn = black` constrains white's actions to planning.
- In Magic: The Gathering, `turn = Bob` constrains Alice's actions to planning and playing instants or flash. With
  `turn = Bob, phase = defender declaration`, Alice is allowed to assign defenders.

Planning isn't a game action: it is always available to the agent, outside the simulation.

**All players play at the same time, all the time.** In a game played in turns, such as chess, a player has no
options outside their turn. Turns, phases and priority are the game's own variables and constraints, not a structure of
OMF's. So:
- `Players.to_act` goes: `Players(names, payoff)`. `StateReader.player_to_act` and the Map of flags go with it.
- Every step of the simulation is a joint action of the players who have legal actions in that state. In chess, that
  is one player; in rock paper scissors, both.

**Draws and abandoning** are ordinary actions the game declares. OMF recognizes them by the effect of performing them,
as the predictor gives it: an end state with its payoffs.

**Heuristic rules.** `RuleDeclarer.position` and `move` go with `RuleDeclarer`. The producers of fitted heuristic terms
(training, the relaxer, the value generator) link their rules into a heuristic ruleset through the knowledge base
directly.

## game: the registry

```python
class GameRegistry:
    def names(self) -> tuple[str, ...]: ...
    def declare(self, name: str, knowledge_base: KnowledgeBase) -> str: ...
        # declares the game (or "game/variant") and gives back its context name; an unknown name raises ValueError
```

Games are found only through the `openmind.domains` entry points. The example games (tic-tac-toe, sudoku, prisoner's
dilemma, rock paper scissors) register there from OMF's own `pyproject.toml`, and the if-chain in
`agent/factory/game_factory.py` goes.

## csp

`Solver` keeps its interface. It imports `rule` instead of `rbs`. It doesn't exclude actions still cooling down yet.

## predictor

```python
class Predictor(Protocol):
    def predict(self, state: State, joint: JointAction) -> OutcomeDistribution: ...


class RulePredictor:                   # the model running a ruleset's effects rules
    def __init__(self, knowledge_base: KnowledgeBase, ruleset_id: str, rule_caller: RuleCaller) -> None: ...
    def predict(self, state: State, joint: JointAction) -> OutcomeDistribution: ...
```

A single action is a joint action of one. The effects rules come from the ruleset, not from the caller.

## rbs

`RuleBasedSystem` is built for a ruleset instead of a context, and weighs rules by their links' weights. Its
`actions`, `outcomes` and `joint_outcomes` go through the CSP and a `RulePredictor` as today; which callers move off it
is for the heuristic and search steps.

## Where the code differs from the above (2026-09-19)

- *(Superseded by the stateless services below: one RBS per ruleset.)* **The RBS was built for a context, not for one
  ruleset.** It runs every ruleset of the context (the simulation, the
  position value, the move value), each rule with its weight in its ruleset. A task the context has no ruleset for is
  taken from the contexts it inherits from (`context_rulesets` in `rbs/factory/rbs_factory.py`), so a round of training
  holding only heuristics still plays the game.
- **The predictor** is split in two: `EffectsRunner` runs effects rules it is given (the old `Predictor`), and
  `RulePredictor` reads a ruleset's effects and runs them through it. `RulePredictor.predict_action(state, action)` runs
  one action without a player, for callers that don't know who takes it.
- **Who acts.** `Simulation.acting(rbs, state)` gives the players with at least one legal action;
  `acting_player(rbs, state)` the one player acting. `actions(state)` without a player gives the one acting player's
  actions and raises `ValueError` where several act at once.
- **Heuristic producers** link through `HeuristicTarget(knowledge_base, context)` (`rbs/model/heuristic_target.py`):
  the value generator declares its rules open and links them into the context's position value ruleset.
- **Readings that pretended it was a player's turn are gone** (your option 1): `ConsequenceLibrary.with_turn`,
  `Mechanics.with_turn`, and the relaxation where a player passes. `moves(player)`, `wins(player)` and the look-aheads
  now read the actions the player really has, none outside their turn. Dropping the turn constraint is the relaxation
  that lets a player act now.
- **Readings that needed `empty` are gone**: `Mechanics.empties`, `cleared(at)` and `alone(at)`, and the generated
  `alone(i).mobility(...)` expressions.
- **Clocks.** Without a timeout rule, a player whose clock runs out in self-play or `openmind-play` is logged, their
  clock stops, and the game goes on.
- **Dependency order.** `rule`'s services open debugger frames, and `debug` imports `knowledge`, which imports
  `rule`'s models: a cycle between the packages, though not between modules.

## Stateless services (proposed, 2026-09-19)

Maxime: moving forward, services are stateless, built once and injected; caches are injected; one RBS per ruleset.
Other classes are reworked as their packages come up. For this step:

```python
@dataclass(frozen=True, slots=True)
class RuleBasedSystem:                          # rbs/model: a model, one ruleset's rules with their weights there
    context: str                                # the context's name
    context_id: str
    ruleset: Ruleset
    rules: tuple[tuple[RuleRecord, float], ...]

class Simulation:                               # rbs/service: stateless; the simulation ruleset's RBS runs through it
    def __init__(self, solver: Solver, predictor: RulePredictor, rule_caller: RuleCaller) -> None: ...
    def start(self, rbs) -> State: ...
    def players(self, rbs) -> Players: ...
    def acting(self, rbs, state) -> tuple[int, ...]: ...
    def acting_player(self, rbs, state) -> str: ...
    def actions(self, rbs, state, limit=None, player=None) -> tuple[Action, ...]: ...
    def joint_actions(self, rbs, state) -> tuple[tuple[int, tuple[Action, ...]], ...]: ...
    def outcomes(self, rbs, state, action) -> OutcomeDistribution: ...
    def joint_outcomes(self, rbs, state, joint) -> OutcomeDistribution: ...
    def ended(self, rbs, state) -> str | None: ...

class Predictor(Protocol):                      # the port takes the model it runs
    def predict(self, model, state, joint) -> OutcomeDistribution: ...

class RulePredictor:                            # stateless: reads the effects rules from the RBS it is given
    def __init__(self, effects_runner: EffectsRunner) -> None: ...
    def predict(self, rbs, state, joint) -> OutcomeDistribution: ...
    def predict_action(self, rbs, state, action) -> OutcomeDistribution: ...

def create_rule_based_system(knowledge_base, context, task=SIMULATION) -> RuleBasedSystem: ...
    # the context's ruleset for that task, or the one of a context it inherits from
```

**Injected caches.** The solver's results and the compiled rules move out of `Solver` and `RuleCompiler` into cache
objects built once and passed in: `SolutionCache` and `CompiledRuleCache`, each keeping the memory guard's eviction.

**Composition.** One function builds the services once (`create_simulation()` and what it needs) and callers receive
them.

**The rest, until its own step.** The heuristic side (`value`, `rate`, `explain`, the consequence library), `mcts`,
`inference`, `training`, `agent` and `dashboard` call today's RBS as a service. It stays as a temporary facade,
renamed `RuleBasedGame`, holding the simulation's RBS and the heuristics' RBSs and delegating to the services. Each
package drops it at its own step.

### As coded

- `RuleBasedSystem` (`rbs/model/rule_based_system.py`), `Simulation` (`rbs/service/simulation.py`), the stateless
  `RulePredictor`, `SolutionCache` (`csp/repository`) and `CompiledRuleCache` (`rule/repository`) are as above. The
  compiled rule cache has no memory guard: the compiler never had one.
- `create_simulation()` builds the service; `create_rule_based_system(knowledge_base, context, task)` the model;
  `create_rule_based_game(...)` and `create_game(...)` the facade, which builds its own simulation unless given one.
- The facade keeps `start()` and `players()` read once; the simulation reads them with every call.
- The predictor port is generic in its model: `Predictor[Model].predict(model, state, joint)`.
- `predictor` now imports `rbs`'s model, and `rbs`'s services import `predictor`: a cycle between the packages, though
  not between modules.
