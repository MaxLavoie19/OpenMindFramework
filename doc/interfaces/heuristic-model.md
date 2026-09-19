# Interfaces: heuristic and model

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

## Decided so far (2026-09-19)

- **A step is solved by a set of models**, one per task it needs: predicting outcomes, valuing a move for each player,
  valuing a state for each player, … The time management policy picks that set from the time available, the models
  available and what each one trades, precision against processing time. **The policy itself comes at the budget
  step**; until then, the best measured model of each task is used.
- **Everything OMF produces goes to the knowledge base**, for introspection: what a model measured among the rest.
- **A model has a record of its own:** an id, a name, its task, its context and where its data lives. The data itself
  lives in `data/` or the integrator's storage.
- **A port takes the model it runs**, since the services are stateless: `values(model, state)`.

## knowledge: the model record

```python
@dataclass(frozen=True, slots=True)
class ModelRecord:
    name: str                      # such as "position value by rules", "position value by the 2026-09 network"
    task: str                      # position value, move value, prediction, … (constants)
    context: str                   # a context id: the game, a variant, a relaxation
    family: str                    # rules, lookup table, decision tree, network, … (constants, open to any name)
    mechanism: str                 # the mechanism id its readings are sourced by, so its accuracy is measured
    location: str = ""             # a ruleset id, a path under data/, a URI: whatever loads it
    tags: Tags = ()
    id: str = ""                   # model-<uuid4>
```

`KnowledgeBase` gains `model(record)`, `model_by_id`, `model_named(context_id, name)`, `models(context_id, task,
tags)`, `readable_model`, and a `models.jsonl` store.

**Its measurements are beliefs**, in the model's context, tagged with its id: `accuracy` and `spread` as the
epistemology already scores a mechanism (`epistemology/service/accuracy_scorer.py`), plus `processing time`, the mean
seconds a reading took, with the count behind it. The epistemology keeps their evidence, so a model's accuracy is
justified like anything else OMF believes.

## model: the registry and the measuring

```python
class ModelRegistry:                    # stateless
    def register(self, knowledge_base, record) -> ModelRecord: ...
        # keeps the record and its mechanism; declaring the same name in the same context again updates it
    def of_task(self, knowledge_base, context_id, task) -> tuple[ModelRecord, ...]: ...
        # every model of that task in the context, or in the contexts it inherits from
    def best(self, knowledge_base, context_id, task) -> ModelRecord | None: ...
        # the highest measured accuracy, ties to the fastest; None where the task has no model
    def measured(self, knowledge_base, model_id) -> ModelMeasure: ...
        # accuracy, spread, processing time and how many readings they rest on


@dataclass(frozen=True, slots=True)
class ModelMeasure:
    accuracy: float | None
    spread: float | None
    processing_seconds: float | None
    readings: int


class ModelTimer:                       # stateless: times a reading and keeps what it measured
    def timed[T](self, knowledge_base, model_id, read: Callable[[], T]) -> T: ...
```

A reading's cost is measured where it is taken, not assumed. A measurement spanning a debugger pause is left out, as
the debug session says (`debugger.measurement_kept`).

## heuristic: the tasks

```python
class PositionValuer[Model](Protocol):
    """What a state is worth to each player, in the players' order; None where the model knows nothing."""
    def values(self, model: Model, state: State) -> tuple[float, ...] | None: ...


class MoveRater[Model](Protocol):
    """What each action is worth to the player taking it; None for an action the model knows nothing about."""
    def rate(self, model: Model, state: State, actions: tuple[Action, ...], player: str) -> tuple[float | None, ...]: ...
```

They move from `mcts/model/` to `heuristic/model/`. `MovePrior` stays with the search, since a prior is how the search
uses a rating.

**The RBS fills both**, through one stateless service:

```python
class RuleHeuristic:                    # rbs/service: runs a heuristic ruleset's RBS
    def __init__(self, rule_caller: RuleCaller, consequence_library: ConsequenceLibrary) -> None: ...
    def values(self, rbs, state) -> tuple[float, ...] | None: ...
    def value(self, rbs, state, player) -> float | None: ...
    def rate(self, rbs, state, actions, player) -> tuple[float | None, ...]: ...
    def explain(self, rbs, state, player) -> tuple[tuple[RuleRecord, float], ...]: ...
    def describe(self, rbs) -> str: ...
```

This is what the temporary `RuleBasedGame` facade holds today; the facade keeps delegating to it until the search and
the agent drop it, at their own steps.

## What this step leaves alone

- The tactic value heuristic: the tactic step.
- The time management policy that picks the set: the budget step.
- Learned model families (lookup table, decision tree, network): the training step registers them; this step only
  records and measures whatever is registered.

## Named tasks and their ports

Every task OMF knows is named now, in a constant, with a port of its own:

| Task | Port | A model today |
|---|---|---|
| prediction | `Predictor` | a simulation ruleset's effects |
| position value | `PositionValuer` | a position value ruleset |
| move value | `MoveRater` | a move value ruleset |
| tactic value | `TacticValuer` | none yet (the tactic step) |
| inference | `Inferrer` | the expression search and the deducer (the inference step) |
| binning | `Binner` | none yet (the utility step) |
| abstraction | `Abstractor` | none yet (the hierarchy) |
| agent model | `AgentModeller` | none yet (the agent_model step) |
| planning | `Planner` | the search (the search step) |
| time management | `TimeManager` | none yet (the budget step) |

**Proposed:** this step writes the names and the three ports that have a model — prediction, position value, move
value. A task whose port has no model yet gets its port at its own step, where its signature is settled; writing it now
would mean inventing a signature for something unbuilt.

## Timing

Every reading is timed, through `ModelTimer`, and kept: the timings are what each model's **performance profile** is
built from, and the profiles are what a specialized build is later generated from, such as a competitive chess solver
with its parts built in. This step keeps the mean and the count per model; fitting a profile is the training step's.

## As coded (2026-09-19)

- `ModelRecord` (`knowledge/model`), its mapper and store, and the knowledge base's `model`, `model_by_id`,
  `model_named`, `models` and `readable_model`.
- `ModelRegistry` and `ModelTimer` (`model/service`), with `ModelMeasure`; `register_ruleset` records a ruleset as a
  model of its task, which the game declarer and the value generator now do.
- The tasks are named in `knowledge/constant/task_constant.py`; the three ports with a model are written, the others
  come at their own step.
- **The ports take a node, not a state:** `PositionValuer.values(model, node)`, `MoveRater.rate(model, node, actions,
  player)`. `Node` (`heuristic/model/node.py`) holds the state, the game it is in and the features extracted from it,
  `feature(name, extract)` being a memoized call. A search builds one per state; anything else valuing a position
  builds one too.
- `RuleHeuristic` lives in `heuristic/service`, not `rbs`, so that `rbs` doesn't depend on `heuristic`. The facade
  keeps `value`, `values`, `rate`, `explain` and `describe`, building a node and delegating.
- **Transitional:** `Node.game` is the `RuleBasedGame` facade, since extracting a feature needs legal actions and
  outcomes. It becomes the simulation service and its RBS at the search and inference steps.
- **Not coded yet:** nothing reads `ModelRegistry.best` to choose a model; the callers still take the facade's
  heuristics. The search step wires the choice, and the budget step brings the policy that picks the set.
