# model

## Purpose

Which models perform which task, and how each one has been measured.

A **model** is one way to perform a task: a ruleset run by the RBS, a lookup table, a decision tree, a network, or
whatever an integrator registers. Every step OMF takes is solved by a set of models, one per task it needs — predicting
outcomes, valuing a move for each player, valuing a state for each player — and the time management policy picks that
set from the time available, the models available and what each one trades, precision against processing time. Until
that policy arrives (the budget step), whatever reads a task takes the best measured model of it.

Everything OMF produces goes to the knowledge base, so a model is a record there and what it is measured at are beliefs
about it: the epistemology weighs them like anything else the agent believes.

## Content

| File | What it is |
|---|---|
| `service/model_registry.py` | `ModelRegistry(accuracy_scorer)`: `register(knowledge_base, record)`, `register_ruleset(knowledge_base, ruleset, name=None)`, `of_task(knowledge_base, context_id, task)`, `best(knowledge_base, context_id, task)`, `measured(knowledge_base, model)` |
| `service/model_timer.py` | `ModelTimer(time_source=None)`: `timed(knowledge_base, model, read)` times a reading and keeps the seconds; `spent(...)` keeps a reading's seconds on their own |
| `model/model_measure.py` | `ModelMeasure(accuracy, spread, processing_seconds, readings)`: what a model has been measured at; None for what isn't measured yet |
| `constant/model_constant.py` | The families OMF knows (`rules`, `lookup table`, `decision tree`, `ensemble`, `network`), and what the timing belief is called |
| `factory/model_factory.py` | `create_model_registry()`, `create_model_timer()` |

The tasks themselves are named in `knowledge/constant/task_constant.py`, and a model record lives in
`knowledge/model/model_record.py` (see `knowledge/README.md`).

## Usage

```python
from openmind.knowledge.constant.task_constant import POSITION_VALUE
from openmind.knowledge.model.model_record import ModelRecord
from openmind.model.constant.model_constant import NETWORK
from openmind.model.factory.model_factory import create_model_registry, create_model_timer

registry, timer = create_model_registry(), create_model_timer()
network = registry.register(
    knowledge_base,
    ModelRecord("the 2026-09 network", POSITION_VALUE, context_id, NETWORK, mechanism_id, "data/model/2026-09.pt"),
)

values = timer.timed(knowledge_base, network, lambda: read_the_network(state))
registry.best(knowledge_base, context_id, POSITION_VALUE)     # the model measured most accurate, ties to the fastest
registry.measured(knowledge_base, network)                    # ModelMeasure(accuracy, spread, seconds, readings)
```

A ruleset is a model of its task: `register_ruleset` records it, found again by the ruleset's id. The game declarer
registers a game's simulation ruleset, and fitting registers the position value ruleset it fills.

## How a model is measured

- **Accuracy and spread** come from the epistemology (`epistemology/README.md`). A model's readings are sourced by its
  mechanism, so when an anchor settles what turned out true, the accuracy scorer scores that mechanism: the share it
  got right, and the root of its mean squared error on numbers. A model never measured has no accuracy, not a low one.
- **Processing time** is measured here: every reading is timed, and the belief holds the mean with the seconds and the
  readings behind it. Every reading, not a sample: the timings are what a model's performance profile is built from,
  and the profiles are what a build shipping with its parts built in is generated from. A reading a debugger pause
  spans is left out.
- **The best** is the highest accuracy, ties going to the fastest, then to the one registered first.

## Logs

- `openmind.model.service.model_registry`: `INFO Registered <model> as a model of <task> in <context>`;
  `DEBUG The best model of <task> in <context> is <model>`.
- The knowledge base logs the record itself at DEBUG, and the beliefs as they are set.

## Notes

- Tests: `service/model_registry_tests.py`, `service/model_timer_tests.py`.
