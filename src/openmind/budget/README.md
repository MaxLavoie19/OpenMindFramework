# budget

## Purpose

What there is to spend, and what to spend it on.

Every step OMF takes is solved by a set of models, one per task it needs, and each can be answered quickly and roughly
or slowly and well. The **time management policy** decides both: which model fills each task, and how much the planner
may explore, from the time available, the models available and what each one trades — its measured accuracy against
its measured processing time.

It works at every level: how many agent models to consider, how much to infer, which heuristics to read, which planner
to plan with and how far it may go, and which task to do next.

**Planning is worth nothing where guesses can't be educated.** With no time, no model or no guiding principle, depth is
wasted on moves that are no better than random, and breadth is what might stumble on a win. This is what decides to go
deep, to go broad, or to improvise.

## Content

| File | What it is |
|---|---|
| `model/budget.py` | `Budget(seconds, of_clock=None)`: what a step may spend, and the clock it was carved from |
| `model/time_manager.py` | `TimeManager[Model]`, the time management task: `manage(model, knowledge_base, node, budget, tasks)` |
| `model/allocation.py` | `Allocation(settings, models)`: the model chosen per task and how far to explore; `of(task)` |
| `service/plain_time_manager.py` | `PlainTimeManager(model_registry)`: the bootstrap model |
| `factory/budget_factory.py` | `create_plain_time_manager()` |

## The bootstrap model

`PlainTimeManager` learns nothing. It reads what the registry has measured and spends what it is given:

- **The model per task** is the best measured one (`ModelRegistry.best`): the highest accuracy, ties to the fastest. A
  task nothing has been measured for still gets its best model; a task with no model is left out, and whatever needed
  it does without.
- **The nodes** are the planner's share of the seconds — 80% of them, the rest being what reading the models and
  acting take — divided by what a node has been costing: the measured processing time of the models it will read, or a
  millisecond where nothing has been measured. Never fewer than one.

The training step fits a model of this task from what each allocation actually brought.

## Usage

```python
from openmind.budget.factory.budget_factory import create_plain_time_manager
from openmind.budget.model.budget import Budget
from openmind.knowledge.constant.task_constant import MOVE_VALUE, PLANNING, POSITION_VALUE

allocation = create_plain_time_manager().manage(
    None, knowledge_base, node, Budget(5.0), (PLANNING, POSITION_VALUE, MOVE_VALUE)
)
allocation.settings.nodes      # how far the planner may explore
allocation.of(POSITION_VALUE)  # the model chosen to value a position
```

## Logs

Logger `openmind.budget.service.plain_time_manager`: `DEBUG <seconds> seconds: <n> nodes, models <task> by <model>, …`.

## Notes

- Tests: `service/plain_time_manager_tests.py`.
