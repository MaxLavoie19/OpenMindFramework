# parallel

## Purpose

Runs work in worker processes. Searches, rules, the CSP and the predictor are Python code, which a single process runs
on one core at a time, so games, positions and reference searches that don't depend on each other are spread over
processes instead. Results come back in order and don't depend on the number of workers.

## Content

| File | What it is |
|---|---|
| `service/task_runner.py` | `TaskRunner(workers)`: `map(function, *argument_lists)` calls the function with the items at each index, in worker processes when it has more than one worker, and gives the results in order; `split(items)` cuts a list into slices for the workers |
| `constant/parallel_constant.py` | `DEFAULT_WORKERS` (half the logical CPUs), `SLICES_PER_WORKER` (4), `START_METHOD` (`spawn`) |

## How work runs

- With one worker, or a single call, everything runs in this process, in order.
- Otherwise a pool of up to `workers` processes starts for the call, as fresh interpreters (`spawn`) on every platform,
  and stops when the results are in. The function, its arguments and its results travel by pickling: the function must
  be importable, or the method of an object that pickles.
- Services that keep caches leave them behind when pickled, so any service can travel with its dependencies and
  rebuild what it needs in the worker: compiled rules and namespaces (`RuleCompiler`, `RuleRunner`,
  `StateNamespaceMapper`), solver results (`Solver`), consequence lookups (`ConsequenceLibrary`) and exact values
  (`ExactSearch`).
- Each worker sends its log records, from the root logger's level up, to this process, where the root logger's
  handlers write them. A run still writes one log; lines of different workers interleave.
- `split` gives one slice to one worker, otherwise up to `SLICES_PER_WORKER` slices per worker, so that work set up once
  per slice, such as building an agent, is set up rarely.

Who runs work in workers, and how their results stay the same whatever the number of workers:

| Service | Work | Seeds |
|---|---|---|
| `training/service/self_play.py` | self-play games | each game draws an agent seed and an outcome seed up front |
| `evaluation/service/match_runner.py` | baseline games | each game draws a policy seed and an outcome seed up front |
| `evaluation/service/evaluator.py` | positions searched at each budget, reference searches | every search uses the evaluation's seed |
| `rbs/service/condition_evaluator.py` | rule conditions checked on search rows, the rows split in slices, for discovery, validation, coverage and primitives | none needed: a condition's value depends only on its row |

## Usage

```python
from openmind.parallel.service.task_runner import TaskRunner

TaskRunner(4).map(pow, [2, 3, 4], [2, 2, 2])   # [4, 9, 16], computed in worker processes
```

From the terminal, `openmind-distill` and `openmind-evaluate` take `--workers N` (see `entrypoint/README.md`).

## Logs

- `openmind.parallel.service.task_runner`: `DEBUG Running <n> calls in <w> worker processes`

## Notes

- Every worker holds its own caches: memory grows with the number of workers. The 4 in a row evaluation peaked at
  2.4 GB in one process.
- Tests: `service/task_runner_tests.py`; end-to-end, same results with 1 and 2 workers:
  `test/end_to_end/distill_tictactoe_tests.py`, `test/end_to_end/evaluate_tictactoe_tests.py`.
