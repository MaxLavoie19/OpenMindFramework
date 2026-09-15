# parallel

## Purpose

Runs work in worker processes, and keeps each process's memory down. Searches, rules, the CSP and the predictor are
Python code, which a single process runs on one core at a time, so games, positions and reference searches that don't
depend on each other are spread over processes instead. Results come back in order and don't depend on the number of
workers.

## Content

| File | What it is |
|---|---|
| `service/task_runner.py` | `TaskRunner(workers, memory_cap=None)`: `map(function, *argument_lists, droppable=False)` calls the function with the items at each index, in worker processes when it has more than one worker, and gives the results in order; `stream(function, count, arguments_for, on_result, droppable=False)` does the same, choosing each call's arguments only when a worker takes it and handing back each result as it ends; `split(items)` cuts a list into slices for the workers |
| `service/memory_guard.py` | `MemoryGuard`: one per process; clears the caches registered with it when the process holds more than its limit, hands freed memory back to the system, and in a worker under a cap watches the memory and ends the worker with a diagnosis |
| `service/memory_meter.py` | `MemoryMeter`: how many bytes this process holds, from `/proc/self/statm`, or its peak where `/proc` is missing |
| `factory/memory_guard_factory.py` | `process_memory_guard()`: this process's guard, the same for every cache in the process |
| `model/memory_cap.py` | `MemoryCap(worker_bytes, diagnosis_directory, grace_seconds=5.0)`: what each worker holds at most, where diagnoses go, and how long a worker may stay over once it asked its caches to clear |
| `model/clearable.py` | `Clearable`: what a cache offers its guard, `memory_entries()` and `clear_memory()` |
| `model/dropped_call.py` | `DroppedCall(index, diagnosis)`: what a droppable call gives when it stayed over the cap in a fresh worker too |
| `model/call_over_memory.py` | `CallOverMemory(index, diagnosis)`: raised for such a call that can't be dropped |
| `model/worker_ended.py` | `WorkerEnded`: raised when a call ends a fresh worker too for another reason |
| `constant/parallel_constant.py` | `DEFAULT_WORKERS` (half the logical CPUs), `SLICES_PER_WORKER` (4), `START_METHOD` (`spawn`), `PARENT_CHECK_SECONDS` (5), `STOP_SECONDS` (5); `HALF_THE_MEMORY` and `DEFAULT_PROCESS_MEMORY` (that half shared between the logical CPUs), `MEMORY_CHECK_INTERVAL` (1,000 entries), `MEMORY_WATCH_SECONDS` (1), `MEMORY_GRACE_SECONDS` (5), `MEMORY_LOG_SECONDS` (60), `MEMORY_EXIT_CODE` (86), `DIAGNOSIS_CALLS` (50), `DIAGNOSIS_TYPES` (20), `DEFAULT_RERUN_LINES` (20) |

## How work runs

- With one worker, or a single call, everything runs in this process, in order.
- Otherwise up to `workers` processes start for the call, as fresh interpreters (`spawn`) on every platform, and end
  when the results are in. Each worker has a pipe of its own to this process, which hands it one call at a time. The
  function, its arguments and its results travel by pickling: the function must be importable, or the method of an
  object that pickles.
- Services that keep caches leave them behind when pickled, so any service can travel with its dependencies and
  rebuild what it needs in the worker: compiled rules and namespaces (`RuleCompiler`, `RuleRunner`,
  `StateNamespaceMapper`), solver results (`Solver`), consequence lookups (`ConsequenceLibrary`), views (`Mechanics`)
  and exact values (`ExactSearch`).
- Each worker sends its log records, from the root logger's level up, through its pipe, and the root logger's handlers
  here write them. A run still writes one log; lines of different workers interleave.
- `split` gives one slice to one worker, otherwise up to `SLICES_PER_WORKER` slices per worker, so that work set up once
  per slice, such as building an agent, is set up rarely.
- A call that raises in a worker raises here, with the worker's traceback as a note; the other workers are terminated.
- **Streams.** Calls wait in a queue and a worker takes the next one when it's free. With `stream`, `arguments_for(index)`
  runs in this process at that moment, and `on_result(index, result)` runs as each call ends, in the order they end, a
  dropped call's `DroppedCall` included; so a call's arguments can depend on every result finished before it starts, as
  games between arms choose their arms. A call run again in a fresh worker keeps its arguments. With one worker, calls
  run here one after another, each result handed back before the next call's arguments are chosen.

## Memory

- **Caches.** The solver, the rule runner, the consequence library and the mechanics register their caches with their
  process's `MemoryGuard` and tell it before keeping an entry. Every `MEMORY_CHECK_INTERVAL` entries the guard reads the
  process's memory, and above its limit it clears every registered cache and hands freed memory back to the system
  (`malloc_trim`). There is no entry count: memory decides. A process's limit is `DEFAULT_PROCESS_MEMORY` until it is
  given one; `openmind-train-values` and `openmind-distill-values` give theirs the expression search's `--memory`.
- **After each call,** a worker collects its garbage and hands freed memory back to the system, since Python keeps
  freed memory otherwise.
- **The cap.** Under a `MemoryCap`, each worker's limit is the cap, and a thread reads the worker's memory every
  `MEMORY_WATCH_SECONDS`. Over the cap, it asks the caches to clear at their next entry: the thread can't safely empty a
  cache another thread is using. Still over the cap after the cap's grace, the worker writes a diagnosis, tells this
  process, and ends itself with `MEMORY_EXIT_CODE`.
- **The diagnosis,** `<diagnosis directory>/<YYYY-MM-DD_HH-MM-SS>-worker-<pid>.txt`, has the memory when the worker was
  first seen over the cap and at the end, the caches last cleared with their copies and entries, the most numerous
  objects the garbage collector tracks, the call being run and the latest calls with the memory held after each. The
  call itself is pickled beside it as `.pickle`, to run again alone with `openmind-rerun-call` (see
  `entrypoint/README.md`).

## Workers that end

- A worker that ends during a call, over its cap or killed from outside, loses only that call: a fresh worker takes its
  place and runs the call again. Its pipe is its own, so nothing the other workers send is left half written.
- A call that ends a fresh worker too:
  - over the cap both times: with `droppable`, gives a `DroppedCall` in its place and the other calls go on; otherwise
    raises `CallOverMemory`;
  - for another reason: raises `WorkerEnded`.
- A worker checks every `PARENT_CHECK_SECONDS` that the process that started it is still its parent, and ends itself
  once it isn't, so a killed run leaves no workers behind.

On maxime-cinamon on 2026-09-15, earlyoom killed training runs whose 20 workers reached about 2.9 GB each; a run killed
that way left 9 workers blocked for hours, and a run whose worker died waited forever. The cap, the parent watch and the
workers' own pipes keep each of these from happening again.

Who runs work in workers, and how their results stay the same whatever the number of workers:

| Service | Work | Seeds | Droppable |
|---|---|---|---|
| `training/service/self_play.py` | self-play games; games between arms through `stream` | each game draws an agent seed and an outcome seed up front; a game's arms depend on the games finished before it starts, so on the number of workers | yes: the game is left out |
| `evaluation/service/match_runner.py` | baseline games | each game draws a policy seed and an outcome seed up front | yes: the game isn't counted |
| `training/service/position_ponderer.py` | positions deduced, and decisive games walked back from their ends, one game per call | none needed: a deduction depends only on its position | yes: the position isn't pondered, or the game isn't walked |
| `rbs/service/term_evaluator.py` | terms evaluated on row slices | none needed | no: the expression search stops with "the memory budget ran out" |
| `evaluation/service/evaluator.py` | positions searched at each budget, reference searches | every search uses the evaluation's seed | no |
| `rbs/service/condition_evaluator.py` | rule conditions checked on search rows, the rows split in slices, for discovery, validation, coverage and primitives | none needed: a condition's value depends only on its row | no |

## Usage

```python
from pathlib import Path

from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.service.task_runner import TaskRunner

TaskRunner(4).map(pow, [2, 3, 4], [2, 2, 2])   # [4, 9, 16], computed in worker processes
TaskRunner(4, MemoryCap(2 * 1024**3, Path("data/log/memory"))).map(play, games, droppable=True)
```

From the terminal, `openmind-distill` and `openmind-evaluate` take `--workers N`; `openmind-distill-values` and
`openmind-train-values` also take `--worker-memory GB` (see `entrypoint/README.md`).

## Logs

- `openmind.parallel.service.task_runner`:
  - `DEBUG Running <n> calls in <w> worker processes`
  - `WARNING Worker <pid> stayed over its memory cap of <bytes> bytes during call <i> of <function> and ended; diagnosis <path>`
  - `WARNING Worker <pid> ended abruptly with exit code <code> during call <i> of <function>`
  - `WARNING Running call <i> of <function> again in a fresh worker`
  - `WARNING Dropped call <i> of <function>: it stayed over the memory cap in a fresh worker too; diagnosis <path>`
  - `WARNING Could not read what worker <pid> sent: <error>`
- `openmind.parallel.service.memory_guard`:
  - `INFO Cleared <n> caches holding <entries> entries as the memory watch asked|on reading the memory: this process held <bytes> bytes, over its limit of <bytes>, and now holds <bytes>; <k> clears since the previous line`,
    at most once every `MEMORY_LOG_SECONDS` unless the watch asked
  - `WARNING This worker holds <bytes> bytes, over its memory cap of <bytes> for more than <seconds> seconds, during call <i>: ending it; diagnosis <path>`

## Notes

- Every worker holds its own caches: memory grows with the number of workers. The 4 in a row evaluation peaked at
  2.4 GB in one process.
- Tests: `service/task_runner_tests.py` (a call holding 400 MB under a 200 MB cap, dropped or raised),
  `service/memory_guard_tests.py`, `service/memory_meter_tests.py`; end-to-end, same results with 1 and 2 workers:
  `test/end_to_end/distill_tictactoe_tests.py`, `test/end_to_end/evaluate_tictactoe_tests.py`.
