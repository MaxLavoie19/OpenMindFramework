# debug

## Purpose

Logging, warnings and a debugger for OMF, run as a debug session.

A **session** says which log lines are kept: minimal by default, or targeted by package or service, by context, by
kind of reasoning frame, or by tags. It also says where to stop (breakpoints), whether OMF's clocks freeze or keep
running while paused, and whether measurements taken across a pause count for training and timing. A session is built
in code, or read from a JSON file (`--debug-session FILE` on every command).

Services open **reasoning frames** as they work: a task, a search, a search node, an evaluation, a rule. The frames
make OMF's reasoning stack, each linked to the Python line that opened it. A breakpoint stops when a frame of its kind
opens and its condition, a Python expression over the frame's state, holds. For chess, that could be
`frame.kind == "evaluation" and en_passant is not None`. Ctrl+C interrupts at the next frame. A pause shows both stacks
in the terminal console and waits for `continue`, `step` or `out`. A worker process that stops sends its pause to the
process that started it, and waits for the answer. Without a breakpoint or a targeted verbosity, opening a frame costs
one call and does nothing.

Every OMF WARNING log line becomes a **warning**. It is kept for the viewer, and, when the session was started with a
knowledge base, it is kept there too as a direct experience of the debugger. That way the knowledge base refusing to
revise a frozen rule, or a conflict epistemology finds, reaches the debugger without the lower packages knowing it.

## Content

| File | What it is |
|---|---|
| `model/target.py` | `Target(logger=None, context=None, frame=None, tags=())`: what a verbosity applies to |
| `model/verbosity.py` | `Verbosity(level=INFO, target=Target())` |
| `model/breakpoint.py` | `Breakpoint(name, frame, condition=None)`: the kind of frame it stops at, and a Python condition |
| `model/debug_session.py` | `DebugSession(name, verbosities, breakpoints, clocks_while_paused, keep_paused_measurements, log_directory, log_file)` |
| `model/frame.py` | `Frame(kind, context, state, details, tags, python)`: one step of the reasoning stack |
| `model/pause.py` | `Pause(reason, stack, python_stack, process)` |
| `model/warning.py` | `Warning(kind, message, context, ids)` |
| `constant/debug_constant.py` | `FREEZE`, `RUN`; the console's commands; the debugger's name in the knowledge base; `LOG_FORMAT` |
| `service/debugger.py` | `Debugger`: `start(session, knowledge=None, pauser=None)`, `attach(handler)`, `stop()`, `frame(kind, …)`, `stack()`, `interrupt()`, `answer(pause)`, `tracking`, `paused()`, `frozen_seconds()`, `measurement_kept(started)`, `warn(warning)`, `warnings()` |
| `service/verbosity_filter.py` | `VerbosityFilter`: keeps the log lines some verbosity wants |
| `service/warning_handler.py` | `WarningHandler`: turns OMF WARNING lines into warnings, kept in the knowledge base when there is one |
| `service/console.py` | `Console(breakpoints, read=input, write=print)`: takes a pause in the terminal |
| `mapper/session_json_mapper.py` | `SessionJsonMapper`: a session to and from JSON |
| `factory/debugger_factory.py` | `process_debugger()`: this process's debugger |

## Usage

A session file:

```json
{
  "name": "centre",
  "verbosities": [{"level": "INFO"}, {"level": "DEBUG", "frame": "evaluation"}],
  "breakpoints": [{"name": "centre taken", "frame": "evaluation", "condition": "cell[2, 2] == 'X'"}],
  "clocks_while_paused": "freeze",
  "keep_paused_measurements": false
}
```

```bash
.venv/bin/openmind-play tictactoe --agent O --debug-session centre.json
```

In code:

```python
from openmind.debug.factory.debugger_factory import process_debugger

with process_debugger().frame("evaluation", context=context, state=state, details={"player": player}):
    ...
```

## Logs

- `openmind.debug.service.debugger`: `INFO Debug session <name>: <verbosities>; <n> breakpoints; clocks <freeze|run>
  while paused`, and `INFO Paused <seconds> seconds (<reason>); <answer>` after each pause.
- Every warning also shows as the WARNING line that made it.
