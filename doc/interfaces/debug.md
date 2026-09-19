# Interfaces: debug and logs

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

## Decided so far

- **Ownership:** `debug` owns logging and warnings, and the dashboard is its viewer.
- **Verbosity is per session:** minimal, or targeted by package or service, context, reasoning frame, or tags.
- **Warnings** cover conflicting rules and observations a frozen rule can't explain. Conflicts also become tasks
  (epistemology).
- **The debugger** has interrupts, a stack, and breakpoints that can be conditional on the game, such as "break when you
  reach an evaluation for a board position that allows en passant".
- **The stack is two stacks, linked:** the Python call stack and OMF's reasoning stack (task → tactic → search node →
  evaluation → rule). Services open explicit frames.
- **The debug session decides** whether OMF's clocks freeze or keep running while paused, and whether measurements taken
  across a pause are kept for training and timing.
- **Setup:** a session is an object built in code, and it can also be loaded from a file.
- **Pausing:** a terminal console now, and the dashboard once it is rebuilt.

## `debug`

It depends on `structure`, `world`, `rule` and `knowledge`, since warnings are kept in the knowledge base (D3). Every
package above it can open frames and raise warnings. The packages below it (`structure`, `world`, `rule`, `knowledge`)
raise warnings through logging: a handler the debugger installs turns every OMF `WARNING` record into a debugger
warning, such as the knowledge base refusing to revise a frozen rule.

**D5 (decided): through logging.** Any OMF `WARNING` log line becomes a debugger warning, and so a direct experience in
the knowledge base.

### Models

```python
@dataclass(frozen=True, slots=True)
class Target:
    """What a verbosity setting applies to. Every field given must match; None matches anything."""
    logger: str | None = None                  # a package or service: "openmind.mcts"
    context: str | None = None                 # a context id or name
    frame: str | None = None                   # a kind of reasoning frame: "evaluation", "tactic", "task"
    tags: tuple[tuple[str, Value], ...] = ()


@dataclass(frozen=True, slots=True)
class Verbosity:
    level: int                                 # logging.DEBUG, INFO, …
    target: Target = Target()


@dataclass(frozen=True, slots=True)
class Breakpoint:
    """Stops when a frame of that kind opens and its condition holds. The condition is a Python rule, like a game's, that
    reads the frame's state (its models by name, as rules see them), `frame` (kind, context, tags, details) and the
    game's definitions: for chess, `frame.kind == "evaluation" and en_passant_square is not None`."""
    name: str
    frame: str
    condition: PythonRule | None = None


@dataclass(frozen=True, slots=True)
class DebugSession:
    name: str
    verbosities: tuple[Verbosity, ...] = (Verbosity(logging.INFO),)     # minimal by default
    breakpoints: tuple[Breakpoint, ...] = ()
    clocks_while_paused: str = "freeze"         # "freeze" or "run"
    keep_paused_measurements: bool = False
    log_directory: Path = Path("data/log")


@dataclass(frozen=True, slots=True)
class Frame:
    """One step of OMF's reasoning, with what the step works on, and the Python frame it was opened from."""
    kind: str                                  # "task", "tactic", "search node", "evaluation", "rule", …
    context: str | None
    state: State | None
    details: tuple[tuple[str, Value], ...]
    tags: tuple[tuple[str, Value], ...]
    python: str                                # file:line of the code that opened it


@dataclass(frozen=True, slots=True)
class Warning:
    kind: str                                  # "conflicting rules", "a frozen rule can't explain an observation", …
    message: str
    context: str | None
    ids: tuple[str, ...]                       # the rules, experiences or beliefs involved
```

### Services

```python
class Debugger:
    """One per process. A no-op, and cheap, when its session asks for nothing."""
    def start(self, session: DebugSession, knowledge: KnowledgeBase | None = None) -> None: ...   # logging per the session; warnings kept in the knowledge base when one is given
    def frame(self, kind: str, *, context: str | None = None, state: State | None = None,
              details: Mapping[str, Value] = {}, tags: Tags = ()) -> ContextManager[Frame]: ...
    def stack(self) -> tuple[Frame, ...]: ...                # the reasoning stack, innermost last
    def warn(self, warning: Warning) -> None: ...            # logged at WARNING and kept for the viewer
    def warnings(self) -> tuple[Warning, ...]: ...
    def interrupt(self) -> None: ...                         # pauses at the next frame
    def paused(self) -> bool: ...                            # whether time is paused, for clocks and measurements


class Console:
    """The terminal side of a pause: shows the reasoning stack and the Python stack, linked; continue, step to the
    next frame, step out, inspect the frame's state, list breakpoints."""


class SessionJsonMapper:
    """A session to and from a JSON file."""
```

- **Logging:** the entrypoints' and the test plugin's own logging setup move into `Debugger.start`. Tests keep saving
  their logs per test, through a session the plugin builds.
- **Frames first appear** in the places that exist now: a task in the continuous trainer, a search, each search node
  expanded, each evaluation (position value, move value), each rule run by the RBS. More are added as components are
  rebuilt.

## Open points

- **D1 (decided): the worker waits.** A worker that hits a breakpoint sends its frame to the main process and waits; the
  main process opens the console for it.
- **D1, as asked: worker processes.** Searches and games run in worker processes, where a terminal console can't open. Options:
  - breakpoints stop only in the main process, and a worker logs that it would have stopped;
  - a worker that hits a breakpoint sends its frame to the main process and waits;
  - running under a session with breakpoints forces a single process.
- **D2 (decided): acceptable.** Targeted debugging may slow a run; a session that targets nothing costs close to
  nothing.
- **D2, as asked: frame cost.** Opening a frame for every rule run and every node is heavy when a session targets them. Is it
  acceptable that targeted debugging slows a run, as long as a session that targets nothing costs close to nothing?
- **D3 (decided): the knowledge base too**, as direct experiences of the debugger, so epistemology and tasks can use
  them.
- **D3, as asked: where warnings go besides the log.** Options: kept in memory for the viewer only; also written to the
  knowledge base, as direct experiences of the debugger (so epistemology and tasks can use them).
- **D4 (decided): JSON**, like the knowledge base's files.
