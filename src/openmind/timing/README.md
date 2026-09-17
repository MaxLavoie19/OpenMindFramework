# timing

## Purpose

Time as agents and referees see it: where time comes from, a moment to stop by, what a player starts a game with, and
what each step leaves on a player's clock. Nothing here knows a game: a step is whatever a domain says it is, a move in
chess or a slice of time elsewhere.

It is the first part of making OMF's agents time-aware: searches that stop on time, a budget per step, and referees
keeping each player's clock build on it. It is also meant not to block what comes after games: time is seconds on a
timeline, not a count of moves, so a robot whose actions last and overlap, and whose plan keeps a schedule, can build on
the same pieces.

## Content

| File | What it is |
|---|---|
| `model/time_source.py` | `TimeSource`: `now()`, seconds that never run backwards; only differences between readings mean anything |
| `service/wall_time_source.py` | `WallTimeSource`: real elapsed time, from `time.monotonic` |
| `service/manual_time_source.py` | `ManualTimeSource(start=0.0)`: time that moves only when `advance(seconds)` says, for tests and simulations |
| `model/deadline.py` | `Deadline(at, source)`, or `Deadline.after(seconds, source)`: `remaining()` and `passed()`, carried with its source so whatever must stop by then can be handed it alone |
| `model/time_control.py` | `TimeControl(base_seconds, increment_seconds=0.0)`: what each player starts with and gains per step; `clock()` gives the starting clock |
| `model/clock.py` | `Clock(remaining, increment=0.0, flagged=False)`: `after(spent)` gives the clock after a step |
| `model/time_budget_estimator.py` | `TimeBudgetEstimator`: `budget(clock, steps_played)`, the seconds a player's next step may take |
| `service/plain_time_budget_estimator.py` | `PlainTimeBudgetEstimator(expected_steps, reserve_seconds=0.0)`: the plain rule, keeping a reserve out of every budget |
| `constant/timing_constant.py` (reserve) | `DEFAULT_TIME_RESERVE`, 0.05: the share of the base time kept in reserve unless an entry point is told otherwise (`--time-reserve`) |
| `constant/timing_constant.py` | `DEFAULT_EXPECTED_STEPS`, 30: the steps the plain rule expects when an entry point isn't told otherwise (`--expected-steps`) |
| `mapper/time_control_text_mapper.py` | `TimeControlTextMapper`: `from_text("3+2")` and `to_text(control)`, as chess writes a time control |

## Rules

- **Wall time, always.** Games run on real elapsed time. On a shared machine a busy moment leaves a step less done in its
  seconds, and results carry that.
- **A clock reaching zero is flagged**, as a chess flag falls at zero. A flagged clock gains no increment, and takes no
  more steps: `after` raises `ValueError`. Its `remaining` can be below zero, telling by how much the step overran.
- **A time control is written as chess writes one**: the base in minutes, then the increment in seconds. `3+2` is 180
  seconds and 2 a step, `1+0` is 60 seconds and nothing added, `0.5+1` is 30 seconds and 1. A base of zero, a negative
  increment or any other text raises `ValueError`.
- **How long a step may take is its own problem**, behind `TimeBudgetEstimator`, open to models later. The plain rule
  gives the time left over the steps still expected, plus the increment, but never more than the time left, since the
  increment only comes once the step is done: 180 seconds left over 30 steps with a 2 second increment is 8 seconds.
  `expected_steps` is how many steps are still expected at any point of a game, not a game's length, so a game running
  long never divides by zero. It has no floor and no cap of its own. A reserve is kept out of it: the rule shares only
  the time left above the reserve, and a clock at or below the reserve gets a budget of 0, which an agent plays with a
  random move, so its clock never runs out. A flagged clock, negative steps played, fewer than 1 step expected, or a
  negative reserve raise `ValueError`.

## Usage

```python
from openmind.timing.mapper.time_control_text_mapper import TimeControlTextMapper
from openmind.timing.model.deadline import Deadline
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.timing.service.wall_time_source import WallTimeSource

control = TimeControlTextMapper().from_text("3+2")
clock = control.clock()                  # Clock(remaining=180.0, increment=2.0, flagged=False)
clock = clock.after(5.0)                 # Clock(remaining=177.0, ...)

budget = PlainTimeBudgetEstimator(30).budget(clock, 1)   # 7.9 seconds: 177 over 30, plus 2
deadline = Deadline.after(budget, WallTimeSource())
while not deadline.passed():
    ...                                  # think until the deadline
```

## Notes

- Logger `openmind.timing.service.plain_time_budget_estimator`: `INFO A step may take <s> seconds: <s> seconds left above a
  <s> second reserve over <n> steps expected, plus a <s> second increment`, ending `, limited to the time left` where
  that limit gave the budget; `INFO A step may take 0 seconds: <s> seconds left, at or below the <s> second reserve`.
  Nothing else here logs.
- Tests: `model/clock_tests.py`, `model/time_control_tests.py`, `model/deadline_tests.py`,
  `service/wall_time_source_tests.py`, `service/manual_time_source_tests.py`, `service/plain_time_budget_estimator_tests.py`, `mapper/time_control_text_mapper_tests.py`.
