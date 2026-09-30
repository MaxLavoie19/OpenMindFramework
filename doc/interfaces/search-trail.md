# Interfaces: a search that pays for what changed

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

Changes how `BacktrackingSearch` holds domains. Nothing about what a constraint is, how `Solver` classifies one,
or what any of it means changes, and `Solver`'s public signatures stay as they are.

## What made this necessary

A positioning problem — which part of a wall you are looking at, read off stapled colored markers — turns out to
be a de Bruijn sequence, and a de Bruijn sequence is a constraint problem with one variable per position. Seven
colors and three markers in view is 343 variables, which the solver handles. Six markers in view is 117 649, and
there the solver does not get started.

Measured standalone, CPython 3.12, one search node at 117 652 variables, before any filtering runs at all:

```
    line   what it does                                   per node
      74   min(open_variables, key=...)                   20.81 ms
      70   rebuild open_variables by scanning every name   8.57 ms
     107   touched: compare every domain by identity        8.57 ms
      79   narrowed = dict(domains)                         2.61 ms
      93   two sum(map(len, ...)) passes for a statistic    2.62 ms
                                                          --------
           bookkeeping before a single value is filtered   43.0 ms
```

All of it is the same defect wearing five hats: **every node pays for every variable that exists, rather than for
the ones that changed.** The per-element constant is not even flat — it doubles from 83 ns at 2 404 variables to
177 ns at 117 652, because the dictionaries stop fitting in cache. Solving time goes as `3.0e-7 * N**2` seconds,
which puts a ten-minute ceiling at about 45 000 variables and makes 117 649 a matter of hours of bookkeeping
before any search happens.

Two more, off the per-node path but fatal at the same scale:

```
    line              what it does                            cost at 117 651 variables
  solver.py:225   `operand.source in names`, names a list      O(M) per operand, O(M^2) overall, ~1 s
  solver.py:193   a value-to-index map per variable            N*Q entries: 4.0e7 at Q=343, multi-GB
```

`solver.py:193` is built only to sort the solutions, and there is nothing to sort when fewer than two were asked
for.

### And the search cannot reach these depths at all

`backtrack` calls itself once per assignment (`backtracking_search.py:94`), so the Python stack has to be as deep
as the number of variables. Sudoku assigns about fifty and tic-tac-toe nine, so this has never shown. The marker
wall assigns 2 401 and then 117 649, and the default recursion limit is a thousand: **the solver raises
`RecursionError` long before it reaches any of the costs above.** Confirmed by running the proposed search
recursively — it dies at 2 401 variables — and then iteratively, where the same problem solves in 0.19 s and the
largest cell reaches a stack depth of 117 304 in 64 s.

Raising `sys.setrecursionlimit` is not the answer: the C stack overflows and the process dies without an
exception. **So the search becomes iterative, with an explicit stack of frames.** That is not a separate change
bolted on — a trail plus an explicit stack is the standard shape, and the trail is what makes the explicit stack
cheap, because a frame then holds two integer marks rather than a copy of every domain.

This was missed in the first two passes over the code, including by the design pass, and it is the finding that
most changes the work.

**Nothing here is specific to the marker wall.** Tic-tac-toe has nine variables and sudoku about fifty, so the
`N**2` never showed. Chess solves for legal moves at every search node it opens, which the game README already
calls the slowest thing OMF does.

## What changes

### The domains, and the trail, become a repository

Today the domains are a `dict[str, frozenset[Value]]` threaded through the recursion and copied at every
assignment. Proposed: one repository holding the values every variable has left and the record of what was
removed, so backtracking undoes a node instead of building a new one.

Removals are stored values, so they live in a repository, beside `SolutionCache`. What a propagator is handed is
data; the propagators keep nothing, and neither does the search.

### A propagator stops returning domains

```python
def propagate(self, domains: dict[str, frozenset[Value]], group: AllDifferentGroup) -> dict[str, frozenset[Value]]:
```

becomes

```python
def propagate(self, repository: DomainRepository, group: AllDifferentGroup) -> None:
```

It narrows what it is given and raises `Wipeout` as it does now. **Returning nothing is the point**: line 107
exists only because a propagator answers with a whole new dictionary and never says which variables it touched.
The repository knows, because it wrote the trail.

### The constraint index is built once, not per call

`ArcConsistency.propagate` rebuilds its `by_variable` map from every table on every call — once per node
(`arc_consistency.py:20-23`). `_propagate` does the same work differently, testing `touched.isdisjoint(scope)`
against every group and every constraint in the space, which is O(total scope size) and not, as I first read it,
O(number of constraints): one constraint scoping every variable costs a full pass by itself.

Both become one index, built once by `search()` and read by everything.

### Variable selection becomes a heap

`min(...)` over a freshly built list is 29 ms of the 43. A heap keyed exactly as today — `(domain size, -degree,
position)` — gives the same answer, because `position` makes every key unique, so the minimum is the same
variable. Entries go stale as domains shrink and are dropped when picked.

**This is the whole regression net for the rewrite.** `backtracking_search_tests.py:93-119` pin which variable
the search tries first in given situations. If those tests pass unchanged, the ordering was preserved.

### `pruned_values` is read off the trail

Two `sum(map(len, ...))` passes become one subtraction of trail heights. The accounting must match today's
exactly, and today's is subtler than it looks: the count excludes the assignment's own collapse to one value
(`backtracking_search.py:80` happens before the measurement) and excludes everything a dead end removed (the
`continue` at line 92 skips the measurement). So a node marks the trail **twice** — once before fixing the
variable, to know where to unwind to, and once after, to know what propagation itself removed.

### The recursion becomes an explicit stack

`backtrack(domains)` becomes a loop over a list of frames, each frame holding the variable it is deciding, the
values it has left to try, and the two trail marks to unwind to. Picking the next variable pushes a frame;
exhausting a frame's values unwinds it and pops.

The frames are what the Python stack was, so nothing about the search order changes — and the ordering tests stay
the net for that.

### Undoing a dead end becomes explicit

Today a dead end is free: the copy is discarded. With a trail, a propagator that raises `Wipeout` has already
removed values, so the search must unwind to its mark in the `except` branch. This is new work that the old
design got for nothing, and it is where a mistake would corrupt a search silently rather than crash it.

## Repositories

**`csp.repository.domain_repository.DomainRepository`** — the values every variable has left, and the trail.

```python
values: dict[str, set[Value]]

@property
def height(self) -> int
    """How many removals have been recorded, which is the mark to unwind back to."""

def remove(self, variable: str, value: Value) -> None
    """Takes the value out of the variable's domain and records it. Raises Wipeout when nothing is left."""

def fix(self, variable: str, value: Value) -> None
    """Removes every other value from the variable's domain, recording each."""

def undo_to(self, height: int) -> None
    """Puts back every value recorded since that height, most recent first."""

def changed_since(self, height: int) -> tuple[str, ...]
    """The variables something was removed from since that height, each once."""

def settled(self, variable: str) -> bool
def only(self, variable: str) -> Value
```

`changed_since` walks the trail from the mark, so it costs what changed. That is the whole reason line 107 goes.

**Open.** Whether `values` holds `set` or something narrower. Measured at 4.03e7 domain elements — the heaviest
cell of the marker wall — a `set` costs 4 958 MB and a `tuple` 1 568 MB, because CPython over-allocates set
tables about five times. Sets fit the machine and are what the propagators want; a sparse-set representation
would cut it and is real complexity. I would ship sets and measure.

**`csp.repository.variable_queue.VariableQueue`** — the open variables, smallest domain first.

```python
def __init__(self, degree: Mapping[str, int], position: Mapping[str, int]) -> None
def push(self, variable: str, size: int) -> None
def pick(self, repository: DomainRepository) -> str | None
    """The open variable with the fewest values left, ties to the highest degree then to declaration order, or
    None where every variable is settled. Entries whose recorded size no longer matches are dropped as they
    surface."""
```

Pushed on: the initial build, every removal, and every undo. Measured 5.65 M pushes on the largest cell that
runs, inside its 4.6 s.

**Open.** Whether the repository pushes to the queue itself or the search does it after each propagation round.
The first keeps them in step and couples two repositories; the second keeps them apart and can forget.

## Models

~~**`csp.model.trailed.Trailed`**~~ — proposed as a protocol so `search()` could mark and unwind every trailed
repository without knowing how many there are. **Built and then deleted: nothing referenced it.** `search()` marks
the domain repository and the circuit chains by name, and the chains are handed to their propagator as the
`CircuitChains` it needs rather than as a protocol. A second kind of trail would be the moment to bring it back.

**`csp.model.constraint_index.ConstraintIndex`** — frozen, per search: for each variable, the tables, groups,
circuits and scoped constraints that touch it. Built once by `search()`.

## Services

`BacktrackingSearch`, `ArcConsistency` and `AllDifferentPropagator` keep no instance state and are still built
once and injected. `search()` creates the repositories, the index and the queue locally and threads them through
the recursion, exactly as it threads `domains` today.

**Open, and it is Maxime's, because it touches a house rule.** Whether the domain repository is built once and
injected, as `SolutionCache` is, or created per search. `SolutionCache` is built once because it caches *across*
calls. Domains are working storage for *one* search, and `Solver` is called once per action in
`rbs/service/simulation.py` and at every node MCTS opens, so a single injected instance would have to be reset on
entry and would not survive nesting. I recommend per search.

## What this replaces, and what it keeps

**Keeps** — `SupportTable`, `AllDifferentGroup`, `ScopedConstraint`, `Wipeout`, `SolveStatistics`,
`SolutionCache`, `ConstraintChecker`, and every filtering algorithm. AC-3 and Régin do the same work on the same
values; only what they are handed changes.

**Keeps, deliberately** — Régin's filtering exactly as it is. The marker wall states one `circuit` constraint and
forms no all-different group, so it never invokes `AllDifferentPropagator`, and sudoku, tic-tac-toe and chess
cannot regress through a change that isn't made.

**Replaces** — `Domains = dict[str, frozenset[Value]]` and the copy at `backtracking_search.py:79`, and with them
lines 70, 74, 93, 107 and 116.

**Leaves alone** — how `Solver` classifies a constraint into its strongest form, the solution cache's keying and
eviction, and every public signature. The one exception is value ordering, below.

## Value ordering becomes the caller's

`_eliminations` (`backtracking_search.py:159-170`) runs whenever a limit is set, and `simulation.py:75` always
sets one, so it is on for every game solve. For a group it is O(group size) per candidate value. Giving it the
index above brings that to O(domain size x in-degree), which is better and still not cheap: 117 649 operations
per node where a column of markers has 343 patterns.

So `solve` and `solve_with_statistics` gain one optional argument choosing the ordering, defaulting to what they
do today. **This is the only public-surface change in the work**, and it is a choice rather than a threshold: no
size decides it.

**Measured, once it was built, and the case for it is weaker than stated above.** Ordering the values costs about
twice the search at forty-nine values a variable and about eight times at three hundred and forty-three:

```
    rows   window   out-degree Q    domain order   least constraining
       2        2             49          0.19 s               0.39 s
       3        1            343          0.13 s               1.00 s
       2        3             49         11.67 s              22.42 s
```

The claim this section was written on — that the heaviest cell could not run at all with the ordering left on —
is **wrong**: eight times fifty-nine seconds is still inside a ten-minute budget. What is true is that the
ordering buys nothing where the search barely backtracks, and there is no reason to pay two to eight times for
it. That is a good enough reason for the argument, and it is not the reason first given.

## How it is judged

- `backtracking_search_tests.py` unchanged. The ordering tests are the net; a rewrite that preserves behaviour
  passes them without being touched.
- The whole framework suite, and OpenMindChess's, which reaches the solver at every node it searches.
- `openmind-solve sudoku/top95` — ninety-five of Norvig's hard puzzles, the strongest existing evidence that
  propagation and backtracking are still correct.
- Per-node cost and total solving time, before and after, on the same problems, reported as numbers with a
  header, not as a claim about quality.

## Open

- **Whether the domain repository is per search or injected once.** Above. Maxime's, and the only place this
  bends a house rule.
- **Whether `values` holds sets or something sparser.** Above. Five gigabytes on one cell, which fits.
- **Whether the repository pushes to the variable queue, or the search does.** Above.
- **Whether undoing on `Wipeout` belongs in the search or in a context manager.** The search is where the mark
  is taken, so it is where the unwind belongs; a manager would make forgetting impossible and adds a frame per
  node. Not decided.
- **`AllDifferentPropagator` recurses** in `augment` and `_components`, so its stack depth is bounded by the
  group size. Nothing here fixes that, and nothing here makes it worse; it fired no `RecursionError` on the
  problems measured, but that is those problems' luck. It belongs in `open-questions.md` rather than in this
  work.
- ~~**The factor between a standalone measurement and the same code inside OMF.**~~ **Settled by building it, and
  the guess was wrong.** The factor was put between three and eight; measured, it is about one, and on the
  heaviest problem that runs OMF is the *faster* of the two — 59.02 s against 64.25 s. So the costs this document
  attributes to the framework around the search (`Wipeout` construction, the fixpoint loop, propagator dispatch,
  the per-assignment log guard, `Value` objects rather than integers) do not amount to anything measurable beside
  the removals themselves. Left here rather than struck, because the reasoning that produced the wrong number is
  worth not repeating.
