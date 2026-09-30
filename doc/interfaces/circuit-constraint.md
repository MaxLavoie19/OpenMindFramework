# Interfaces: circuit, the constraint that says one cycle

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

Depends on [search-trail.md](search-trail.md) for the propagator contract. Adds one constraint form beside
`all_different`; changes nothing about the forms already there.

## What made this necessary

`Solver` gives every constraint the strongest form its scope allows: a pre-check, a domain filter, an
all-different group, a support table, or forward checking. That list has one global constraint in it, and
`csp/README.md` says so plainly. What it lacks is any way to say **connectivity** — that a set of chosen edges
forms a single cycle rather than several.

Without it, a problem whose answer is a tour has to be stated positionally, one variable per step, and the
statement is quadratic in the wrong places. With it, the problem is stated as what it is.

The problem that asked for it: seven colors of marker stapled along a wall, and which stretch of wall a few
visible markers identify ([marker-wall.md](marker-wall.md)). Its answer is a Hamiltonian cycle through a de
Bruijn graph, which is `circuit` and one line of declaration.

## The finding: one convention is expressible and the other is not

`circuit` has to map a **value** to a **variable** — the successor of a node is another node, which is another
variable. Two conventions exist, and OMF's rule surface admits exactly one of them.

**Values that name variables cannot work.** A constraint is compiled to a function of the parameters it reads and
called with values only (`rule/service/rule_caller.py:98`). It never learns which parameter a value came from.
So `circuit(successor_a, successor_b, successor_c)` would be handed `("b", "c", "a")` with no way to know that
the first argument came from `a` — the runtime predicate is not writable. `Solver.allows`, which is how an
optimizer's computed action gets checked (`solver.py:65`), and the forward-check fallback
(`backtracking_search.py:123`) would both be unable to say whether the constraint holds. Supporting it needs a
new structured rule kind carrying its own scope, and a branch for it in `RuleCaller.prepare`, `call`, `check` and
`source`. **This is why the plan does not do it.**

**Positional values need nothing new.** `circuit(x_0, ..., x_n-1)` where each `x_i` takes a value in `0..n-1`,
and `x_i = j` means *the successor of the node at operand position i is the node at operand position j*. The
value-to-variable map is the constraint's own operand order, which `CallOperandMapper` already hands over. The
runtime predicate is a pure function of the values: walk from nought, and require `n` steps to return to nought.

This is also how `circuit` is defined in the constraint-programming literature, so the convention is not ours to
invent.

**What it costs.** A domain value is now an index into one constraint's argument list, so a node's identity stops
being self-describing and whoever declares the problem owns the numbering. And the same parameters in two
`circuit` calls written in different operand orders mean different things. Both are standard, and both are worth
writing down rather than discovering.

## The surface syntax

```python
PythonRule(f"circuit({', '.join(successor_names)})")
```

`CIRCUIT = "circuit"` joins `ALL_DIFFERENT` in `rule/constant/rule_constant.py`, and a `_circuit` predicate joins
`_all_different` in the namespace every rule sees (`rule/service/rule_runner.py:24` and `:113`):

```python
def _circuit(*successors: object) -> bool:
    """Whether the successors form one cycle through every position: the successor of position i is position
    successors[i], and walking from nought comes back to nought in exactly as many steps as there are
    positions."""
```

`n == 0` holds vacuously. `n == 1` holds only for a self-loop, which is that case's only cycle — so the self-loop
is forbidden when, and only when, there is more than one position.

## Detection in `Solver`

`CallOperandMapper` needs no change: `to_operands(rule, function)` already takes the function name and caches per
`(rule, function)` (`call_operand_mapper.py:14-19`). What `Solver._solve_action` gains is a second classification
branch beside `_group`, tried before the arity branches.

**The trap.** `CalledRule.arguments` is in parameter *declaration* order, not operand order
(`rule/service/rule_compiler.py:34`). `all_different` does not care, because a set of values has no order.
`circuit` cares entirely. **The node numbering comes from the operand tuple and never from `prepared.arguments`.**
Getting this wrong gives a solver that finds a cycle through a permuted graph and reports it as a solution, which
is the worst kind of wrong: silent.

What `Solver` refuses, each with a message naming what it found:

- an operand that is not one of the action's parameters. A node with a fixed successor is standard and useful and
  is **not** in this version.
- a parameter appearing twice among the operands.
- a values rule giving a value outside `0..n-1`.

## Models

**`csp.model.circuit_constraint.CircuitConstraint`** — frozen, slotted.

```python
variables: tuple[str, ...]   # operand order; position i is variables[i], and value i means that position
```

`SearchSpace` gains `circuits: tuple[CircuitConstraint, ...] = ()`, **appended last with a default**, so the
positional construction in `backtracking_search_tests.py:28` keeps working untouched. `search()` counts circuit
scopes into `degree` the way it counts groups (`backtracking_search.py:45-47`); every node of a de Bruijn graph
then has the same degree, so declaration order breaks the tie, which is what we want.

## Repositories

**`csp.repository.circuit_chains.CircuitChains`** — which partial chains the recorded edges have formed. A
`Trailed`, like the domain repository.

```python
def __init__(self, positions: int) -> None
@property
def height(self) -> int
def recorded(self, position: int) -> bool
def record(self, position: int, successor: int) -> None
def start_of(self, end: int) -> int
def end_of(self, start: int) -> int
def length_of(self, start: int) -> int
def undo_to(self, height: int) -> None
```

The invariant: every position belongs to exactly one chain, a lone position being a chain of length one that
starts and ends at itself. Recording `i -> j` merges the chain ending at `i` with the chain starting at `j`: the
merged chain runs from `s = start_of(i)` to `e = end_of(j)` and its length is `length_of(s) + length_of(j)`.

## Services

**`csp.service.circuit_propagator.CircuitPropagator`** — stateless, built once, injected by `SolverBuilder`.

```python
def propagate(
    self,
    repository: DomainRepository,
    constraint: CircuitConstraint,
    chains: CircuitChains,
    changed: Iterable[str],
) -> None
```

Two duties, because the third is cheaper elsewhere.

**Permutation, by forward checking.** A position settled on `j` has `j` removed from every other variable in
scope. This is why `circuit` is sound on its own and needs no `all_different` beside it. It wants a
value-to-positions index, built once per constraint, so the removal costs the in-degree rather than the whole
scope.

**Subtour elimination.** For each newly settled `i -> j`: merge as above, then, with the merged chain running
`s` to `e` and covering `L` positions, remove `s` from `e`'s domain while `L < n`, and fix `e` to `s` once
`L == n`. O(1) amortised, and it is the only thing standing between the search and a set of short cycles.

**No self-loop** is the third duty and belongs in `Solver` at classification time, as a domain filter, which is
the existing "strongest form" idiom — value `i` comes out of variable `i` once, rather than being re-checked at
every node. Only when `n > 1`.

### Two correctness details that will bite if unwritten

**The merge must be idempotent.** `_propagate` re-invokes a propagator inside its fixpoint loop, so an edge
already merged must be recognised — hence `recorded` — or a chain is merged twice and its length is wrong, and a
wrong length either closes a short cycle or refuses the real one.

**A domain narrowed to one value is an assignment.** `backtracking_search.py:70` treats a singleton domain as
settled and never assigns it, so a position that propagation narrowed to one successor never arrives as a
"try". The propagator has to merge on singletons, not on assignments. Régin already behaves this way; this must
match it, or the search reports a solution whose last few edges were never checked.

## Propagation strength, and why the strong level is left out

Three levels are available. The numbers decide between them, and they were measured, not reasoned.

```
    level                                 cost                      at N=2401 Q=7   at N=117649 Q=49
    permutation, forward checking         O(in-degree)/assignment         ~6 us             ~41 us
    permutation, Regin (existing)         O(N*Q)/call                    0.014 s            ~4.5 s
    subtour elimination                   O(1) amortised                     --                 --
    connectivity, Tarjan over live edges  O(N*Q) edge visits/check        ~30 ms              ~10 s
```

Per node rather than per call, Régin is 34 s at the small cell and about six days at the large one, and
connectivity is 73 s and thirteen days. **Régin is the default and stays the default** for every existing
caller; the marker wall simply never forms a group, so it never pays it.

**Connectivity pruning is left out, but not for the reason first written down here.** The proposed search —
smallest-domain-first with the declaration-order tie-break, forward-checked permutation, subtour elimination by
chain ends, iterative, chains trailed — was implemented twice, once by the design pass and once independently to
check it. **The two disagree, and the independent run is the one recorded**, because a claim of no backtracking
at all is exactly the kind that should not be taken on trust:

```
    rows   window   positions N   out-degree Q   dead ends   value trials   stack depth   seconds
       1        1             7              7           0              5             5      0.00
       1        2            49              7           0             40            40      0.00
       2        1            49             49           0             47            47      0.00
       1        3           343              7           0            292           292      0.00
       3        1           343            343           0            341           341      0.11
       1        4         2 401              7      11 520         21 899         2 054      0.19
       2        2         2 401             49           0          2 350         2 350      0.10
       2        3       117 649             49           0        115 246       115 246      8.83
       3        2       117 649            343           0        117 304       117 304     64.25
```

**Built, and the table held.** Run through OMF's own solver afterwards, every dead-end and assignment count above
came back identical, including the 11 520 dead ends at one row by four. Two implementations written from the same
description but not from each other agreeing to the assignment is the strongest evidence either is right.

The design pass reported zero dead ends for every cell, including 2 401 by 7. **That is wrong**: the same
variable ordering and the same value ordering take 11 520 dead ends there, and tightening the fixpoint to revisit
only what changed does not move the number. So the search does backtrack.

**What the numbers do support** is weaker and enough: backtracking is shallow and cheap. Eleven and a half
thousand dead ends cost 0.19 s, because each is one assignment refused immediately by the permutation filter or
by the chain's own start, not a deep subtree explored and abandoned. Connectivity pruning would cost about 30 ms
per check at this size and ten seconds at 117 649, to save something already costing under a fifth of a second.
It is left out because it is not worth its price, not because there is nothing to prune.

**Which cell is hard is not explained, and is not going to be guessed at.** One row by four is the only cell of
the nine that backtracks. Two facts about it, and they do not combine into a story:

- **Out-degree is implicated.** The two cells at 2 401 positions differ only in out-degree, and the one with 7
  backtracks while the one with 49 does not. Fail-first commits the scarcest variable before its options run out,
  and a thin graph gives it less slack to be right with.
- **But thinness alone is not it.** Two rows by three is far thinner in proportion — out-degree 49 over 117 649
  positions — and takes no dead ends at all. So window length may matter as much as out-degree, and one run per
  cell cannot separate the two.

What the table does say is that the count is not monotone in size, so the largest cell is *not* the one to watch.
One row by four is, and it is cheap enough to watch closely.

**Where this is still a guess:** nine instances of one family of graph, one run each, no proof. If a problem does
thrash badly, the recovery is connectivity pruning under a budget the caller supplies.

## What this replaces, and what it keeps

**Keeps** — every existing constraint form and every existing propagator, `CallOperandMapper` unchanged, and
`Solver`'s public signatures.

**Adds only** — one constant, one predicate in the rule namespace, one model, one repository, one service, one
defaulted field on `SearchSpace`, one classification branch, one line in `SolverBuilder`.

**Leaves alone** — `all_different` and Régin entirely. `circuit` subsumes the permutation, so the two are
redundant where both are stated; that redundancy is legitimate modelling and would let one be dropped as a
relaxation, but under the current code it would drag Régin's per-node cost back in. The marker wall states
`circuit` alone.

## How it is judged

- `circuit_propagator_tests.py`, pinning: three variables admit exactly the two cyclic permutations; the
  self-loop is gone when there is more than one position and present when there is one; a chain's own start is
  refused at its end until the chain covers everything, and required on the last step; a singleton domain merges
  its chain; re-invoking on an unchanged repository merges nothing twice; `undo_to` restores domains and chains
  exactly; a scope that cannot be closed raises `Wipeout`.
- `rule_runner_tests.py`: rules see `circuit(...)`; true for one cycle, false for two disjoint cycles and for a
  repeated successor.
- `solver_tests.py`: `circuit(a, b, c)` becomes a circuit constraint; each refusal above raises `ValueError`
  naming what it found; `allows()` accepts a cycle and refuses a subtour through the runtime predicate.
- End to end on the small marker-wall cells, checked against the closed form rather than against the solver's
  own output.

## Open

- **Whether a parameter-free operand is allowed** — a node whose successor is already fixed. Standard, useful,
  and not asked for here. Refused in this version.
- **Whether `circuit` should also carry the connectivity level** as a caller-supplied budget now, or wait until
  something thrashes. The evidence says wait; the argument for now is that adding it later changes the
  propagator's signature.
- **Whether the value-to-positions index belongs to the constraint or to the index in
  [search-trail.md](search-trail.md).** It is the same shape as what `_eliminations` needs, and building it twice
  would be silly.
- **What `circuit` means for a relaxation.** `GameRelaxer` drops a constraint to make an easier problem, and
  dropping `circuit` leaves no permutation at all, not merely a weaker tour. Nothing here needs it; it is worth
  knowing before something does.
