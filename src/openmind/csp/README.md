# csp

## Purpose

Constraint satisfaction: the RBS's solver for legal moves. It solves one action at a time, over rules: a values rule
for each parameter, giving the values it can take, and the constraint rules its values must satisfy in a state. Both
are Python rules (see `rbs/README.md`), and they can see a definitions script. The solver finds the solutions, each
one an action with its parameter values: every legal move of a game such as tic-tac-toe, or the whole grid of a puzzle
such as sudoku, whose single action fills every empty cell at once. It returns every solution, or at most a limit,
using propagation and backtracking.

The solver knows nothing of a game. It's given the rules and gives back actions; which actions a game has, and which of
its rules belong to each, is the RBS's to say (`rbs/service/rule_based_game.py`). A relaxation is just a game whose
RBS hands the solver fewer constraint rules.

## Content

| File | What it is |
|---|---|
| `model/support_table.py` | `SupportTable(first, second, allowed)`: the value pairs a two-parameter constraint allows |
| `model/all_different_group.py` | `AllDifferentGroup(variables)`: parameters that must all take different values |
| `model/circuit_constraint.py` | `CircuitConstraint(variables)`: parameters whose values make one cycle through every one of them, each value naming a parameter by where it stands |
| `model/scoped_constraint.py` | `ScopedConstraint(rule, scope)`: a prepared constraint (`CalledRule`) on three or more parameters |
| `model/search_space.py` | `SearchSpace(action, variables, tables, groups, constraints, circuits)`: what backtracking explores for one action, each parameter with the values it has left |
| `model/constraint_index.py` | `ConstraintIndex`: which constraints each variable is in, and which parameters of a circuit could still take a value; built once per search |
| `repository/domain_repository.py` | `DomainRepository`: the values every variable has left, and the trail of what propagation took away |
| `repository/variable_queue.py` | `VariableQueue`: the variables still to decide, fewest values first, stale entries dropped as they surface |
| `repository/circuit_chains.py` | `CircuitChains`: the runs of positions a circuit's decided edges have joined, with its own trail |
| `model/solve_statistics.py` | `SolveStatistics(solutions, assignments, dead_ends, pruned_values)`: what a search did |
| `model/wipeout.py` | `Wipeout`: raised when propagation leaves a variable without any value |
| `builder/solver_builder.py` | `SolverBuilder`: wires a solver with its rule caller, call operand mapper, constraint checker, propagators and search |
| `factory/csp_factory.py` | `create_solver()` |
| `constant/solver_constant.py` | `openmind-solve`'s default solution limit (2) |
| `service/constraint_checker.py` | `ConstraintChecker(rule_caller)`: checks a prepared constraint for some parameter values; a constraint must give true or false |
| `service/arc_consistency.py` | `ArcConsistency`: AC-3 over support tables |
| `service/all_different_propagator.py` | `AllDifferentPropagator`: Régin's matching-based filtering for all-different |
| `service/circuit_propagator.py` | `CircuitPropagator`: the successors all different, and joined into one cycle rather than several loops |
| `service/backtracking_search.py` | `BacktrackingSearch`: backtracking with maintained propagation and ordering heuristics |
| `repository/solution_cache.py` | `SolutionCache`: the solutions found so far, built once and given to the solver, which keeps nothing itself; the memory guard evicts the oldest |
| `service/solver.py` | `Solver(..., solution_cache)`: `allows(state, action, values, constraints=(), definitions=None, player=None)` checks one action, which is how an optimizer's computed action is checked; `solve(state, action, values, constraints=(), definitions=None, limit=None, player=None)` solves one action over its rules: gives each constraint its strongest form, searches, then orders and caches the solutions; `solve_with_statistics` also gives what the search did |

## Usage

```python
from openmind.csp.factory.csp_factory import create_solver
from openmind.rbs.model.python_rule import PythonRule
from openmind.world.model.state import State

positions = PythonRule("(1, 2)")
state = State((("cell(1,1)", "X"), ("cell(1,2)", None), ("cell(2,1)", None), ("cell(2,2)", None)))

create_solver().solve(
    state,
    "place",
    {"row": positions, "col": positions},
    (PythonRule("cell[row, col] is EMPTY"),),
    PythonRule("EMPTY = None"),
)
# (Action(name='place', parameters=(('col', 2), ('row', 1))),
#  Action(name='place', parameters=(('col', 1), ('row', 2))),
#  Action(name='place', parameters=(('col', 2), ('row', 2))))
```

A game's legal moves come from its RBS, which hands each of its actions' rules to the solver in turn:
`rbs.actions(state)`, and `rbs.actions_with_statistics(state)` with the statistics summed over the game's actions.

## How the solver works

For the action given:

1. Each constraint is compiled once, with the definitions; its scope is the set of the action's parameters
   it reads. A name that is neither a parameter, a state variable nor defined raises `NameError` when the constraint
   runs.
2. The constraints that read no parameter are checked first: if one is false, the action has no solution.
3. Each parameter gets the values its rule gives in the state, each value once, in the order given. A rule that
   reads nothing gives the same values in every position; one that reads the state lets a game generate its legal
   moves once per position instead of checking every combination of parameter values.
4. Each other constraint takes the strongest form its scope allows:
   - **a single `circuit(...)` call over the action's parameters:** the parameters form a circuit constraint, in
     the order the call names them, and no position may follow itself. A value names a position by where it stands
     among the call's arguments, which is the only way a constraint can name a parameter — a rule is handed values
     and never learns which parameter each came from. An operand that is not a parameter, a parameter named twice,
     or a value outside the positions raises `ValueError`;
   - **a single `all_different(...)` call whose arguments are parameters or parameter-free expressions:** the
     parameter-free values are removed from the parameters' domains, and the parameters form an all-different group
     (equal fixed values or a repeated parameter mean no solution);
   - **one parameter:** filters that parameter's domain;
   - **two parameters:** evaluated once per pair of values into a support table;
   - **three or more:** forward-checked during search.
5. Before searching, arc consistency on the support tables, all-different filtering on the groups, circuit
   filtering and forward checking run until nothing changes, revisiting only the constraints the changed variables
   are in. A variable left without values means no solution.
6. Backtracking assigns the open variable with the fewest values left; ties go to the variable in the most
   constraints, then to declaration order. After every assignment the same propagation runs again (maintaining arc
   consistency); a variable left without values is a dead end.

   **The domains are one repository with a trail, and the frames are a list.** Deciding a variable takes a mark and
   an assignment rather than a copy of every domain, and undoing one puts back the few values that were taken — so
   a node costs what it changed, not what exists. The frames are held in a list rather than on the interpreter's
   stack, because a search is as deep as it has variables.
7. Values are tried in domain order, or least-constraining first when a limit is set. `solve` takes a
   `value_order` to say which; left out, it is what it has always been. Least-constraining costs a pass over what
   each value would rule out, which earns its keep for a game's moves and does not where a variable has hundreds of
   values and the search never backtracks. The search stops once it has `limit` solutions.
8. Solutions are sorted by the parameters' value order, first parameter slowest, and cached per state, limit and
   rules, in the solution cache it is given, until the process's memory guard evicts them (see `parallel/README.md`).

`solve(state, action, values, constraints, definitions, limit=None, player=None)`: given a player, such as one of several players acting at once, every
rule also reads it as `player`: it is added to the state as a variable, so results are cached per player too, and a
state that already has a variable named `player` raises `ValueError`.

`solve_with_statistics` gives the same solutions with the search's statistics; an action whose parameter-free
constraint is false has no solution and adds nothing. A cached result keeps the statistics of the search that found
it.

A constraint must give `True` or `False`; any other value raises `TypeError`.

## Logs

- `openmind.csp.service.solver`:
  - `DEBUG <action>: no solution, constraint is false: <constraint source>`
  - `DEBUG <action>: <n> solutions, <a> assignments, <d> dead ends, <p> values pruned`
- `openmind.csp.service.backtracking_search`:
  - `DEBUG <action>: try <variable> = <value>`
  - `DEBUG <action>: dead end at <variable> = <value>, <variable> has no value left`
  - `DEBUG <action>: no solution, <variable> has no value left`, when propagation fails before searching

Cached results log nothing. MCTS rollouts call the solver, so everything is at DEBUG.

## Notes

- Tests: `builder/solver_builder_tests.py`, `factory/csp_factory_tests.py`,
  `service/all_different_propagator_tests.py`, `service/arc_consistency_tests.py`,
  `service/circuit_propagator_tests.py`, `repository/domain_repository_tests.py`,
  `repository/variable_queue_tests.py`, `repository/circuit_chains_tests.py`,
  `service/backtracking_search_tests.py`, `service/constraint_checker_tests.py`, `service/solver_tests.py`;
  integration: `test/integration/tictactoe_actions_tests.py`, `test/integration/sudoku_solve_tests.py`,
  `test/integration/sudoku_collections_solve_tests.py`,
  `test/integration/marker_wall/marker_wall_circuit_tests.py` — a de Bruijn sequence over a hundred thousand
  parameters, which is what the trail and `circuit` were built for and the only thing here that searches that
  deep. `test/integration/marker_wall/marker_wall_sweep.py` runs it across sizes and is not a test.
