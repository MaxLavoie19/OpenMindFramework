# csp

## Purpose

Constraint satisfaction. A problem describes, for each action a domain allows, the parameters to choose and the
constraints their values must satisfy in a state. The solver finds the solutions, each one an action with its
parameter values: every legal move of a game such as tic-tac-toe, or the whole grid of a puzzle such as sudoku, whose
single action fills every empty cell at once. It returns every solution, or at most a limit, using propagation and
backtracking.

## Content

| File | What it is |
|---|---|
| `model/discrete_domain.py` | `DiscreteDomain(values)`: the finite values a variable can take, in order |
| `model/variable.py` | `Variable(name, domain)`: an action parameter to solve for |
| `model/action_definition.py` | `ActionDefinition(name, variables, constraints)`: an action and the constraints that make it legal |
| `model/problem.py` | `Problem(actions)`: every action definition of a domain |
| `model/support_table.py` | `SupportTable(first, second, allowed)`: the value pairs a two-parameter constraint allows |
| `model/all_different_group.py` | `AllDifferentGroup(variables)`: parameters that must all take different values |
| `model/scoped_constraint.py` | `ScopedConstraint(expression, scope)`: a constraint on three or more parameters |
| `model/search_space.py` | `SearchSpace(action, variables, tables, groups, constraints)`: what backtracking explores for one action |
| `model/solve_statistics.py` | `SolveStatistics(solutions, assignments, dead_ends, pruned_values)`: what a search did |
| `model/wipeout.py` | `Wipeout`: raised when propagation leaves a variable without any value |
| `builder/problem_builder.py` | `ProblemBuilder`: collects action definitions; rejects a repeated action or variable name |
| `builder/solver_builder.py` | `SolverBuilder`: wires a solver with its constraint checker, scope mapper, propagators, search and text mapper |
| `factory/csp_factory.py` | `create_solver()` |
| `constant/solver_constant.py` | The cache size (100,000 results) and `openmind-solve`'s default solution limit (2) |
| `service/constraint_checker.py` | `ConstraintChecker`: evaluates a constraint or an operand for some parameter values; a constraint must give true or false |
| `service/arc_consistency.py` | `ArcConsistency`: AC-3 over support tables |
| `service/all_different_propagator.py` | `AllDifferentPropagator`: Régin's matching-based filtering for all-different |
| `service/backtracking_search.py` | `BacktrackingSearch`: backtracking with maintained propagation and ordering heuristics |
| `service/solver.py` | `Solver`: gives each constraint its strongest form, searches, then orders and caches the solutions; `solve_with_statistics` also gives what the search did |

## Usage

```python
from openmind.csp.builder.problem_builder import ProblemBuilder
from openmind.csp.factory.csp_factory import create_solver
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.variable import Variable
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.world.model.state import State

positions = DiscreteDomain((1, 2))
problem = (
    ProblemBuilder()
    .with_action(
        "place",
        (Variable("row", positions), Variable("col", positions)),
        (Equals(StateVariable("cell", (ActionParameter("row"), ActionParameter("col"))), Constant(None)),),
    )
    .build()
)
state = State((("cell(1,1)", "X"), ("cell(1,2)", None), ("cell(2,1)", None), ("cell(2,2)", None)))

create_solver().solve(problem, state)
# (Action(name='place', parameters=(('col', 2), ('row', 1))),
#  Action(name='place', parameters=(('col', 1), ('row', 2))),
#  Action(name='place', parameters=(('col', 2), ('row', 2))))
create_solver().solve(problem, state, limit=1)   # at most one solution
actions, statistics = create_solver().solve_with_statistics(problem, state)
# the same solutions, with SolveStatistics(solutions, assignments, dead_ends, pruned_values)
```

## How the solver works

For each action definition:

1. Each constraint's scope is the set of parameters it reads. A constraint reading a parameter the action doesn't
   have raises `KeyError`.
2. Each constraint takes the strongest form its scope allows:
   - **no parameter:** checked once; if it's false, the action has no solution;
   - **`all_different` over parameters and parameter-free operands:** the parameter-free operands' values are removed
     from the parameters' domains, and the parameters form an all-different group (equal fixed values or a repeated
     parameter mean no solution);
   - **one parameter:** filters that parameter's domain;
   - **two parameters:** evaluated once per pair of values into a support table;
   - **three or more:** forward-checked during search.
3. Before searching, arc consistency on the support tables, all-different filtering on the groups and forward checking
   run until nothing changes. A variable left without values means no solution.
4. Backtracking assigns the open variable with the fewest values left; ties go to the variable in the most
   constraints, then to declaration order. After every assignment the same propagation runs again (maintaining arc
   consistency); a variable left without values is a dead end.
5. Values are tried in domain order, or least-constraining first when a limit is set. The search stops once it has
   `limit` solutions.
6. Solutions are sorted by the variables' domain order, first variable slowest, and cached per problem, state and
   limit: up to 100,000 results, the least recently used going first.

`solve_with_statistics` gives the same solutions with the search's statistics summed over the action definitions; an
action whose parameter-free constraint is false adds nothing. A cached result keeps the statistics of the search that
found it.

A constraint must evaluate to `True` or `False`; any other value raises `TypeError`. Rules stay structured: support
tables are computed from them, never compiled into code.

## Logs

- `openmind.csp.service.solver`:
  - `DEBUG <action>: no solution, constraint is false: <constraint>`
  - `DEBUG <action>: <n> solutions, <a> assignments, <d> dead ends, <p> values pruned`
- `openmind.csp.service.backtracking_search`:
  - `DEBUG <action>: try <variable> = <value>`
  - `DEBUG <action>: dead end at <variable> = <value>, <variable> has no value left`
  - `DEBUG <action>: no solution, <variable> has no value left`, when propagation fails before searching

Cached results log nothing. MCTS rollouts call the solver, so everything is at DEBUG.

## Notes

- Tests: `builder/problem_builder_tests.py`, `builder/solver_builder_tests.py`, `factory/csp_factory_tests.py`,
  `service/all_different_propagator_tests.py`, `service/arc_consistency_tests.py`,
  `service/backtracking_search_tests.py`, `service/constraint_checker_tests.py`, `service/solver_tests.py`;
  integration: `test/integration/tictactoe_actions_tests.py`, `test/integration/sudoku_solve_tests.py`,
  `test/integration/sudoku_collections_solve_tests.py`.
