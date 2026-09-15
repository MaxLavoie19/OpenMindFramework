# rule

## Purpose

Rules are Python. A constraint, an effect, an RBS condition: each is a `PythonRule`, the source of a Python expression
or a whole script. Any Python is allowed — imports, libraries, classes, `eval` — because OpenMind is a framework
people use to write the rules of their own problems, and they must be able to plug in whatever their problem needs.
This domain compiles rules once and runs them against states.

A saved rule is code: loading a rule base runs what it contains, so it needs the same trust as a source file.

## Content

| File | What it is |
|---|---|
| `model/python_rule.py` | `PythonRule(source)`: an expression, or a script |
| `model/compiled_rule.py` | `CompiledRule(rule, kind, code, arguments, definitions)`: a rule compiled once, with the definitions it sees |
| `constant/rule_constant.py` | The three kinds (`value`, `effects`, `definitions`), the compiled function's name and `all_different` |
| `service/rule_compiler.py` | `RuleCompiler`: compiles a rule for its value, for its effects, or as definitions, and keeps it |
| `service/rule_runner.py` | `RuleRunner`: a value rule's value in a state, or the state an effects rule leaves |
| `mapper/state_namespace_mapper.py` | `StateNamespaceMapper`: a state as the names a rule reads, those names back to a state, and how a rule reads a variable (`cell[2, 3]`) |
| `mapper/call_operand_mapper.py` | `CallOperandMapper`: the arguments of a rule that is a single call, such as `all_different(a, b, cell[1, 1])` |

## What a rule sees

- **State variables.** A plain variable is its value: `turn`. Variables named with indices are gathered under their
  base in a dict: `cell(2,3)` is `cell[2, 3]`, `payoff(X)` is `payoff["X"]`. An index written as a whole number is an
  `int`.
- **Action parameters**, by name: `row`, `col`. A parameter can't share its name with a state variable.
- **Definitions.** A domain's definitions script runs once; every name it leaves — constants, functions, imported
  modules — is visible to the domain's rules.
- **`all_different(*values)`**, true when no two values are equal. The CSP turns a constraint that is a single
  `all_different` call over parameters and parameter-free values into an all-different group.

## Three kinds of rules

| Kind | Compiled with | Written as | Run by |
|---|---|---|---|
| value | `compile_value(rule, parameters, definitions)` | an expression, or a script that `return`s | `value(compiled, state, parameters, names)`: the result |
| effects | `compile_effects(rule, definitions)` | a script | `apply(compiled, state, parameters)`: the next state |
| definitions | `compile_definitions(rule)` | a script | once per compiled definitions, when a rule seeing them first runs |

- **Value rules** become a function whose arguments are the given parameters the rule reads, in the given order, so a
  caller passes only those; `CompiledRule.arguments` is that list, which the CSP uses as the constraint's scope. A
  parameter it reads that isn't given raises `KeyError`. Its globals are the definitions' names and the state's
  variables, built once per state and kept; a value rule must not change them. `names`, optional, adds names of the
  caller's own, such as the RBS's `win_chance`; a name a state variable already has raises `ValueError`. The kept
  names are per state and per `names` mapping, by identity, so a caller passes the same mapping for the same state.
  They are kept until the process's memory guard clears them (see `parallel/README.md`).
- **Effects rules** run as a module in a fresh copy of those names plus every parameter. What the script leaves in the
  state's variables is the next state: `turn = other(turn)` or `cell[row, col] = turn`. Other names it assigns are its
  own. An index added under a base the state has (`played[11, 'A'] = 'defect'`) is a new variable, after the state's
  own, in the order the script added it; an index a variable name can't read back the same (`'1'`, `'a,b'`, `True`,
  `1.5`) raises `ValueError`. A new base can't be added this way, and a misspelt plain variable is just a new local
  name.
- A syntax error names the rule and the line; a runtime error's traceback shows the rule's own source.

## Usage

```python
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State

compiler, runner = RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper()))
state = State((("cell(1,1)", "X"), ("cell(1,2)", None), ("turn", "O")))
definitions = PythonRule("def other(player):\n    return 'X' if player == 'O' else 'O'")

empty = compiler.compile_value(PythonRule("cell[row, col] is None"), ("row", "col"))
runner.value(empty, state, {"row": 1, "col": 2})   # True

place = compiler.compile_effects(PythonRule("cell[row, col] = turn\nturn = other(turn)"), definitions)
runner.apply(place, state, {"row": 1, "col": 2})
# State(variables=(('cell(1,1)', 'X'), ('cell(1,2)', 'O'), ('turn', 'X')))
```

## Notes

- The compiler and the runner don't log; the services using them log their decisions, quoting rules by their source.
- Compiled code and namespaces don't travel between processes: a pickled `RuleCompiler`, `RuleRunner` or
  `StateNamespaceMapper` arrives without its compiled rules, namespaces or layouts and builds them again (see
  `parallel/README.md`).
- Tests: `mapper/call_operand_mapper_tests.py`, `mapper/state_namespace_mapper_tests.py`,
  `service/rule_compiler_tests.py`, `service/rule_runner_tests.py`.
