# rule

## Purpose

Rules as data, and running a rule on a state: what a game's rule is, whoever reads or runs it. The knowledge base keeps
rules; the RBS, the CSP and the predictor run them through `RuleCaller`.

## Content

| File | What it is |
|---|---|
| `model/python_rule.py` | `PythonRule(source)`: a rule written in Python, an expression or a whole script |
| `model/rule.py` | `Rule`, a game's rule: `PythonRule` source OMF compiles, or one of the project's own functions; and the prototypes a function follows, `ConstraintRule`, `ValuesRule`, `EffectsRule`, `EndingRule` and `RecordRule` |
| `service/rule_caller.py` | `RuleCaller`: calls a rule of either kind on a state; `check` refuses a function no worker process could find |
| `service/rule_compiler.py`, `service/rule_runner.py` | compile a rule's source once, keeping it in the compiled rule cache it is given, and run it on a state's namespace |
| `repository/compiled_rule_cache.py` | `CompiledRuleCache`: the rules compiled so far, built once and given to the compiler; a copy sent to another process arrives empty |
| `mapper/state_namespace_mapper.py`, `mapper/call_operand_mapper.py` | a state as the names a rule reads, and a call's operands |
| `constant/rule_constant.py` | the names rules are compiled under, and the names of the two definitions scripts |
| `factory/rule_factory.py` | `create_rule_caller()` |

## Usage

```python
from openmind.rule.model.python_rule import PythonRule

legal = PythonRule("cell[row, col] == EMPTY")
```

## Logs

The rule caller opens a `rule` frame for the debugger when a session tracks frames (see `debug/README.md`).
