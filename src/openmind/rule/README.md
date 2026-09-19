# rule

## Purpose

Rules as data: what a game's rule is, whoever reads or runs it. The knowledge base keeps rules, the RBS runs them, the
CSP and the predictor are handed them; each imports this package, which depends only on `world`.

## Content

| File | What it is |
|---|---|
| `model/python_rule.py` | `PythonRule(source)`: a rule written in Python, an expression or a whole script |
| `model/rule.py` | `Rule`, a game's rule: `PythonRule` source OMF compiles, or one of the project's own functions; and the prototypes a function follows, `ConstraintRule`, `ValuesRule`, `EffectsRule`, `EndingRule` and `RecordRule` |

## Usage

```python
from openmind.rule.model.python_rule import PythonRule

legal = PythonRule("cell[row, col] == EMPTY")
```

## Logs

Nothing here logs: rules are data.
