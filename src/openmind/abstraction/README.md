# abstraction

## Purpose

What one level of the hierarchy sees of a state.

OMF works on a hierarchy of contexts, and a level holds the detail it works on and none of the rest: macromanagement
sees the economy and the army, micromanagement the units in this fight, a coding agent's high level a file's path and
not its content. What a level doesn't hold, it fetches by acting — reading the file is a move of its own game — so an
abstraction never has to guess what will be needed.

Abstracting is a task like any other, so any model may fill it: a ruleset, a decoder, a summary, a relaxation that
drops detail. Which one a level runs with is the time management policy's choice, as for every other task.

## Content

| File | What it is |
|---|---|
| `model/abstractor.py` | `Abstractor[Model]`, the abstraction task: `abstract(model, node, context)` |
| `service/rule_abstractor.py` | `RuleAbstractor(rule_caller)`: the RBS filling it, running an abstraction ruleset |
| `factory/abstraction_factory.py` | `create_rule_abstractor()` |

## The rule-based model

An abstraction rule gives the models the level holds, by name, and what the level sees is what its rules gave
together, a later rule replacing a model an earlier one gave:

```python
ruleset = knowledge_base.ruleset(Ruleset("abstraction", context_id, ABSTRACTION, source))
rule = knowledge_base.declare(RuleRecord("it sees what it has", ABSTRACTION, PythonRule("{'resources': resources}"), source))
knowledge_base.link(ruleset.id, rule.id)
```

Rules and rulesets go into the knowledge base the way a heuristic's do; the game declarer declares a game's own rules
and nothing declares an abstraction ruleset for a level yet.

A rule reading nothing, raising, or giving anything but names and models gives nothing, so a level survives a rule
that doesn't apply here. Where every rule gave nothing, the abstraction is None and the level sees the state as it is.

## Usage

```python
from openmind.abstraction.factory.abstraction_factory import create_rule_abstractor

create_rule_abstractor().abstract(rule_based_system, node, "macromanagement")
```

A level holds the model and the service together and calls this itself (`agent/service/level.py`).

## Logs

Logger `openmind.abstraction.service.rule_abstractor`: `DEBUG <level> sees <models>`.

## Notes

- Tests: `service/rule_abstractor_tests.py`.
- The rule kind is `abstraction` (`knowledge/constant/rule_kind_constant.py`), and the task is `abstraction`
  (`knowledge/constant/task_constant.py`).
