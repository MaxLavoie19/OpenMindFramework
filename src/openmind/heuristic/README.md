# heuristic

## Purpose

The heuristic tasks — what a position is worth, what a move is worth — and what a model is given to answer them.

A heuristic is a fast estimate, and any model can give it: a ruleset run by the RBS, a network, a lookup table, a set
of relaxed constraints. The ports here say what a model answers, never how. The services that fill them are stateless
and are given the model they run (see `model/README.md`).

**A win always carries the payoff value.** Whatever the heuristic, a finished position is worth what the game paid,
never what a model guesses.

## Content

| File | What it is |
|---|---|
| `model/node.py` | `Node(state, game=None, features={})`: a state with what has been worked out about it; `feature(name, extract)` extracts once and shares after, `of(state)` a node for another state of the same game |
| `model/position_valuer.py` | `PositionValuer[Model]`, the position value task: `values(model, node)` |
| `model/move_rater.py` | `MoveRater[Model]`, the move value task: `rate(model, node, actions, player)` |
| `service/rule_heuristic.py` | `RuleHeuristic(rule_caller, consequence_library=None)`: runs a heuristic ruleset's RBS — `value`, `values`, `rate`, `explain`, `describe` — each rule's reading times its weight in the ruleset, summed |

## Nodes

A heuristic is given a node, not a bare state: the state, the game it is in, and the features extracted from it.

- A model that uses features either extracts them or fetches them from where they are stored. `Node.feature` is a
  memoized call: the first model that asks pays for it, every model after shares it.
- A model that reads no feature, such as a network, pays nothing for the node.
- A search builds a node per state it explores; anything else valuing a position, training included, builds one too.

`Node.game` is what a feature is extracted through. It is the temporary `RuleBasedGame` facade for now, and becomes
the simulation service and its RBS once the search and the inference readings are reworked.

## How a rule-based heuristic reads

`RuleHeuristic` runs the rules of one heuristic ruleset's RBS:

- Each rule's reading is multiplied by its weight in the ruleset listing it, and the products are summed.
- A rule reading nothing here adds nothing; one that raises, or gives something other than a finite number, is left
  out, so a heuristic survives a rule that doesn't apply.
- What a rule reads besides the state's models — `me`, `other`, `win_chance`, `wins`, `near`, `here` — is the
  position's consequences, extracted through the node (see `rbs/README.md` for what they are).
- `explain` gives each rule with what it added, and `describe` writes the ruleset as the heuristics it judges with, so
  two RBSs describing alike judge alike.

## Usage

```python
from openmind.knowledge.constant.task_constant import POSITION_VALUE
from openmind.rbs.factory.rbs_factory import create_rule_based_game, create_rule_based_system, create_rule_heuristic

game = create_rule_based_game(knowledge_base, "tictactoe")
position = create_rule_based_system(knowledge_base, "tictactoe", POSITION_VALUE)
reader = create_rule_heuristic()

node = game.node(game.start())
reader.value(position, node, "X")
reader.values(position, node, game.players().names)
```

## Logs

Nothing here logs of its own: a reading opens an `evaluation` frame for the debugger when a session tracks frames (see
`debug/README.md`).

## Notes

- Tests: `model/node_tests.py`, `service/rule_heuristic_tests.py`.
