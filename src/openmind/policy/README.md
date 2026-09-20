# policy

## Purpose

How an agent chooses to play before it chooses what to play.

The set of all possible actions is too large to weigh one by one, and much of it is nonsense. A **policy** is a way of
playing tuned to a subset of them — flee, charge, kite — so the agent picks how it wants to play, then picks the
action that serves that best. A policy performs two tasks: a position value heuristic valuing the states it is after,
and a move value heuristic valuing the moves that serve it. One model can perform both, such as a network with a head
for each.

A policy is manufactured — an agent told to learn freeze, fight and flight — or learned by policy optimization.

**Optimizing is the alternative to expanding.** Expanding lists the valid actions and rates them, which the CSP and
the heuristics do. Optimizing computes the one action that serves the sub-goal, as an artillery piece calculates its
firing solution rather than enumerating every ballistic one.

## Content

| File | What it is |
|---|---|
| `model/policy_valuer.py` | `PolicyValuer[Model]`, the policy value task: `rate(model, node, policies, player)` |
| `model/policy_picker.py` | `PolicyPicker[Model]`, the policy picking task: `pick(model, node, policies, player)`, which policies are worth expanding |
| `model/optimizer.py` | `Optimizer[Model]`, the optimizing task: `optimize(model, policy, node, player)`, the next best step |
| `service/rule_policy_valuer.py` | `RulePolicyValuer(rule_caller)`: rates a policy by a ruleset, its rules reading the policy's name and sub-goal |
| `service/rated_policy_picker.py` | `RatedPolicyPicker(policy_valuer)`: keeps what a policy value model rates above picking at random |
| `service/random_picker.py` | `RandomPicker(source=None)`: an optimizer taking a legal action at random |
| `service/solver_optimizer.py` | `SolverOptimizer`: an optimizer letting the CSP work the values out, so its action is legal by construction |
| `service/rule_optimizer.py` | `RuleOptimizer(rule_caller)`: an optimizer running a ruleset that computes the action |
| `factory/policy_factory.py` | `create_random_picker(source=None)`, `create_solver_optimizer()`, `create_rule_optimizer()`, `create_rule_policy_valuer()`, `create_rated_policy_picker(policy_valuer=None)` |

A policy itself is a record in the knowledge base (`knowledge/model/policy.py`), like the goals and preferences
utility weighs (see `knowledge/README.md`).

## Picking a policy

`RatedPolicyPicker` reads what a policy value model says and keeps what is worth expanding:

- A policy rated below the default policy — picking at random — isn't worth expanding: the cheapest way of playing
  already does better.
- When the default policy is the best rated, it is picked alone: there is no time to think, so nothing else is worth
  considering.
- A policy nothing is known about is kept, last: one nobody has measured is worth trying before it is dismissed.

## Optimizing

| Model | What it does |
|---|---|
| `RandomPicker` | one of the legal actions, at random: what a policy does when there is no time to think |
| `SolverOptimizer` | the CSP assigning the parameters one at a time, so its action is legal by construction |
| `RuleOptimizer` | a ruleset computing the action — equations, a controller — checked against the constraints afterwards |

An **optimum rule** reads the state, the player it optimizes for and the policy's sub-goal, and gives an `Action` or
its name with its parameters, such as `("fire", {"angle": 41.3})`. It knows the goal, not the rules, so what it gives
goes through `Simulation.allows` before it is taken.

**Control systems fit here.** A system with several inputs and several outputs is a game whose actions carry several
parameters. Model predictive control's plan is the series of steps the optimizers give as the search explores, so an
optimizer answers for one step; model reference adaptive control keeps what it adapted as its model's own data.

## Usage

```python
from openmind.policy.factory.policy_factory import create_rated_policy_picker, create_solver_optimizer

picked = create_rated_policy_picker().pick(rating_model, node, knowledge_base.policies(context_id), "X")
action = create_solver_optimizer().optimize(None, picked[0], node, "X")
```

## Logs

- `openmind.policy.service.rated_policy_picker`: `DEBUG Picked <n> of <m> policies: <names>`, or `DEBUG Picking at
  random is the best rated policy here: nothing else is worth considering`.
- `openmind.policy.service.random_picker`, `solver_optimizer`, `rule_optimizer`: `DEBUG` what each settled on, and
  what the constraints refused.

## Notes

- Tests: `service/optimizer_tests.py`, `service/policy_picking_tests.py`.
