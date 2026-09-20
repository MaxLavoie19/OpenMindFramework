# search

## Purpose

Planning: what to do from here, and what to do about what comes back.

**Planning is a task, and a search is one model of it.** OMF enforces none. Minimax suits tic-tac-toe, Monte-Carlo
tree search suits chess, semi-determinized Monte-Carlo tree search suits a game with hidden information such as poker
or Stratego, and some work needs no search at all: a conversation is improvised, not planned. The time management
policy picks among them as it picks any model.

**Planning gives a strategy, not a plan.** A plan is a series of actions and states, and it solves nothing a strategy
doesn't solve better. A strategy is a move distribution per state — in this state play these actions at these
probabilities, in that one play those — so it covers the responses and the paths that invalidate it. The plan can be
read off it, the line where everyone plays as expected, and a state the strategy says nothing about is where the agent
strategizes again.

**What a strategy covers** is what is likely to be met: three moves deep in the main line, as a plan would, and the
unlikely result of the action being taken, which a plan can't hold. Not an implausible state, unless strategizing is so
far ahead that those are what is left to improve. A fast model can play a pre-move from the strategy while the planner
keeps strategizing.

## Content

| File | What it is |
|---|---|
| `model/planner.py` | `Planner[Model]`, the planning task: `plan(model, knowledge_base, node, guidance, settings)` |
| `model/strategy.py` | `Strategy(moves)`: a move distribution per state; `at(state)`, `chosen(state)`, `states()`, `Strategy.of(state, action)` |
| `model/guidance.py` | `Guidance(player, position_value, move_value, agents, hypotheses, policies)`: the models the caller chose for each task the planner needs |
| `model/search_settings.py` | `SearchSettings(nodes, depth, seed)`: how much a planner may explore |
| `model/hypothesis.py` | `Hypothesis(likelihood, state=None, policy=None)`: one guess at what can't be seen |
| `model/hypotheses.py` | `Hypotheses[Model]`, the hypothesis task: `about(model, node, agent)` |
| `service/improvised.py` | `Improvised(policy_picker, optimizer)`: the planner that doesn't plan |
| `service/minimax.py` | `Minimax(utility)`: reads a small game out to its end, each agent taking its own best |
| `factory/search_factory.py` | `create_improvised(...)`, `create_minimax()` |

## Agent models

An agent model is not another AI. It is the position value and move value heuristics trained to predict one agent's
play, through the ports that already exist, so modelling an agent is not a task of its own: those models live in that
agent's context.

**The caller passes them**, in `Guidance.agents`: playing a named opponent calls for that opponent's model, playing a
stranger for a generic one, such as a network trained to predict human moves. Theory of mind is what emerges from
this, the knowledge base and the beliefs held with an agent as holder — there is no theory-of-mind component.

## Hidden information

A model of the hypothesis task answers a probability distribution over what an agent can't see:

- **Where the possibilities can be counted**, the distribution is over them: this player holds a royal flush.
- **Where they can't**, it collapses into a distribution over policies. Under the fog of war an opponent can't be
  enumerated, but they are going economy, nuclear or swarm, and the odds shift as the game shows which.

How a model arrives at one is its own business: a hidden Markov model and a ruleset express it differently and answer
the same question. A game that hides nothing registers no model, and a planner runs on the state it was given.

## The planners built so far

| Model | What it does |
|---|---|
| `Improvised` | the policy worth following is picked, its optimizer gives the next best step, and that is the strategy |
| `Minimax` | the game read out to its end, each agent taking its own best, so a game of two is minimax and a game of more is max-n |

A finished position is worth what the game paid: a win always carries the payoff value. An unfinished one at the depth
given is worth what the position value heuristic says, and nothing where the caller gave none.

Monte-Carlo tree search and its semi-determinized form come later, as models of the same task.

## Usage

```python
from openmind.search.factory.search_factory import create_minimax
from openmind.search.model.guidance import Guidance
from openmind.search.model.search_settings import SearchSettings

strategy = create_minimax().plan(None, knowledge_base, game.node(state), Guidance("X"), SearchSettings())
strategy.chosen(state)      # what to play here
strategy.at(state)          # the distribution it plays by
```

## Logs

- `openmind.search.service.minimax`: `INFO Read <context> out to its end: <n> states worked out`.
- `openmind.search.service.improvised`: `DEBUG Improvised <action> for <player> by <policy>`, or `DEBUG Nothing to
  improvise for <player> here`.

## Notes

- Tests: `model/strategy_tests.py`, `service/planner_tests.py`.
