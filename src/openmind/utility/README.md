# utility

## Purpose

What a move is worth to an agent: the value of each of its outcomes times that outcome's probability, summed.

An outcome's value is its worth on every goal, weighed by the preferences held for them. A chess coach plays to barely
win while making teachable moments come up: winning and teaching are two goals, and the weights between them are what
makes the coach a coach. A game's payoff is the goal every game has, so a context with no goal of its own is valued by
its payoffs alone.

Preferences live in the knowledge base, so an agent can look back on its own choices and say what it preferred.

## Content

| File | What it is |
|---|---|
| `service/utility.py` | `Utility`: `of(knowledge_base, node, outcomes, player, payoff, holder=(), role="")`, what a move is worth; `value(...)`, what one outcome is worth |
| `model/binner.py` | `Binner[Model]`, the binning task: `bins(model, outcomes, player, payoff)` |
| `model/bin.py` | `Bin(low, high, likelihood)`: a range of values an outcome can fall in, with how likely it is |
| `service/even_binner.py` | `EvenBinner(bins=3)`: cuts the range of payoffs into bins of equal width |
| `factory/utility_factory.py` | `create_utility()`, `create_even_binner(bins=3)` |

Goals and preferences are records in the knowledge base (`knowledge/model/goal.py`, `preference.py`).

## Binning

A continuous payoff can't be weighed outcome by outcome, so it is binned and each bin's value is multiplied by its
likelihood. A trip to the casino pays 0 to 1000 $: 0–100 $ isn't worth the trip, 100–200 $ is break-even, 200–1000 $ is
worth it. Weighed by their likelihoods, the bins say whether to go.

Binning is a task with its own models: bins of equal width (`EvenBinner`), bins placed around a decision threshold,
bins a ruleset decides. Even binning is what binning falls back on before a model is fitted for it.

## How a value is read

- **A goal's worth** is read from the outcome: a model named after the goal, a Map by player (`taught["X"]`) or a
  plain number. A goal the state says nothing about, or one nobody holds a preference for, is left out.
- **A role's preference** is taken over the one held for any role, so a coach and a player value the same outcome
  differently.
- **Falling back on the payoff:** where no goal could be weighed, the outcome is worth what the game paid.
- An outcome nothing can be read from is left out, and a move with no readable outcome is worth nothing at all.

## Usage

```python
from openmind.utility.factory.utility_factory import create_utility

worth = create_utility().of(knowledge_base, node, outcomes, "X", "payoff", role="coach")
```

## Logs

Logger `openmind.utility.service.utility`: `DEBUG The move is worth <utility> to <player> over <n> outcomes`.
`openmind.utility.service.even_binner`: `DEBUG Binned <n> outcomes from <low> to <high> into <k> bins`.

## Notes

- Values are numbers here. In some domains a value is a judgement rather than a number — "sounds selfish" in rhetoric —
  and those arrive at the rhetoric step.
- Tests: `service/utility_tests.py`.
