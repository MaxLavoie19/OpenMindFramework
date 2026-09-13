# rbs

## Purpose

The rule-based system: an explainable model distilled from search. Its rules rate actions by expected payoff for the
player to act, and those ratings guide MCTS. The RBS is one kind of model behind the generic `ActionRater` interface in
`mcts`; a decision tree, a DNN or another model could implement the same interface later, chosen for its speed,
accuracy and explainability. Position evaluation, turning a board position into a value, is a planned second goal of
the RBS for deeper games.

## Content

| File | What it is |
|---|---|
| `model/rule.py` | `Rule(action, conditions, expected_value, visits)`: for an action with that name, when every condition holds (none means any state), its expected payoff for the player to act over that many search visits |
| `model/rule_base.py` | `RuleBase(domain, rules)` |
| `model/induction_settings.py` | `InductionSettings(min_visits, max_conditions, min_rule_visits, min_gain)` |
| `constant/induction_constant.py` | Default induction settings: 5 visits per sample, 2 conditions, 50 visits per rule, 0.05 gain |
| `service/rule_inducer.py` | `RuleInducer`: induces a rule base from search samples |
| `service/rule_rater.py` | `RuleRater`: an `ActionRater`; rates each action with its most specific matching rule and explains the rating |
| `mapper/rule_text_mapper.py` | `RuleTextMapper`: a rule as readable text |
| `mapper/rule_base_json_mapper.py` | `RuleBaseJsonMapper`: a rule base as JSON text and back |
| `repository/rule_base_repository.py` | `RuleBaseRepository`: saves a rule base as `<directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` and loads it |
| `builder/rule_inducer_builder.py` | `RuleInducerBuilder`: wires the inducer's interpreter and mappers |
| `builder/rule_rater_builder.py` | `RuleRaterBuilder`: sets the rule base and wires the rater's interpreter; rejects a missing rule base |
| `factory/rbs_factory.py` | `create_rule_inducer()` and `create_rule_rater(rule_base)` |

## How rules are induced

1. Samples with fewer than `min_visits` visits are dropped. The others are grouped by action name and weighted by
   their visits.
2. Candidate conditions come from what the samples contain:
   - `variable == value`, for every state variable and every value seen, such as `cell(2,2) == None`;
   - `indexed variable at the action's parameters == value`, such as `cell(row,col) == None`, for every order of the
     parameters;
   - `parameter == value`, such as `row == 2`.

   A candidate that can't be evaluated for every sample of the action is dropped.
3. Each action name gets a rule without conditions: its visit-weighted mean payoff.
4. A rule gains one more condition when the narrower rule keeps at least `min_rule_visits` visits and its expected
   value differs from its parent's by at least `min_gain`, up to `max_conditions` conditions. Conditions are added in
   candidate order, so each combination appears once; a condition that doesn't narrow the rule is skipped.

## How rules rate

For each action, the matching rule with the most conditions, then the most visits, gives the rating. An action name
without rules gets `None`. A condition on a variable the state doesn't have doesn't hold. `explain(state, action)`
returns the rule behind the rating.

## Usage

```python
from openmind.rbs.factory.rbs_factory import create_rule_inducer, create_rule_rater
from openmind.rbs.model.induction_settings import InductionSettings

rule_base = create_rule_inducer().induce("tictactoe", samples, InductionSettings(5, 2, 50, 0.05))  # SearchResult.samples
rater = create_rule_rater(rule_base)
rater.rate(state, actions)          # one rating per action
rater.explain(state, actions[0])    # the rule behind the first rating
```

Training does this from self-play: see `training/README.md` and `openmind-distill`.

## Logs

Logger `openmind.rbs.service.rule_inducer`:

- `INFO Induced <n> rules for <action> from <m> samples`
- `DEBUG <rule as text>`, one line per rule, in the form `place when cell(row,col) == None: EV <value> over <visits> visits`

The rater doesn't log: it runs inside rollouts.

## Notes

- Tests: `builder/rule_inducer_builder_tests.py`, `builder/rule_rater_builder_tests.py`, `factory/rbs_factory_tests.py`,
  `mapper/rule_base_json_mapper_tests.py`, `mapper/rule_text_mapper_tests.py`,
  `repository/rule_base_repository_tests.py`, `service/rule_inducer_tests.py`, `service/rule_rater_tests.py`;
  integration: `test/integration/tictactoe_distillation_tests.py`.
