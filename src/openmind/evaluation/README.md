# evaluation

## Purpose

Measures how well an agent plays a domain, so training can be judged: results against baseline opponents, and how
often the agent's choices agree with perfect play at several iteration budgets, and how long each choice takes. The
training loop, not built yet, will run it after every round to draw learning curves.

This measures the agent. It is unrelated to the RBS's planned position evaluation, which will turn a board position
into a value.

## Content

| File | What it is |
|---|---|
| `model/evaluation_settings.py` | `EvaluationSettings(games, iterations, positions, budgets, seed)`; `positions` 0 skips agreement, `None` takes every position |
| `model/match_results.py` | `MatchResults(opponent, games, wins, draws, losses)`: a series against one opponent, from the evaluated agent's side |
| `model/agreement.py` | `Agreement(iterations, positions, optimal, optimal_visit_share, mean_regret, seconds_per_choice)`: how an agent searching with a number of iterations did on the sampled positions |
| `model/rater_agreement.py` | `RaterAgreement(positions, distinguishing, optimal, mean_regret)`: how a rater alone did on the sampled positions |
| `model/evaluation_report.py` | `EvaluationReport(domain, created_at, rules_file, settings, baselines, every_action_optimal, agreement, unguided_agreement, rater)`; `rules_file` names the rule base guiding the agent, `None` when unguided, in which case `unguided_agreement` is empty and `rater` is `None` |
| `service/exact_search.py` | `ExactSearch`: every legal action's value and the optimal actions in a state, by searching every reachable state; the positions with a legal action |
| `service/match_runner.py` | `MatchRunner`: plays a series between two policies in a two-player domain, switching seats every game |
| `service/evaluator.py` | `Evaluator`: runs the baseline series and the agreement measures, guided and unguided and for the rater alone when given a rater, and returns a report |
| `builder/evaluator_builder.py` | `EvaluatorBuilder`: wires the services an evaluator measures with |
| `factory/evaluator_factory.py` | `create_evaluator()` |
| `constant/evaluation_constant.py` | Default games, iterations, positions, budgets and seed; the word for every position (`all`); baseline opponent names |
| `mapper/report_json_mapper.py` | `ReportJsonMapper`: a report as JSON text |
| `mapper/report_text_mapper.py` | `ReportTextMapper`: a report as readable text: baseline results, an agreement table with guided and unguided side by side when both were measured, and the rules alone |
| `repository/report_repository.py` | `ReportRepository`: saves a report as `<directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` |

## How it measures

1. **Results against baselines:** the agent, built with `iterations` and `seed`, plays `games` games against
   `RandomPolicy`, then `games` games against untrained MCTS with the same iterations, switching seats every game. A
   higher payoff than the opponent is a win, an equal one a draw, a lower one a loss.
2. **Agreement with perfect play:** `positions` states are sampled with the seed from every reachable position with a
   legal action; `None` takes them all, in the order `ExactSearch` finds them. `ExactSearch` gives every legal action's
   value for the acting player, and the actions with the highest value are optimal (several can tie). The report
   counts the positions where every action is optimal: they can't tell good play from bad. For each budget, an agent
   built with that many iterations searches each sampled state, and three things are recorded:
   - **optimal choices:** the positions where its most visited action is optimal;
   - **visit share on optimal actions:** the share of the root's visits that went to optimal actions, averaged over
     the positions: where the search spent its effort;
   - **mean regret:** the best value minus the chosen action's value, averaged over the positions: 0 for an optimal
     choice, up to 1 for throwing away a win.
3. **Guided against unguided:** when a rater guides the agent (`rater`, with `rules_file` naming it), an unguided agent
   with the same iterations and seed is measured on the same positions at every budget. The rater is also measured
   alone: the positions where its ratings separate the actions at all, and, picking uniformly among its top-rated
   actions, the expected number of optimal picks and the expected mean regret. As in the search, an action rated
   `None` gets the mean of the other ratings, and when every rating is `None` every action is top-rated.
4. **Fewer iterations, same play:** the agreement across budgets shows how many iterations the agent needs; a trained
   agent should reach the same agreement with fewer.
5. **Speed:** every agreement budget records the mean seconds per choice.

`ExactSearch` visits every reachable state. That suits small domains such as tic-tac-toe (4,520 positions with a legal
action), not 4 in a row or chess. With `positions` = 0, the evaluator skips agreement and never calls `ExactSearch`:
the report's agreement is empty. That is how `tictactoe/fourinarow` is evaluated. `MatchRunner` needs exactly two
players.

## Usage

```python
from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.factory.domain_factory import create_domain
from openmind.evaluation.factory.evaluator_factory import create_evaluator
from openmind.evaluation.model.evaluation_settings import EvaluationSettings

report = create_evaluator().evaluate(
    create_domain("tictactoe"),
    AgentBuilder().with_exploration(EXPLORATION),   # evaluate sets the builder's iterations and seed
    EvaluationSettings(games=100, iterations=200, positions=100, budgets=(10, 20, 50, 100, 200, 500), seed=1),
)
```

To evaluate an agent guided by rules, create `rater = create_rule_rater(rule_base)`, give the builder
`with_guidance(rater)`, and pass the rule base's path as `rules_file` and the same `rater` to `evaluate`; the report
then also holds the unguided agreement and the rater alone. `ReportTextMapper().to_text(report)` gives the summary
`openmind-evaluate` prints. From the terminal: `openmind-evaluate tictactoe [--rules PATH] [--positions all]` (see
`entrypoint/README.md`).

## Logs

- `openmind.evaluation.service.evaluator`:
  - `INFO Against random: <games> games, <wins> wins, <draws> draws, <losses> losses`
  - `INFO Every action is optimal in <n> of <positions> positions`
  - `INFO Agreement with perfect play at <iterations> iterations: <optimal> of <positions> positions, <share> of visits on optimal actions, mean regret <regret>, <seconds> seconds per choice`
  - `INFO Unguided agreement with perfect play at <iterations> iterations: ...`, the same measures for the unguided agent, with a rater
  - `INFO Rater alone: ratings separate actions in <n> of <positions> positions; a top-rated action is optimal in <expected> of <positions>; mean regret <regret>`, with a rater
  - `INFO Agreement with perfect play skipped: no positions`, when `positions` is 0
  - `DEBUG <Agreement or Unguided agreement> at <iterations> iterations: chose <action>, regret <regret>; optimal: <actions>; <share> of visits on optimal actions; state: <name = value, ...>`
  - `DEBUG Rater alone: top-rated <actions>; optimal: <actions>; ratings <action>=<rating>, ...; state: <name = value, ...>`
- `openmind.evaluation.service.match_runner`:
  - `DEBUG Game <n> against <opponent>: evaluated agent plays <player>, payoffs <player>=<payoff> ...`
- `openmind.evaluation.service.exact_search`:
  - `INFO <domain> has <n> positions with a legal action`

Every agent search also logs its summary at INFO (see `mcts/README.md`).

## Notes

- Tests: `builder/evaluator_builder_tests.py`, `factory/evaluator_factory_tests.py`,
  `mapper/report_json_mapper_tests.py`, `mapper/report_text_mapper_tests.py`, `repository/report_repository_tests.py`,
  `service/evaluator_tests.py`,
  `service/exact_search_tests.py`, `service/match_runner_tests.py`; integration:
  `test/integration/tictactoe_exact_search_tests.py`; end-to-end: `test/end_to_end/evaluate_tictactoe_tests.py`,
  `test/end_to_end/evaluate_fourinarow_tests.py`.
