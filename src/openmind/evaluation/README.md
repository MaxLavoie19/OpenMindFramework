# evaluation

## Purpose

Measures how well an agent plays a domain, so training can be judged: results against baseline opponents, and how
often the agent's choices agree with perfect play at several iteration budgets. Training (iteration 6) will run it
after every round to draw learning curves.

This measures the agent. It is unrelated to the RBS's planned position evaluation, which will turn a board position
into a value.

## Content

| File | What it is |
|---|---|
| `model/evaluation_settings.py` | `EvaluationSettings(games, iterations, positions, budgets, seed)` |
| `model/match_results.py` | `MatchResults(opponent, games, wins, draws, losses)`: a series against one opponent, from the evaluated agent's side |
| `model/agreement.py` | `Agreement(iterations, positions, optimal)`: how many sampled positions got an optimal choice at a number of iterations |
| `model/evaluation_report.py` | `EvaluationReport(domain, created_at, settings, baselines, agreement)` |
| `service/exact_search.py` | `ExactSearch`: the optimal actions in a state by searching every reachable state; the positions with a legal action |
| `service/match_runner.py` | `MatchRunner`: plays a series between two policies in a two-player domain, switching seats every game |
| `service/evaluator.py` | `Evaluator`: runs the baseline series and the agreement measure and returns a report |
| `builder/evaluator_builder.py` | `EvaluatorBuilder`: wires the services an evaluator measures with |
| `factory/evaluator_factory.py` | `create_evaluator()` |
| `constant/evaluation_constant.py` | Default games, iterations, positions, budgets and seed; baseline opponent names |
| `mapper/report_json_mapper.py` | `ReportJsonMapper`: a report as JSON text |
| `repository/report_repository.py` | `ReportRepository`: saves a report as `<directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` |

## How it measures

1. **Results against baselines:** the agent, built with `iterations` and `seed`, plays `games` games against
   `RandomPolicy`, then `games` games against untrained MCTS with the same iterations, switching seats every game. A
   higher payoff than the opponent is a win, an equal one a draw, a lower one a loss.
2. **Agreement with perfect play:** `positions` states are sampled with the seed from every reachable position with a
   legal action. For each budget, an agent built with that many iterations chooses in each sampled state; the choice
   counts as optimal when `ExactSearch` rates it best for the acting player (any of several tied best actions counts).
3. **Fewer iterations, same play:** the agreement across budgets shows how many iterations the agent needs; a trained
   agent should reach the same agreement with fewer.

`ExactSearch` visits every reachable state. That suits small domains such as tic-tac-toe (4,520 positions with a legal
action), not 4 in a row or chess. `MatchRunner` needs exactly two players.

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

From the terminal: `openmind-evaluate tictactoe` (see `entrypoint/README.md`).

## Logs

- `openmind.evaluation.service.evaluator`:
  - `INFO Against random: <games> games, <wins> wins, <draws> draws, <losses> losses`
  - `INFO Agreement with perfect play at <iterations> iterations: <optimal> of <positions> positions`
  - `DEBUG At <iterations> iterations, chose <action>; optimal: <actions>; state: <name = value, ...>`
- `openmind.evaluation.service.match_runner`:
  - `DEBUG Game <n> against <opponent>: evaluated agent plays <player>, payoffs <player>=<payoff> ...`
- `openmind.evaluation.service.exact_search`:
  - `INFO <domain> has <n> positions with a legal action`

Every agent search also logs its summary at INFO (see `mcts/README.md`).

## Notes

- Tests: `builder/evaluator_builder_tests.py`, `factory/evaluator_factory_tests.py`,
  `mapper/report_json_mapper_tests.py`, `repository/report_repository_tests.py`, `service/evaluator_tests.py`,
  `service/exact_search_tests.py`, `service/match_runner_tests.py`; integration:
  `test/integration/tictactoe_exact_search_tests.py`; end-to-end: `test/end_to_end/evaluate_tictactoe_tests.py`.
