# evaluation

## Purpose

Measures how well an agent plays a domain, so training can be judged: results against baseline opponents, and how
often the agent's choices agree with perfect play at several iteration budgets, and how long each choice takes. The
training loop, not built yet, will run it after every round to draw learning curves.

This measures the agent. The RBS's value rules, which turn a position into a value, are a model the agent can search
with; when it does, they are measured here too.

## Content

| File | What it is |
|---|---|
| `model/evaluation_settings.py` | `EvaluationSettings(games, iterations, positions, budgets, seed, reference_iterations=None, guided_rollouts=True, rollout_actions=0, rollout_limit=None, unfinished_payoff=None, time_control=None, expected_steps=30, selection='ucb1', puct_exploration=1.5, prior='uniform', prior_temperature=0.1)`; `positions` 0 skips agreement, `None` takes every position; `reference_iterations` replaces exact search with long unguided searches; `guided_rollouts` false makes the guided agent rate only its tree's nodes; `rollout_actions` is how many rollout actions a valuing agent plays before valuing; `rollout_limit` and `unfinished_payoff` stop the rollouts of every agent the evaluation builds, reference searches aside; `time_control` puts the baseline series on a clock, every agent budgeting its moves with `PlainTimeBudgetEstimator(expected_steps)`; every agent searches by `selection`, the evaluated one following the prior named (from its rules or values), the untrained one the uniform prior |
| `model/match_game.py` | `MatchGame(payoffs, flagged, plies, ending, actions, policy_seed, outcome_seed, seconds=(), clocks=())`: one game of a match, as `MatchRunner.play_game` gives it |
| `mapper/match_game_summary_mapper.py` | `MatchGameSummaryMapper`: a match's game as a `GameSummary`, the evaluated model in its seat; a match's steps have no budget |
| `model/match_results.py` | `MatchResults(opponent, games, wins, draws, losses, time_control=None, wins_on_time=0, losses_on_time=0)`: a series against one opponent, from the evaluated agent's side; on a clock, its time control and how many wins and losses came from a player's time running out |
| `model/agreement.py` | `Agreement(iterations, positions, optimal, optimal_visit_share, mean_regret, seconds_per_choice)`: how an agent searching with a number of iterations did on the sampled positions |
| `model/rater_agreement.py` | `RaterAgreement(positions, distinguishing, optimal, mean_regret)`: how a rater alone did on the sampled positions |
| `model/guidance_test.py` | `GuidanceTest(iterations, positions, low_value_share_difference, low_value_share_p, regret_difference, regret_p, optimal_only_guided, optimal_only_unguided, optimal_choice_p)`: the guided agent against the unguided one at a budget, paired by position |
| `model/evaluation_report.py` | `EvaluationReport(domain, created_at, rules_file, settings, baselines, every_action_optimal, agreement, unguided_agreement, rater=None, guidance_tests=(), values_file=None, valuer=None)`; `rules_file` names the rule base guiding the agent and `values_file` the value base valuing its positions, each `None` when not used; with neither, `unguided_agreement` and `guidance_tests` are empty and `rater` and `valuer` are `None` |
| `model/value_measure.py` | `ValueMeasure(positions, valued, mean_absolute_error, optimal, mean_regret)`: how a valuer alone did on the sampled positions |
| `service/value_measurer.py` | `ValueMeasurer`: a valuer's error on each position and its choice one step ahead |
| `service/exact_search.py` | `ExactSearch`: every legal action's value, the optimal actions and each player's value in a state, by searching every reachable state; the positions with a legal action; a domain with an observation raises `ValueError`, since perfect play with hidden information needs mixed strategies |
| `service/reference_search.py` | `ReferenceSearch`: distinct positions from uniformly random games, and every legal action's value from a long unguided search, for domains exact search can't reach |
| `service/match_runner.py` | `MatchRunner`: plays a series between two policies in a two-player domain, switching seats every game; each game creates its policies from `PolicyFactory`s with a seed of its own, in the task runner's workers; in a domain with an observation, a policy is given only what its player sees; a game that took its worker over the memory cap in a fresh worker too isn't counted; where players act at once, every player to act chooses, given its player's name, and the actions are taken together with `Predictor.predict_joint`. `series` and `play_game` take `time_control`: each choice is then timed and charged to its player's clock, given to the policy with the steps that player has played, where players act at once each on its own player's clock; a player whose time runs out doesn't play that move, and the domain's timeout rule ends the game, applied for each player whose time ran out, in the players' order. `play_game` gives a `MatchGame`, and `series` hands each one to `on_game(index, seat, game)` as it ends, the evaluated policy's seat given; with a game memory (`EvaluatorBuilder.with_game_memory`), the evaluator remembers every match game that way. A policy gives only its action, so a match's steps have no budget |
| `service/choice_measurer.py` | `ChoiceMeasurer`: searches positions with an agent built once and measures each choice against the action values; the optimal actions within a tolerance |
| `model/choice_measure.py` | `ChoiceMeasure(optimal, optimal_visit_share, regret, seconds)`: one position's choice |
| `model/action_values.py` | `ActionValues`: every legal action of a position with its value |
| `factory/baseline_policy_factory.py` | `create_built_agent(agent_builder, seed)`, which keeps the builder's seed in every game, `create_seeded_agent(agent_builder, seed)`, which searches with the game's own seed, and `create_random_policy(seed)`: policy factories for series of games |
| `service/evaluator.py` | `Evaluator`: runs the baseline series and the agreement measures, guided and unguided, for the rater alone and the paired tests when given a rater, and returns a report |
| `builder/evaluator_builder.py` | `EvaluatorBuilder`: sets how many worker processes games and searches run in (`with_workers`, 1 by default) and wires the services an evaluator measures with |
| `factory/evaluator_factory.py` | `create_evaluator(workers=1)` |
| `constant/evaluation_constant.py` | Default games, iterations, positions, budgets and seed; the word for every position (`all`); baseline opponent names; `REFERENCE_TOLERANCE` (0.05), how far below the best reference value an action is still optimal; `POSITION_ATTEMPTS` (20), random games per reference position wanted |
| `mapper/report_json_mapper.py` | `ReportJsonMapper`: a report as JSON text |
| `mapper/report_text_mapper.py` | `ReportTextMapper`: a report as readable text: baseline results, an agreement table with guided and unguided side by side when both were measured, the rules alone, and the paired tests as a table |
| `repository/report_repository.py` | `ReportRepository`: saves a report as `<directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` |

## How it measures

1. **Results against baselines:** the agent, built with `iterations`, `seed`, `guided_rollouts`, `rollout_actions` and the rollout limit, plays `games` games against
   `RandomPolicy`, then `games` games against untrained MCTS with the same iterations, switching seats every game. A
   higher payoff than the opponent is a win, an equal one a draw, a lower one a loss. Every game draws a policy seed
   and an outcome seed up front; the random policy chooses with its game's policy seed, and both agents search with
   the evaluation's `seed`.
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
3. **Guided against unguided:** when a rater guides the agent (`rater`, with `rules_file` naming it) or a valuer values
   its positions (`valuer`, with `values_file` naming it), an unguided agent with the same iterations and seed is
   measured on the same positions at every budget. The rater is also measured alone: the positions where its ratings
   separate the actions at all, and, picking uniformly among its top-rated actions, the expected number of optimal
   picks and the expected mean regret. As in the search, an action rated `None` gets the mean of the other ratings, and
   when every rating is `None` every action is top-rated. The valuer is measured alone too, by `ValueMeasurer`:
   - **value error:** the positions it values, and the mean absolute difference there between its value for the player
     to act and the best action's value;
   - **one step ahead:** every action is worth its outcomes' values for the player to act, weighted by their
     probabilities, a finished game's payoffs or the valuer's value of a position in play; picking uniformly among the
     top-valued actions, the expected number of optimal picks and the expected mean regret. An action with an outcome
     the valuer knows nothing about gets the mean of the other actions' values.
4. **Does the guidance cut low-value exploration?** At every budget, the guided and the unguided agents are compared
   position by position, as an intervention test: the same positions, the same budget, the same seed, only the
   guidance differs. Every difference is guided minus unguided.
   - **low-value visits:** the share of the root's visits on actions that aren't optimal. Its mean difference, and the
     two-sided Wilcoxon signed-rank p-value of the differences, zero differences split between the signs;
   - **regret:** the same, for the chosen action's regret;
   - **optimal choice:** the positions where only the guided agent chose an optimal action, those where only the
     unguided one did, and McNemar's exact p-value: a two-sided binomial test of the first count among both, at 1/2.

   A test without any nonzero difference, or without any position where the two disagree, has a p-value of 1. A
   negative low-value or regret difference with a small p-value says the guidance steered the search away from
   low-value actions on these positions; the p-values aren't adjusted across budgets.
5. **Fewer iterations, same play:** the agreement across budgets shows how many iterations the agent needs; a trained
   agent should reach the same agreement with fewer.
6. **Speed:** every agreement budget records the mean seconds per choice.

`ExactSearch` visits every reachable state. That suits small domains such as tic-tac-toe (4,520 positions with a legal
action), not 4 in a row or chess. For those, `reference_iterations` stands in for perfect play: `ReferenceSearch`
plays uniformly random games, takes one position with a legal action from each, drawn uniformly, until it has
`positions` distinct ones or has played `POSITION_ATTEMPTS` games per position wanted; then an unguided search of
`reference_iterations` iterations, seeded with `seed`, gives every legal action's value as its mean payoff, 0 for an
action it never visited. An action within `REFERENCE_TOLERANCE` of the best value counts as optimal. These values are
estimates: they are only as good as the search, and they favour what an unguided search finds. A reference search
needs a number of positions: `positions` = `None` raises `ValueError`.

With `positions` = 0, the evaluator skips agreement and calls neither search: the report's agreement is empty.
`MatchRunner` needs exactly two players.

**Workers.** Baseline games, reference searches, and the positions searched at each budget (split into slices, each
searched by an agent built once) run in the task runner's worker processes (see `parallel/README.md`). Every search
is seeded and every game draws its seeds up front, so a report is the same whatever the number of workers, apart from
its seconds per choice. To run in several workers, the agent builder, with the rater guiding it and the valuer valuing
its positions, must pickle; a `RuleRater` and a `RuleValuer` do. The rater alone is measured in this process, the valuer
alone in the workers, the positions split into slices.

## Usage

```python
from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.factory.domain_factory import create_domain
from openmind.evaluation.factory.evaluator_factory import create_evaluator
from openmind.evaluation.model.evaluation_settings import EvaluationSettings

report = create_evaluator(workers=8).evaluate(
    create_domain("tictactoe"),
    AgentBuilder().with_exploration(EXPLORATION),   # evaluate sets the builder's iterations and seed
    EvaluationSettings(games=100, iterations=200, positions=100, budgets=(10, 20, 50, 100, 200, 500), seed=1),
)
```

To evaluate an agent guided by rules, create `rater = create_rule_rater(rule_base, domain)`, give the builder
`with_guidance(rater)`, and pass the rule base's path as `rules_file` and the same `rater` to `evaluate`; the report
then also holds the unguided agreement, the rater alone and the paired tests. To evaluate an agent valuing positions
with value rules, create `valuer = create_rule_valuer(value_base, domain)`, give the builder `with_valuation(valuer)`,
and pass the value base's path as `values_file` and the same `valuer` to `evaluate`, with `rollout_actions` in the
settings; the report then holds the values alone instead of, or besides, the rules alone. For a domain exact search
can't reach, add `reference_iterations=2000` to the settings. `ReportTextMapper().to_text(report)` gives the summary
`openmind-evaluate` prints. From the terminal:
`openmind-evaluate tictactoe [--rules PATH] [--values PATH] [--positions all] [--reference-iterations N]` (see
`entrypoint/README.md`).

## Logs

- `openmind.evaluation.service.evaluator`:
  - `INFO Against random: <games> games, <wins> wins, <draws> draws, <losses> losses`
  - `INFO Every action is optimal in <n> of <positions> positions`
  - `INFO Agreement with perfect play at <iterations> iterations: <optimal> of <positions> positions, <share> of visits on optimal actions, mean regret <regret>, <seconds> seconds per choice`
  - `INFO Unguided agreement with perfect play at <iterations> iterations: ...`, the same measures for the unguided agent, with a rater
  - `INFO Guided against unguided at <iterations> iterations on <positions> positions: low-value visit share <difference> (p <p>), regret <difference> (p <p>), optimal choice only guided <n>, only unguided <m> (p <p>)`, with a rater
  - `INFO Rater alone: ratings separate actions in <n> of <positions> positions; a top-rated action is optimal in <expected> of <positions>; mean regret <regret>`, with a rater
  - `INFO Values alone: valued <n> of <positions> positions, mean absolute error <error> against the best action's value; one step ahead, a top-valued action is optimal in <expected> of <positions>; mean regret <regret>`, with a valuer
  - `INFO Reference values from <iterations>-iteration unguided searches on <positions> positions`, with `reference_iterations`
  - `INFO Agreement with perfect play skipped: no positions`, when `positions` is 0
  - `DEBUG <Agreement or Unguided agreement> at <iterations> iterations: chose <action>, regret <regret>; optimal: <actions>; <share> of visits on optimal actions; state: <name = value, ...>`
  - `DEBUG Rater alone: top-rated <actions>; optimal: <actions>; ratings <action>=<rating>, ...; state: <name = value, ...>`
- `openmind.evaluation.service.match_runner`:
  - `INFO Game with seeds <policy seed> and <outcome seed> finished in <plies> plies, the evaluated policy playing <player>: payoffs <player>=<payoff> ...`,
    logged by the worker as soon as the game ends, with ` by <ending>` after the plies when the domain says why games end,
    ` by <player>'s flag` when a player's time ran out, and on a clock ` on <time control>, clocks <player>=<seconds left> ...`
  - `INFO Game with seeds <policy seed> and <outcome seed> record: <record>`, when the domain records games, such as a
    chess game's PGN; games where players act at once aren't recorded yet
  - `DEBUG Step <n>: <player> took <seconds> seconds, <seconds> left`, for every choice on a clock, <n> counting that
    player's steps
  - `DEBUG Game <n> against <opponent>: evaluated agent plays <player>, payoffs <player>=<payoff> ...`
- `openmind.evaluation.service.exact_search`:
  - `INFO <domain> has <n> positions with a legal action`
- `openmind.evaluation.service.reference_search`:
  - `INFO <domain>: <n> reference positions from <games> random games`

Every agent search, reference searches included, also logs its summary at INFO (see `mcts/README.md`).

## Notes

- Tests: `builder/evaluator_builder_tests.py`, `factory/evaluator_factory_tests.py`,
  `mapper/report_json_mapper_tests.py`, `mapper/report_text_mapper_tests.py`, `repository/report_repository_tests.py`,
  `service/evaluator_tests.py`,
  `service/exact_search_tests.py`, `service/match_runner_tests.py`, `service/reference_search_tests.py`,
  `service/choice_measurer_tests.py`, `service/value_measurer_tests.py`, `factory/baseline_policy_factory_tests.py`;
  integration:
  `test/integration/tictactoe_exact_search_tests.py`; end-to-end: `test/end_to_end/evaluate_tictactoe_tests.py`,
  `test/end_to_end/evaluate_fourinarow_tests.py`.
