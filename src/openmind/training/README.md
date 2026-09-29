# training

## Purpose

Learns from the agent's own play, and decides between what it has learned by playing it.

Two things happen here and they are different questions. **Distilling** fits a position heuristic on the games the
agent played against itself, which is learning *from* games. **Matching** plays two registered models of one task
against each other and writes what came of it where the registry reads it, which is deciding *by* games. Nothing
else in OMF can say which of two heuristics is better, because how close a value is to the truth is what a fit's
held-out loss already measures and a heuristic can be well calibrated and lose.

Self-play is the only source of games. No external teacher is used, and that is a decision rather than an
omission: an opponent that already played the game well would teach the game, and what is being tested is whether
the rules can be learned from nothing. The price of it is known — Baxter, Tridgell and Weaver found KnightCap
"unable to learn effectively from games of self-play", and Veness and others measured a trained-against-an-engine
run about 150 Elo above the same algorithm self-played. It is the thing to watch, not the thing to fix.

## Content

| File | What it is |
|---|---|
| `service/self_play.py` | `SelfPlay.play(knowledge_base, game, guidance, settings)`: that many games, each from its own seed, every finished one remembered as it ends; `guidance` is one `Guidance` for every player or a mapping of player to `Guidance`, which is how two models are put on opposite sides. `play_game` plays one |
| `service/value_distiller.py` | `ValueDistiller.distill(knowledge_base, game, guidance, play, values, held_out_games=0)`: plays, turns the games into rows, fits terms over them and declares what held up. The held-out games are played after the rest and kept back, so the rules are chosen on games they were not fitted on |
| `service/model_match.py` | `ModelMatch.play(knowledge_base, game, one, other, settings)`: two models of one task over that many games, each playing each side, scored onto both through `AccuracyScorer.played`. None where they are models of no one task, where the game is not of two players, or where either will not load |
| `service/game_replayer.py` | `GameReplayer.positions(game, summary)`: a remembered game played again from its actions and its outcome seed, its positions exactly as they were — so a game can be shown again without every position having been kept |
| `service/checked_play.py` | Whatever decides whether a move was really allowed and what it leads to, so a run of learned rules can be played against the game that owns the truth |
| `mapper/position_row_mapper.py` | `PositionRowMapper.to_rows(game, games)`: played games into the rows a heuristic is fitted on — every position a game went through, once for each player, valued at what that game paid that player |
| `model/played_game.py` | `PlayedGame(states, actions, payoffs=(), ending=None, agent_seed=None, outcome_seed=None)`: a game played out. Its two seeds are kept apart so it can be played again from its actions alone |
| `model/self_play_settings.py` | `SelfPlaySettings(games=1, seconds=1.0, steps=None, seed=0)`: what self-play may spend |
| `model/distillation.py` | `Distillation(context, rules, games, decisive, training_rows, held_out_rows, seconds, error=None)`: what distilling came to |
| `model/match.py` | `Match(context, task, one, other, games, points, sides)`: what a match came to, with `decisive` and `winner`. A win is one point, a draw a half, a loss none |
| `factory/training_factory.py` | `create_self_play`, `create_value_distiller`, `create_position_row_mapper`, `create_model_match` |

## How distilling works

1. `SelfPlay` plays the training games, then the held-out games. Each draws two seeds: one its agent searches with,
   one its outcomes are drawn with. Every finished game is remembered in the knowledge base as it ends, with the
   model each player was played with.
2. `PositionRowMapper` turns the games into rows: every position, once per player, valued at what that game paid
   that player. One game's result is a noisy value for its early positions, and it is what the position led to in
   the agent's own play.
3. `HeuristicFinder` fits terms over the training rows and chooses a price on the held-out ones (see `rbs/README.md`).
4. `error` is the chosen rules' loss on the held-out rows. **Those rows also chose the price**, so it is not
   independent of that choice.

A game that paid nobody is not learned from: what is remembered is finished games, and a game cut short has no
result to remember it by.

## How matching works

`ModelMatch.play` puts one model in each seat, plays half the games one way round and half the other, and scores
each by the share of the points it took. Both models played every game, one on each side of it, so both are scored
over the same count.

The score goes through `AccuracyScorer.played` into the mechanism's accuracy belief, which is what
`ModelRegistry.best` reads. That link is the whole point of this service. Every finished game was already kept —
with a payoff and an outcome belief per player, tagged with the model that played that side — and none of it
reached the belief anything selects on, so a model could win a hundred games and `best` would go on preferring
whichever was registered first.

A model never played keeps **no** accuracy rather than a low one, and a match leans on what a mechanism declared of
itself the same way a settled belief does, so three games are not trusted like three hundred.

## Usage

```python
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.rbs.factory.rbs_factory import create_game
from openmind.rbs.model.value_settings import ValueSettings
from openmind.search.model.guidance import Guidance
from openmind.training.factory.training_factory import create_value_distiller
from openmind.training.model.self_play_settings import SelfPlaySettings

knowledge_base = create_knowledge_base("tictactoe", somewhere)
game = create_game("tictactoe", knowledge_base)
values = ValueSettings((0.1, 0.01, 0.001), 1000, 1e-6, 60.0, 8 * 1024**3)   # see rbs/README.md
distilled = create_value_distiller(workers=8).distill(
    knowledge_base, game, Guidance("X"), SelfPlaySettings(games=100, seed=1), values, held_out_games=25
)
distilled.rules   # the position rules declared under distilled.context
```

## Logs

- `openmind.training.service.self_play`: a line per game as it ends, with its steps and whether it was decisive,
  then a line for the lot with how many were decisive and the mean steps
- `openmind.training.service.value_distiller`: what was distilled, from how many rows, and the held-out error;
  `Nothing to learn from <n> games of <game>: none of them paid anyone` where no game finished
- `openmind.training.service.model_match`: `<one> took <points> and <other> took <points> of <n> games of <game>
  at <task>`, or why there was no match
- `openmind.inference.service.accuracy_scorer`: `<mechanism> took <points> of <n> points: accuracy <a> over <n>
  games`

Every search logs its own summary (see `mcts/README.md`), and fitting logs its search and its prices (see
`rbs/README.md`).

## Notes

- **This file described a system that was never built.** Until it was rewritten it documented an arm library, a
  continuous trainer, an ending walker, a game study, an arm selector and their settings and repositories — seven
  of the thirteen classes it named did not exist, and two of the rest live in `agent/`. The bandit stack it
  described was deleted rather than finished: which two models to play is the caller's question and what came of
  it is a fact, and neither needs UCB1.
- Tests: `service/self_play_tests.py`, `service/model_match_tests.py`, `service/game_replayer_tests.py`,
  `service/checked_play_tests.py`, `mapper/position_row_mapper_tests.py`.
- **`ValueDistiller` and the factory have no tests**, which is worth saying here rather than leaving to be
  discovered. The first draft of this rewrite listed tests for both, because listing what ought to exist is
  exactly the habit that produced the file being replaced.
