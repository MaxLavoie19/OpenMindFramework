# entrypoint

## Purpose

Every way to run the framework. Entrypoints handle input, output and where logs go; the domains do the work.

## Content

| File | What it is |
|---|---|
| `play.py` | `openmind-play`: play a domain within the agent in the terminal, as humans or with the agent |

## `openmind-play`

```bash
.venv/bin/openmind-play tictactoe                                 # two humans
.venv/bin/openmind-play tictactoe --agent O                       # human X against the agent
.venv/bin/openmind-play tictactoe --agent X --agent O --seed 7    # the agent against itself
```

| Option | Default | Meaning |
|---|---|---|
| `--agent PLAYER` | none | a player the agent controls; repeat for several |
| `--iterations N` | `1000` | MCTS iterations per agent move |
| `--seed S` | unseeded | random seed for the agent's search |
| `--log-level LEVEL` | `INFO` | lowest level saved in the game log: `DEBUG`, `INFO` or `WARNING` |
| `--log-directory DIR` | `data/log/play` | where game logs are saved |

1. Prints the state, one `name = value` line per variable.
2. On a human's turn, lists the legal actions by number and reads the number of the action to perform. Anything else
   asks again; end of input (Ctrl+D) ends the session.
3. On an agent's turn, the agent searches and the CLI prints `<player> chose <action>`.
4. The predictor gives the outcome distribution; when there are several outcomes, one is drawn by its probability.
5. When no action is legal, prints the final state, payoffs included.

Domains are created by name through `agent/factory/domain_factory.py`; an `--agent` player the domain doesn't have is
rejected.

## Logs

Each session writes `<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, from `--log-level` up, as
`LEVEL logger message` lines. At `INFO`: the agent's search results (see `mcts/README.md`), plus, from logger
`openmind.entrypoint.play`:

- `INFO Playing <domain>`
- `INFO Chose <action>`
- `INFO No legal action left: game over`
- `INFO Input ended before the game was over`

At `DEBUG`, every solver candidate, predictor effect and search iteration is added, including those inside the
agent's rollouts.

## Notes

- End-to-end test: `test/end_to_end/play_tictactoe_tests.py`.
