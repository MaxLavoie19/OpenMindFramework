# entrypoint

## Purpose

Every way to run the framework. Entrypoints handle input, output and where logs go; the domains do the work.

## Content

| File | What it is |
|---|---|
| `play.py` | `openmind-play`: play a domain within the agent in the terminal |

## `openmind-play`

```bash
.venv/bin/openmind-play tictactoe
.venv/bin/openmind-play tictactoe --log-directory data/log/play   # the default
```

1. Prints the state, one `name = value` line per variable, and the legal actions as a numbered list.
2. Reads the number of the action to perform. Anything else asks again; end of input (Ctrl+D) ends the session.
3. The predictor gives the outcome distribution; when there are several outcomes, one is drawn by its probability.
4. When no action is legal, prints the final state, payoffs included.

Domains are created by name through `agent/factory/domain_factory.py`.

## Logs

Each session writes `<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, from DEBUG up, as `LEVEL logger message`
lines: the solver's and predictor's logs, plus, from logger `openmind.entrypoint.play`:

- `INFO Playing <domain>`
- `INFO Chose <action>`
- `INFO No legal action left: game over`
- `INFO Input ended before the game was over`

## Notes

- End-to-end test: `test/end_to_end/play_tictactoe_tests.py`.
