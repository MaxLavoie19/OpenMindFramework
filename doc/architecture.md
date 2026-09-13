# Architecture

## Vocabulary

- **State**: a set of named variables with values, such as `cell(2,3) = "X"` or `turn = "O"`.
- **Action**: the thing performed (dance, sing, move, …) with its parameters, such as `place(row=2, col=3)`.
- **Outcome**: the new state after an action.
- **Outcome probability distribution**: each possible outcome of an action with its probability, given by the
  predictor.
- **Payoff**: what each player gets when the game ends; a predicted outcome of a transition to a win or a draw.
- **Domain (within the agent)**: a problem the agent works on, such as tic-tac-toe. It is not a domain in the code.

## Rules of a domain

| Rule | Lives in |
|---|---|
| Constraints on actions and their parameters | the domain's constraint satisfaction problem (CSP) |
| Transitions, end of game and payoffs | the predictor |
| Strategy, weighted by expected value | the strategy rule-based system (RBS) |
| Players and initial state | the domain's factory |

Rules are structured Python objects that an interpreter evaluates. They are never functions and cannot generate code.
Factories bootstrap them for now; later, a decoder will turn unstructured data into structured instructions that
become rules.

## Code domains

| Domain | Owns | Status |
|---|---|---|
| `world` | `Value`, `State`, `Action` | iteration 1 |
| `expression` | Rule expressions and the `Interpreter` that evaluates them | iteration 1 |
| `csp` | Action definitions, parameter domains, constraints, `Solver` | iteration 1 |
| `agent` | The agent and its domains | iterations 1–2: tic-tac-toe factory only |
| `predictor` | Transitions, outcome probability distributions | iteration 2 |
| `mcts` | Search | planned |
| `tracking` | Played games, training runs, metrics and logs in `data/` | planned |
| `rbs` | Strategy rules, distillation, explanations | planned |
| `optimizer` | Strategic discrete actions from continuous action spaces | later |

## Conventions

- Code lives in `src/openmind/<domain>/<class type>/`. Class types: `model` (datatypes), `factory` (functions that
  build an object with a builder and a recipe), `builder`, `service` (operations on data), `repository` (storing and
  retrieving data), `mapper` (format conversions) and `constant`. Entrypoints live in `src/openmind/entrypoint/`.
- Each domain folder has a `README.md` covering its purpose, content, usage and logs.
- One class per module, named after it; a name that clashes with a Python keyword gets a trailing underscore
  (`not_.py`).
- Models are frozen, slotted dataclasses. Logic lives in services; I/O only in entrypoints.
- Services log through `logging.getLogger(__name__)`: what was decided and why, with the key values. Actions and
  expressions appear in logs as readable text (`ActionTextMapper`, `ExpressionTextMapper`).
- Unit tests sit beside their target as `<module>_tests.py`; integration and end-to-end tests live in `test/`.
- `data/` holds all data (databases, trained models, logs, …); its layout is decided as we go. Tests save their logs
  in `data/log/<test file>/<test name>.log`, which git ignores.
