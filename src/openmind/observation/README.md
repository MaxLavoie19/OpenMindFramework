# observation

## Purpose

What each player sees of a state. In a game with hidden information, such as the prisoner's dilemma, where a player
can't see the other's choice, a player must choose from what they see, never from the true state. A domain's
`Observation` says which variables each player can't see and what those variables could be; the `StateObserver`
applies it. Policies are given what their player sees, and the search considers every state that could be true.

## Content

| File | What it is |
|---|---|
| `model/observation.py` | `Observation(hidden, completions, definitions=None)`: two rules reading a state and the parameter `player`, and the script they see when they are source; each is `PythonRule` source or the project's own function (`HiddenRule`, `CompletionsRule`, see `rule/README.md`) |
| `constant/observation_constant.py` | `HIDDEN` (`"<hidden>"`), the value a hidden variable shows, and `PLAYER` (`"player"`), the rules' parameter |
| `service/state_observer.py` | `StateObserver`: what a player sees of a state, and the states that could be true given what they see |
| `builder/state_observer_builder.py` | `StateObserverBuilder`: wires a state observer with its rule compiler and rule runner |
| `factory/state_observer_factory.py` | `create_state_observer()` |

## The rules

- **`hidden`** gives the names of the variables the player can't see, such as `(chosen_name(other(player)),)`. A name
  the state doesn't have raises `KeyError`.
- **`completions`** reads what the player sees, hidden variables showing `HIDDEN`, and gives the possible values of the
  hidden variables together, as a list of `(values by name, probability)`. Each completion gives exactly the hidden
  variables, and the probabilities sum to 1; otherwise `ValueError`.

A domain's actions must be decidable from what the player to act sees: the search finds the root's legal actions from
it.

## Usage

```python
from openmind.agent.factory.domain_factory import create_domain
from openmind.observation.factory.state_observer_factory import create_state_observer
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.model.action import Action

domain = create_domain("prisonersdilemma")
observer = create_state_observer()
((state, _),) = create_predictor().predict(
    domain.transitions, domain.initial_state, Action("choose", (("choice", "defect"),))
).outcomes

seen = observer.observe(domain.observation, state, "B")        # chosen(A) = '<hidden>'
observer.completions(domain.observation, seen, "B")            # chosen(A) cooperate at 0.5, defect at 0.5
```

## Where it applies

- `Domain.observation`, `None` when every player sees everything.
- `MatchRunner` and `openmind-play` give each policy, and print for each human, what the player to act sees.
- `Agent` and `TreeSearch` search from what the player to act sees, over the states that could be true (see
  `mcts/README.md`); an agent with a theory of mind searches once per hypothesis, `CompletionTheory` grouping the
  completions into hypotheses.
- `ExactSearch` raises `ValueError` on a domain with an observation.
- Self-play and training still record true states: not done yet.

## Notes

- The observer doesn't log.
- Tests: `service/state_observer_tests.py`.
