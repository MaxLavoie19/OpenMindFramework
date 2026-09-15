import random
from dataclasses import replace

import pytest

from openmind.agent.model.domain import Domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.observation.model.observation import Observation
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


class Always:
    """A policy that always chooses the same action."""

    def __init__(self, name: str) -> None:
        self._action = Action(name, ())

    def choose(self, domain: Domain, state: State) -> Action:
        return self._action


def new_match_runner() -> MatchRunner:
    return MatchRunner(create_solver(), create_predictor(), StateReader(), TaskRunner(1))


def first_mover_decides(players: tuple[str, ...] = ("A", "B")) -> Domain:
    """Only the first player acts: win gives them 1 and the other 0, lose the reverse, tie 0.5 each."""
    first, *others = players
    unset = PythonRule(f"payoff[{first!r}] is None")

    def payoffs(mine: float, theirs: float) -> PythonRule:
        return PythonRule(
            "\n".join((f"payoff[{first!r}] = {mine!r}", *(f"payoff[{other!r}] = {theirs!r}" for other in others)))
        )

    return Domain(
        "first mover decides",
        State((*((f"payoff({player})", None) for player in sorted(players)), ("turn", first))),
        Problem(tuple(ActionDefinition(name, (), (unset,)) for name in ("win", "lose", "tie"))),
        TransitionModel(
            (
                Transition("win", (Branch(1.0, payoffs(1.0, 0.0)),)),
                Transition("lose", (Branch(1.0, payoffs(0.0, 1.0)),)),
                Transition("tie", (Branch(1.0, payoffs(0.5, 0.5)),)),
            )
        ),
        Players(players, "turn", tuple(f"payoff({player})" for player in players)),
    )


def series(evaluated: str, opponent: str, games: int = 2) -> MatchResults:
    return new_match_runner().series(
        first_mover_decides(),
        lambda seed: Always(evaluated),
        lambda seed: Always(opponent),
        "always " + opponent,
        games,
        random.Random(1),
    )


def test_seats_switch_every_game() -> None:
    # game 1: the evaluated policy is A and wins; game 2: the opponent is A, loses, so the evaluated B wins
    assert series("win", "lose") == MatchResults("always lose", 2, 2, 0, 0)


def test_losses_and_wins_count_from_the_evaluated_side() -> None:
    assert series("lose", "lose") == MatchResults("always lose", 2, 1, 0, 1)


def test_equal_payoffs_are_draws() -> None:
    assert series("tie", "tie", games=3) == MatchResults("always tie", 3, 0, 3, 0)


def test_each_game_creates_its_policies_from_its_own_seed() -> None:
    seeds: list[int] = []

    def remember(seed: int) -> Always:
        seeds.append(seed)
        return Always("tie")

    new_match_runner().series(first_mover_decides(), remember, lambda seed: Always("tie"), "always tie", 3, random.Random(1))

    assert len(set(seeds)) == 3


class Recording:
    """A policy that always chooses the same action and records every state it is given."""

    def __init__(self, name: str) -> None:
        self._action = Action(name, ())
        self.states: list[State] = []

    def choose(self, domain: Domain, state: State) -> Action:
        self.states.append(state)
        return self._action


def test_a_policy_is_given_only_what_its_player_sees() -> None:
    hides_b = Observation(PythonRule("('payoff(B)',) if player == 'A' else ()"), PythonRule("[({'payoff(B)': None}, 1.0)]"))
    domain = replace(first_mover_decides(), observation=hides_b)
    recording = Recording("win")

    new_match_runner().play_game(domain, lambda seed: recording, lambda seed: Always("win"), 0, 1, 1)

    assert recording.states == [State((("payoff(A)", None), ("payoff(B)", "<hidden>"), ("turn", "A")))]


def test_a_domain_without_two_players_raises() -> None:
    with pytest.raises(ValueError, match="two players"):
        new_match_runner().series(
            first_mover_decides(("A",)), lambda seed: Always("win"), lambda seed: Always("win"), "always win", 1, random.Random(1)
        )
