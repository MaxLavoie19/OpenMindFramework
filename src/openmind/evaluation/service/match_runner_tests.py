import logging
import random
import re
from dataclasses import replace

import pytest

from openmind.agent.factory.rock_paper_scissors_factory import create_rock_paper_scissors_domain
from openmind.agent.model.domain import Domain
from openmind.agent.service.random_policy import RandomPolicy
from openmind.agent.service.timekeeper import Timekeeper
from openmind.agent.service.timekeeper_tests import TIMEOUT, alternating_steps
from openmind.csp.factory.csp_factory import create_solver
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.evaluation.model.match_game import MatchGame
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.observation.model.observation import Observation
from openmind.parallel.service.task_runner import TaskRunner
from openmind.mcts.service.tree_search_tests import Ticking
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
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


def test_each_game_is_logged_as_it_ends(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.evaluation.service.match_runner")

    series("win", "lose")

    lines = [
        record.getMessage()
        for record in caplog.records
        if record.name == "openmind.evaluation.service.match_runner" and record.levelno == logging.INFO
    ]
    assert [re.sub(r"seeds \d+ and \d+", "seeds", line) for line in lines] == [
        "Game with seeds finished in 1 plies, the evaluated policy playing A: payoffs A=1.0 B=0.0",
        "Game with seeds finished in 1 plies, the evaluated policy playing B: payoffs A=0.0 B=1.0",
    ]


def test_players_acting_at_once_each_choose_and_their_actions_are_taken_together(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.evaluation.service.match_runner")
    domain = create_rock_paper_scissors_domain()

    def random_policy(seed: int) -> RandomPolicy:
        return RandomPolicy(create_solver(), random.Random(seed))

    results = new_match_runner().series(domain, random_policy, random_policy, "random", 4, random.Random(1))

    lines = [record.getMessage() for record in caplog.records if record.name == "openmind.evaluation.service.match_runner"]
    assert results.games == 4 and results.wins + results.draws + results.losses == 4
    assert len([line for line in lines if "finished in 1 plies" in line]) == 4


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


class Stepping:
    """A policy stepping every time, remembering the steps it's told it has played and the clocks it's given."""

    def __init__(self) -> None:
        self.told: list[tuple[int, Clock | None]] = []

    def choose(
        self, domain: Domain, state: State, player: str | None = None, clock: Clock | None = None, steps_played: int = 0
    ) -> Action:
        self.told.append((steps_played, clock))
        return Action("step", ())


def clocked_match_runner() -> MatchRunner:
    """A match runner whose referee reads a time source moving on by a second a reading: every choice takes a second."""
    return MatchRunner(
        create_solver(), create_predictor(), StateReader(), TaskRunner(1), timekeeper=Timekeeper(create_rule_caller(), Ticking())
    )


def test_on_a_clock_each_policy_is_given_its_clock_and_its_own_steps_played() -> None:
    first, second = Stepping(), Stepping()

    game = clocked_match_runner().play_game(
        alternating_steps(TIMEOUT), lambda seed: first, lambda seed: second, 0, 1, 1, TimeControl(1.5, 1.0)
    )

    payoffs, flagged = game.payoffs, game.flagged
    assert payoffs == (0.5, 0.5) and flagged is None
    assert first.told == [(0, Clock(1.5, 1.0)), (1, Clock(1.5, 1.0))]
    assert second.told == [(0, Clock(1.5, 1.0)), (1, Clock(1.5, 1.0))]


def test_wins_and_losses_on_time_count_from_the_evaluated_side() -> None:
    results = clocked_match_runner().series(
        alternating_steps(TIMEOUT), lambda seed: Stepping(), lambda seed: Stepping(), "stepping", 2, random.Random(1), TimeControl(1.5)
    )

    # A runs out of time on its second move: game 1 the evaluated policy is A and loses, game 2 it is B and wins
    assert results == MatchResults("stepping", 2, 1, 0, 1, TimeControl(1.5), 1, 1)


def test_players_acting_at_once_are_each_charged_on_their_own_clock(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.evaluation.service.match_runner")
    domain = replace(create_rock_paper_scissors_domain(), timeout=TIMEOUT)

    def random_policy(seed: int) -> RandomPolicy:
        return RandomPolicy(create_solver(), random.Random(seed))

    game = clocked_match_runner().play_game(domain, random_policy, random_policy, 0, 1, 1, TimeControl(1.5))

    assert game.flagged is None
    assert any("finished in 1 plies on 0.025+0, clocks A=0.50 B=0.50, the evaluated policy" in message for message in caplog.messages)


def test_players_acting_at_once_whose_time_runs_out_end_the_game_by_the_timeout_rule() -> None:
    domain = replace(create_rock_paper_scissors_domain(), timeout=TIMEOUT)

    def random_policy(seed: int) -> RandomPolicy:
        return RandomPolicy(create_solver(), random.Random(seed))

    game = clocked_match_runner().play_game(domain, random_policy, random_policy, 0, 1, 1, TimeControl(0.5))

    assert game.flagged == "A"


def test_a_series_on_a_clock_needs_a_timeout_rule() -> None:
    with pytest.raises(ValueError, match="no timeout rule"):
        clocked_match_runner().series(
            first_mover_decides(), lambda seed: Always("win"), lambda seed: Always("win"), "always win", 1, random.Random(1), TimeControl(60.0)
        )


def test_each_game_is_given_as_it_ends_with_the_evaluated_policy_s_seat() -> None:
    given: list[tuple[int, int, MatchGame]] = []

    new_match_runner().series(
        first_mover_decides(),
        lambda seed: Always("win"),
        lambda seed: Always("lose"),
        "always lose",
        3,
        random.Random(1),
        on_game=lambda index, seat, game: given.append((index, seat, game)),
    )

    assert [(index, seat, game.payoffs, game.plies) for index, seat, game in given] == [
        (0, 0, (1.0, 0.0), 1),
        (1, 1, (0.0, 1.0), 1),
        (2, 0, (1.0, 0.0), 1),
    ]
