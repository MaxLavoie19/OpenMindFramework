from collections.abc import Callable

from openmind.agent.factory.agent_factory import create_actor, create_agent, create_level
from openmind.agent.model.delegation import Delegation
from openmind.budget.model.budget import Budget
from openmind.knowledge.constant.knowledge_constant import DONE
from openmind.knowledge.constant.rule_kind_constant import ABSTRACTION
from openmind.knowledge.constant.task_constant import ABSTRACTION as ABSTRACTION_TASK
from openmind.knowledge.model.goal import Goal
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule
from openmind.search.factory.search_factory import create_minimax
from openmind.search.model.guidance import Guidance
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.world import World

type Game = Callable[[str], RuleBasedGame]
type Declared = Callable[..., RuleBasedGame]
type Linked = Callable[..., RuleRecord]


class Playing:
    """The integrator of these tests: it performs what the level dispatches through the game's own rules, plays the
    other player's first legal reply, and pushes back the state that came of it."""

    def __init__(self, game: RuleBasedGame, world: World, other: str) -> None:
        self._game = game
        self._world = world
        self._other = other

    def dispatch(self, action: Action) -> None:
        state = self._applied(self._world.current(), action)
        replies = self._game.actions(state, limit=1, player=self._other)
        if self._game.ended(state) is None and replies:
            state = self._applied(state, replies[0])
        self._world.happened(action, state)

    def performing(self) -> tuple[Action, ...]:
        return ()

    def _applied(self, state: State, action: Action) -> State:
        outcome, _ = self._game.outcomes(state, action).outcomes[0]
        return outcome


def _goal(knowledge: KnowledgeBase, game: RuleBasedGame, name: str = "win") -> str:
    return knowledge.goal(Goal(name, game.context_id)).id


def test_a_level_plays_its_own_game_until_it_ends_and_reports_what_came_of_it(
    game: Game, knowledge: KnowledgeBase
) -> None:
    played = game("tictactoe")
    world = World(played.start())
    level = create_level(
        knowledge, played, Guidance("X"), create_agent(create_minimax(), create_actor(Playing(played, world, "O"))), world
    )

    report = level.run(knowledge, Delegation(played.context_id, _goal(knowledge, played), Budget(30.0), "X"))

    assert report.reached and report.state is not None and not played.joint_actions(report.state)
    assert report.seconds > 0.0


def test_a_level_that_runs_out_of_seconds_reports_that_it_didn_t_get_there(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    level = create_level(knowledge, played, Guidance("X"))

    report = level.run(knowledge, Delegation(played.context_id, _goal(knowledge, played), Budget(0.05), "X"))

    assert not report.reached
    assert report.seconds >= 0.05


def test_what_a_delegation_brought_is_kept_as_a_task_in_the_child_s_context(
    game: Game, knowledge: KnowledgeBase
) -> None:
    played = game("tictactoe")
    world = World(played.start())
    level = create_level(
        knowledge, played, Guidance("X"), create_agent(create_minimax(), create_actor(Playing(played, world, "O"))), world
    )

    report = level.run(knowledge, Delegation(played.context_id, _goal(knowledge, played), Budget(30.0), "X"))

    (task,) = knowledge.tasks(played.context_id, DONE)
    assert task.id == report.task and task.name == f"win in {knowledge.readable_context(played.context_id)}"
    assert task.expected_time.value == 30.0
    assert [belief.value for belief in task.value] == [report.value]


def test_a_level_sees_the_abstraction_its_own_context_declares_of_the_world(
    declared: Declared, knowledge: KnowledgeBase, heuristic: Linked
) -> None:
    state = State.of(turn="X", resources=12, secret="where the mines are")
    played = declared(state)
    heuristic(
        "a game", "it sees what it has", PythonRule("{'resources': resources}"), 1.0, kind=ABSTRACTION, task=ABSTRACTION_TASK
    )
    level = create_level(knowledge, played, Guidance("me"))

    seen = level.perceived(played.node(state))

    assert seen == State.of(resources=12)
    assert level.world.current() == seen


def test_a_level_with_no_abstraction_model_sees_the_state_as_it_is(declared: Declared, knowledge: KnowledgeBase) -> None:
    state = State.of(turn="X", resources=12)
    played = declared(state)
    level = create_level(knowledge, played, Guidance("me"))

    assert level.perceived(played.node(state)) == state
