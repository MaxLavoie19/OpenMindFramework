from collections.abc import Callable
from dataclasses import replace

import pytest

from openmind.agent.factory.agent_factory import create_actor, create_agent, create_hierarchy, create_level
from openmind.agent.model.delegation import Delegation
from openmind.agent.service.level_tests import Playing
from openmind.budget.model.budget import Budget
from openmind.knowledge.constant.rule_kind_constant import ABSTRACTION
from openmind.knowledge.constant.task_constant import ABSTRACTION as ABSTRACTION_TASK
from openmind.knowledge.model.goal import Goal
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule
from openmind.search.factory.search_factory import create_minimax
from openmind.search.model.guidance import Guidance
from openmind.world.model.state import State
from openmind.world.service.world import World

type Declared = Callable[..., RuleBasedGame]
type Linked = Callable[..., RuleRecord]

#: A state nobody has a legal action in: a level starting there is over as soon as it looks.
SETTLED = State.of(turn="X", resources=12, secret="where the mines are")


def _sitting_in(knowledge: KnowledgeBase, child: RuleBasedGame, parent: RuleBasedGame) -> None:
    """Puts the child's context in the parent's, as a level of the hierarchy sits in the one above it."""
    context = knowledge.context_by_id(child.context_id)
    assert context is not None
    knowledge.context(replace(context, parent=parent.context_id))


def test_a_parent_delegating_runs_the_child_s_level_and_gets_its_report(
    declared: Declared, knowledge: KnowledgeBase
) -> None:
    child = declared(SETTLED, context="micromanagement")
    hierarchy = create_hierarchy([create_level(knowledge, child, Guidance("me"))])
    goal = knowledge.goal(Goal("take the fight", child.context_id)).id

    report = hierarchy.delegate(knowledge, Delegation(child.context_id, goal, Budget(5.0), "me"))

    assert report.reached and report.state == SETTLED
    assert report.seconds < 5.0  # it reported as soon as its level was over, rather than spending what it was given


def test_delegating_where_no_level_runs_says_so(declared: Declared, knowledge: KnowledgeBase) -> None:
    declared(SETTLED, context="micromanagement")
    hierarchy = create_hierarchy()

    with pytest.raises(ValueError, match="No level runs in"):
        hierarchy.delegate(knowledge, Delegation("context-nothing-runs-here", "goal-none", Budget(5.0)))


def test_the_children_of_a_context_are_the_levels_sitting_in_it(declared: Declared, knowledge: KnowledgeBase) -> None:
    parent = declared(SETTLED, context="macromanagement")
    child = declared(SETTLED, context="micromanagement")
    _sitting_in(knowledge, child, parent)
    hierarchy = create_hierarchy(
        [create_level(knowledge, parent, Guidance("me")), create_level(knowledge, child, Guidance("me"))]
    )

    assert [level.context for level in hierarchy.children(knowledge, parent.context_id)] == [child.context_id]
    assert hierarchy.children(knowledge, child.context_id) == ()


def test_what_the_world_says_reaches_every_level_as_that_level_sees_it(
    declared: Declared, knowledge: KnowledgeBase, heuristic: Linked
) -> None:
    parent = declared(SETTLED, context="macromanagement")
    child = declared(SETTLED, context="micromanagement")
    heuristic(
        "macromanagement",
        "it sees what it has",
        PythonRule("{'resources': resources}"),
        1.0,
        kind=ABSTRACTION,
        task=ABSTRACTION_TASK,
    )
    hierarchy = create_hierarchy(
        [create_level(knowledge, parent, Guidance("me")), create_level(knowledge, child, Guidance("me"))]
    )

    seen = hierarchy.perceived(parent.node(SETTLED))

    assert seen[parent.context_id] == State.of(resources=12)
    assert seen[child.context_id] == SETTLED


def test_a_parent_is_told_where_its_child_stands_after_every_step(
    game: Callable[[str], RuleBasedGame], declared: Declared, knowledge: KnowledgeBase, heuristic: Linked
) -> None:
    """The coach's case: its child plays the moves and tells it of each one as it is played, in time to comment on it.
    What the coach makes of what it is told is its own abstraction's."""
    coaching = declared(State.of(turn="X"), context="coaching")
    played = game("tictactoe")
    _sitting_in(knowledge, played, coaching)
    heuristic(
        "coaching", "it sees the position", PythonRule("{'position': str(cell.cells)}"), 1.0, kind=ABSTRACTION, task=ABSTRACTION_TASK
    )
    world = World(played.start())
    child = create_level(
        knowledge, played, Guidance("X"), create_agent(create_minimax(), create_actor(Playing(played, world, "O"))), world
    )
    coach = create_level(knowledge, coaching, Guidance("coach"))
    hierarchy = create_hierarchy([coach, child])
    goal = knowledge.goal(Goal("reach an interesting position", played.context_id)).id

    report = hierarchy.delegate(knowledge, Delegation(played.context_id, goal, Budget(30.0), "X"))

    assert report.reached
    seen = coach.world.current()
    assert seen.names() == ("position",)  # the coach holds what its own rules say, not the child's whole state
    assert seen.value("position") == str(report.state.model("cell").cells)  # type: ignore[union-attr]
    assert coach.world.changes() > 1  # it was told as each move was played, not only once the game was over


def test_a_child_running_alongside_its_parent_reports_when_it_is_done(
    declared: Declared, knowledge: KnowledgeBase
) -> None:
    child = declared(SETTLED, context="micromanagement")
    hierarchy = create_hierarchy([create_level(knowledge, child, Guidance("me"))])
    goal = knowledge.goal(Goal("take the fight", child.context_id)).id
    parent = Budget(10.0)

    delegated = hierarchy.delegate_alongside(knowledge, Delegation(child.context_id, goal, parent.alongside(5.0), "me"))
    report = delegated.report(5.0)

    assert report is not None and report.reached
    assert parent.seconds == 10.0  # the child's seconds were its own
    assert not delegated.running()
