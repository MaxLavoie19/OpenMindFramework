from collections.abc import Callable
from dataclasses import replace

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.constant.prisoners_dilemma_constant import STANDARD
from openmind.agent.factory.agent_factory import create_agent
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.agent.factory.prisoners_dilemma_factory import declare_prisoners_dilemma
from openmind.agent.model.prisoners_dilemma_variant import PrisonersDilemmaVariant
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]


def dilemma(knowledge: KnowledgeBase, variant: PrisonersDilemmaVariant) -> RuleBasedSystem:
    """The prisoner's dilemma variant declared under its own context, whether or not it is one of the known ones."""
    return create_rule_based_system(knowledge, declare_prisoners_dilemma(knowledge, variant))

ONE_ROUND = replace(STANDARD, name="oneround", rounds=1)


def choose(choice: str) -> Action:
    return Action("choose", (("choice", choice),))


def after(knowledge: KnowledgeBase, *choices: str) -> State:
    rbs = dilemma(knowledge, ONE_ROUND)
    state = rbs.start()
    for choice in choices:
        ((state, _),) = rbs.outcomes(state, choose(choice)).outcomes
    return state


def test_in_one_round_the_agent_defects_as_either_player(knowledge: KnowledgeBase, game: Game) -> None:
    rbs = dilemma(knowledge, ONE_ROUND)

    assert create_agent(300, seed=1).choose(rbs, rbs.start()) == choose("defect")
    assert create_agent(300, seed=1).choose(rbs, after(knowledge, "cooperate")) == choose("defect")


def test_the_second_player_defects_whatever_the_first_chose(knowledge: KnowledgeBase) -> None:
    rbs = dilemma(knowledge, ONE_ROUND)

    for first in ("cooperate", "defect"):
        assert create_agent(200, seed=3).choose(rbs, after(knowledge, first)) == choose("defect")


SIMULTANEOUS_ONE_ROUND = replace(STANDARD, name="simultaneousoneround", rounds=1, simultaneous=True)


def test_a_simultaneous_round_plays_both_choices_at_once(knowledge: KnowledgeBase, game: Game) -> None:
    rbs = dilemma(knowledge, SIMULTANEOUS_ONE_ROUND)
    joint = JointAction((("A", choose("cooperate")), ("B", choose("defect"))))

    ((state, _),) = rbs.joint_outcomes(rbs.start(), joint).outcomes

    variables = dict(state.variables)
    assert (variables["payoff(A)"], variables["payoff(B)"], variables["turn(A)"], variables["turn(B)"]) == (0, 5, False, False)


def test_in_one_simultaneous_round_both_players_mostly_defect(knowledge: KnowledgeBase, game: Game) -> None:
    rbs = dilemma(knowledge, SIMULTANEOUS_ONE_ROUND)
    agent = AgentBuilder().with_iterations(2000).with_exploration(EXPLORATION).with_seed(1).build()

    for player in ("A", "B"):
        assert dict(agent.search(rbs, rbs.start(), player).strategy)[choose("defect")] > 0.7
