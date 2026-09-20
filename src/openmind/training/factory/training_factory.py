from openmind.agent.factory.agent_factory import create_agent
from openmind.agent.service.outfitter import Outfitter
from openmind.rbs.factory.rbs_factory import create_rule_heuristic
from openmind.training.service.arm_selector import ArmSelector
from openmind.training.service.heuristic_ranker import HeuristicRanker
from openmind.agent.service.agent import Agent
from openmind.search.factory.search_factory import create_monte_carlo_tree_search
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.service.self_play import SelfPlay
from openmind.training.service.value_distiller import ValueDistiller
from openmind.rbs.factory.rbs_factory import create_value_generator


def create_self_play(agent: Agent | None = None) -> SelfPlay:
    """An agent playing a game against itself, exploring the lines it can unless another agent is given."""
    return SelfPlay(create_agent(create_monte_carlo_tree_search()) if agent is None else agent)


def create_position_row_mapper() -> PositionRowMapper:
    """Games played into the rows a heuristic is fitted on."""
    return PositionRowMapper()


def create_value_distiller(agent: Agent | None = None, workers: int = 1) -> ValueDistiller:
    """Learning a position heuristic from games the agent played against itself, fitting terms in that many worker
    processes."""
    return ValueDistiller(create_self_play(agent), create_position_row_mapper(), create_value_generator(workers))


def create_heuristic_ranker(agent: Agent | None = None) -> HeuristicRanker:
    """Ranking heuristics by playing them against each other, pairing them by UCB1."""
    return HeuristicRanker(create_self_play(agent), ArmSelector(), Outfitter(create_rule_heuristic()))
