from openmind.parallel.service.task_runner import TaskRunner
from openmind.training.service.agreement_dispatcher import AgreementDispatcher
from openmind.agent.factory.agent_factory import create_agent
from openmind.agent.service.outfitter import Outfitter
from openmind.inference.service.accuracy_scorer import AccuracyScorer
from openmind.model.service.model_timer import ModelTimer
from openmind.rbs.factory.rbs_factory import create_rule_heuristic
from openmind.training.service.game_replayer import GameReplayer
from openmind.training.service.game_study import GameStudy
from openmind.training.service.model_match import ModelMatch
from openmind.agent.service.agent import Agent
from openmind.search.factory.search_factory import create_monte_carlo_tree_search
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.service.self_play import SelfPlay
from openmind.training.service.value_distiller import ValueDistiller
from openmind.rbs.factory.rbs_factory import create_heuristic_finder


def create_self_play(agent: Agent | None = None) -> SelfPlay:
    """An agent playing a game against itself, exploring the lines it can unless another agent is given."""
    return SelfPlay(create_agent(create_monte_carlo_tree_search()) if agent is None else agent)


def create_position_row_mapper() -> PositionRowMapper:
    """Games played into the rows a heuristic is fitted on."""
    return PositionRowMapper()


def create_value_distiller(agent: Agent | None = None, workers: int = 1) -> ValueDistiller:
    """Learning a position heuristic from games the agent played against itself, fitting terms in that many worker
    processes."""
    return ValueDistiller(create_self_play(agent), create_position_row_mapper(), create_heuristic_finder(workers))


def create_game_study() -> GameStudy:
    """The positions of games already played, read back so they can be learned from afterwards. Playing is
    timely and studying is not, so what a move may do is write the game down and what a study may do is take
    as long as it needs."""
    return GameStudy(GameReplayer())


def create_model_match(agent: Agent | None = None) -> ModelMatch:
    """Two models of one task played against each other, and the result written where the registry reads it.

    Built with what plays the games, what loads a model into the port it fills, and what scores a mechanism.
    No arm, no pairing rule: which two to play is the caller's question, and what came of it is a fact."""
    return ModelMatch(create_self_play(agent), Outfitter(create_rule_heuristic(), ModelTimer()), AccuracyScorer())


def create_agreement_dispatcher(
    workers: int = 1, among: int = 0, reading: float = 1.0, seconds: float = 0.0
) -> AgreementDispatcher:
    """Every heuristic put to the same decisions, over that many worker processes.

    One worker runs everything here, which is what a run that asked for no dispatch did before. How many is
    the caller's: the machine's cores are shared with whatever is playing games at the same time.

    `among` is how many actions a decision is put with, the played one always among them, `reading` what
    share of a ruleset's weight is read, and `seconds` how long the whole judging may take — which divides
    down to what one valuing may spend, so no single heuristic can hold the round. Nought and one ask
    everything, which is what this did before."""
    return AgreementDispatcher(TaskRunner(workers), among, reading, seconds)
