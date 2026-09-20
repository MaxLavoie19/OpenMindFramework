import argparse
import logging
import random
from pathlib import Path

from openmind.agent.factory.agent_factory import create_actor, create_agent
from openmind.budget.model.budget import Budget
from openmind.entrypoint.clock_options import add_knowledge_option
from openmind.entrypoint.debug_options import add_debug_option, start_debugging
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.rbs.factory.rbs_factory import create_game
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.factory.search_factory import create_improvised, create_minimax
from openmind.search.model.guidance import Guidance
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.grid_text_mapper import GridTextMapper
from openmind.world.model.action import Action
from openmind.world.service.world import World

logger = logging.getLogger(__name__)

#: The planners a player can be given, and what each one is.
PLANNERS = {"minimax": create_minimax, "improvised": create_improvised}

#: How long an agent's move may take, in seconds.
DEFAULT_SECONDS = 5.0


class Referee:
    """The integrator of this example: it runs the game, so it performs what OMF dispatches and pushes back what came
    of it.

    OMF doesn't run games. Here the terminal does: it applies an action through the game's own rules, draws an outcome
    where there are several, and hands the state that came of it to the world both the agent and its planner read."""

    def __init__(self, game: RuleBasedGame, world: World, source: random.Random) -> None:
        self._game = game
        self._world = world
        self._source = source
        self._performing: tuple[Action, ...] = ()

    def dispatch(self, action: Action) -> None:
        """Performs the action: what the game's rules make of it becomes the state now."""
        self._performing = (action,)
        outcomes = self._game.outcomes(self._world.current(), action).outcomes
        (outcome,) = self._source.choices(
            [state for state, _ in outcomes], weights=[probability for _, probability in outcomes]
        )
        print(f"{ActionTextMapper().to_text(action)}")
        print()
        self._world.happened(action, outcome)
        self._performing = ()

    def performing(self) -> tuple[Action, ...]:
        return self._performing


def main(argv: list[str] | None = None) -> None:
    """Plays a game in the terminal: humans pick legal actions by number, OMF strategizes for the players it is given,
    until no action is left. The terminal is the integrator: it runs the game and pushes what happened back to OMF."""
    parser = argparse.ArgumentParser(prog="openmind-play", description="Play a game in the terminal.")
    parser.add_argument("domain", help="game to play, such as tictactoe")
    parser.add_argument(
        "--agent", action="append", default=[], metavar="PLAYER", help="a player OMF plays; repeat for several"
    )
    parser.add_argument(
        "--planner",
        default="minimax",
        choices=sorted(PLANNERS),
        help="the planning model OMF plays with (default: minimax)",
    )
    parser.add_argument(
        "--seconds", type=float, default=DEFAULT_SECONDS, help=f"seconds a move may take (default: {DEFAULT_SECONDS})"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for the outcomes drawn")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING"),
        help="lowest level saved in the game log (default: INFO)",
    )
    parser.add_argument(
        "--log-directory", default="data/log/play", help="where game logs are saved (default: data/log/play)"
    )
    add_knowledge_option(parser)
    add_debug_option(parser)
    arguments = parser.parse_args(argv)
    knowledge_base = create_knowledge_base(arguments.domain.split("/")[0], arguments.knowledge)
    game = create_game(arguments.domain, knowledge_base)
    unknown = [player for player in arguments.agent if player not in game.players().names]
    if unknown:
        parser.error(f"unknown player {', '.join(unknown)}; {game.context} players: {', '.join(game.players().names)}")

    debugger = start_debugging(
        arguments, f"play {game.context}", Path(arguments.log_directory) / game.context, arguments.log_level, knowledge_base
    )
    try:
        _play(
            game,
            knowledge_base,
            frozenset(arguments.agent),
            PLANNERS[arguments.planner](),
            arguments.seconds,
            random.Random(arguments.seed),
        )
    except EOFError:
        print()
        logger.info("Input ended before the game was over")
    finally:
        debugger.stop()


def _play(
    game: RuleBasedGame,
    knowledge_base: object,
    agent_players: frozenset[str],
    planner: object,
    seconds: float,
    source: random.Random,
) -> None:
    state_text = GridTextMapper()
    world = World(game.start())
    referee = Referee(game, world, source)
    agent = create_agent(planner, create_actor(referee))  # type: ignore[arg-type]

    logger.info("Playing %s", game.context)
    while game.actions(world.current()):
        state = world.current()
        print(state_text.to_text(state))
        print()
        player = game.acting_player(state)
        if player in agent_players:
            print(f"{player} plays ", end="")
            strategy = agent.play(knowledge_base, game, world, Guidance(player), Budget(seconds))  # type: ignore[arg-type]
            if strategy is None or world.current() == state:
                print("nothing: OMF has no move here")
                return
        else:
            actions = game.actions(state, player=player)
            for number, candidate in enumerate(actions, start=1):
                print(f"{number}. {ActionTextMapper().to_text(candidate)}")
            referee.dispatch(actions[_read_choice(len(actions)) - 1])
    print(state_text.to_text(world.current()))
    logger.info("No legal action left: game over")


def _read_choice(count: int) -> int:
    while True:
        try:
            number = int(input("Action number: "))
        except ValueError:
            number = 0
        if 1 <= number <= count:
            return number
        print(f"Enter a number from 1 to {count}.")


if __name__ == "__main__":
    main()
