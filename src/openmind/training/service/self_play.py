import logging
import random
from collections.abc import Mapping
from dataclasses import replace

from openmind.agent.constant.agent_constant import SELF_PLAY_GAME
from openmind.agent.model.game_summary import GameSummary
from openmind.agent.model.model_description import ModelDescription
from openmind.agent.service.agent import Agent
from openmind.agent.service.game_memory import GameMemory
from openmind.budget.model.budget import Budget
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.model.guidance import Guidance
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.self_play_settings import SelfPlaySettings
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State
from openmind.world.service.world import World

logger = logging.getLogger(__name__)

#: What a game's two seeds are drawn from.
SEEDS = 2**32


class SelfPlay:
    """An agent playing a game against itself, which is how a heuristic deduced from the rules finds out whether it
    holds up.

    OMF simulates games; something always runs them, and here that something is self-play itself: it asks each
    acting player what it would do, takes what they do together, draws one of the outcomes the predictor gives, and
    plays on from there. That is the same referee `openmind-play` is for a terminal, with nobody to prompt.

    Each player is given its own guidance, so a game can be played between two heuristics to find out which is
    worth more: a heuristic playing itself wins half its games whatever it is. Given one guidance, both sides play
    with it and what tells one game from another is the seed.

    Every game is remembered in the knowledge base as it ends — what was played, what it paid, why it ended, and the
    heuristics each side played with — so a game can be looked at afterwards rather than only counted."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def play(
        self,
        knowledge_base: KnowledgeBase,
        game: RuleBasedGame,
        guidance: Guidance | Mapping[str, Guidance],
        settings: SelfPlaySettings,
    ) -> tuple[PlayedGame, ...]:
        """That many games, each from its own seed."""
        memory = GameMemory(knowledge_base)
        played = []
        for number in range(settings.games):
            one = self.play_game(knowledge_base, game, guidance, replace(settings, seed=settings.seed + number))
            self._remembered(memory, game, one)
            played.append(one)
        played = tuple(played)
        decisive = sum(1 for one in played if one.decisive)
        logger.info(
            "Played %d games of %s: %d decisive, %d drawn or cut short, %.0f steps on average",
            len(played),
            game.context,
            decisive,
            len(played) - decisive,
            sum(one.steps for one in played) / max(len(played), 1),
        )
        return played

    def play_game(
        self,
        knowledge_base: KnowledgeBase,
        game: RuleBasedGame,
        guidance: Guidance | Mapping[str, Guidance],
        settings: SelfPlaySettings,
    ) -> PlayedGame:
        """One game, from where it starts to where it stops: every position, what was played in each, and what it
        paid.

        Two streams of chance, drawn from the seed and kept apart: one the players' mixed strategies follow, one the
        game's own outcomes do. Apart, the game can be played again from its actions alone."""
        drawn = random.Random(settings.seed)
        agent_seed, outcome_seed = drawn.randrange(SEEDS), drawn.randrange(SEEDS)
        rng, chance = random.Random(agent_seed), random.Random(outcome_seed)
        world = World(game.start())
        states: list[State] = [world.current()]
        actions: list[JointAction] = []
        while settings.steps is None or len(actions) < settings.steps:
            state = world.current()
            joint = self._chosen(knowledge_base, game, world, guidance, settings, rng)
            if joint is None:
                break
            outcomes = game.joint_outcomes(state, joint).outcomes
            if not outcomes:
                break
            outcome = chance.choices(
                [state for state, _ in outcomes], weights=[probability for _, probability in outcomes]
            )[0]
            world.happened(joint.actions[0][1], outcome)
            actions.append(joint)
            states.append(outcome)
        return PlayedGame(
            tuple(states),
            tuple(actions),
            self._payoffs(game, world.current()),
            game.ended(world.current()),
            agent_seed,
            outcome_seed,
        )

    def _chosen(
        self,
        knowledge_base: KnowledgeBase,
        game: RuleBasedGame,
        world: World,
        guidance: Guidance | Mapping[str, Guidance],
        settings: SelfPlaySettings,
        rng: random.Random,
    ) -> JointAction | None:
        """What every player who can act here does, taken together: each one asked as the agent it is, with the seed
        of this game, so what a mixed strategy calls for is drawn rather than always taken at its likeliest. None
        where nobody can act."""
        state = world.current()
        players = game.players().names
        acting = [players[index] for index, _ in game.joint_actions(state)]
        if not acting:
            return None
        picked: list[tuple[str, object]] = []
        for player in acting:
            strategy = self._agent.play(
                knowledge_base, game, world, self._playing(guidance, player), Budget(settings.seconds)
            )
            distribution = () if strategy is None else strategy.at(state)
            if not distribution:
                continue
            action = rng.choices(
                [action for action, _ in distribution], weights=[chance for _, chance in distribution]
            )[0]
            picked.append((player, action))
        return JointAction(tuple(picked)) if picked else None  # type: ignore[arg-type]

    def _remembered(self, memory: GameMemory, game: RuleBasedGame, played: PlayedGame) -> None:
        """Keeps the game in the knowledge base: what was played, what it paid, and the heuristics it was played with,
        which is what tells a game played before a heuristic was learned from one played after.

        A game that paid nobody isn't kept: what is remembered is finished games, and a game cut short before it
        ended has no result to remember it by."""
        if len(played.payoffs) != len(game.players().names):
            return
        actions = tuple(joint.actions[0][1] for joint in played.actions if len(joint.actions) == 1)
        alone = len(actions) == len(played.actions)
        model = ModelDescription(SELF_PLAY_GAME, game.describe())
        memory.remember(
            GameSummary(
                game.context,
                SELF_PLAY_GAME,
                None,
                memory.last_number(SELF_PLAY_GAME) + 1,
                (played.agent_seed or 0, played.outcome_seed or 0),
                game.players().names,
                (model,) * len(game.players().names),
                played.payoffs,
                played.steps,
                played.ending,
                game.record(actions, played.payoffs) if alone else None,
                actions=actions if alone else (),
            )
        )

    def _playing(self, guidance: Guidance | Mapping[str, Guidance], player: str) -> Guidance:
        """What that player plays with: its own guidance where each side was given one, and the one guidance
        otherwise. Either way it is named for the player, since a planner plans for somebody."""
        held = guidance.get(player) if isinstance(guidance, Mapping) else guidance
        if held is None:
            raise ValueError(f"No guidance for {player}: every player who can act needs one")
        return replace(held, player=player)

    def _payoffs(self, game: RuleBasedGame, state: State) -> tuple[float, ...]:
        """What the game paid each player where it is over, and nothing where it paid nobody."""
        payoff = game.players().payoff
        held = state.model(payoff) if state.has(payoff) else None
        if held is None or not hasattr(held, "get"):
            return ()
        paid = [held.get(player) for player in game.players().names]
        if any(isinstance(value, bool) or not isinstance(value, int | float) for value in paid):
            return ()
        return tuple(float(value) for value in paid)  # type: ignore[arg-type]
