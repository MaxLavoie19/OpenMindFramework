import random

from openmind.agent.model.game_summary import GameSummary
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State


class GameReplayer:
    """Plays a remembered game again from its actions and its outcome seed, drawing each chance as the game drew it,
    so its positions come back exactly as they were.

    A game is remembered by what was played in it, not by every position it went through: a chess game is a hundred
    positions of 64 squares, and its moves are a hundred short strings. What a page needs to show it is worked out
    again here.

    Only a game where one player acted at a time can be replayed from its actions: where several acted at once, the
    summary doesn't say who did what, and the page shows what the game paid without its positions."""

    def positions(self, game: RuleBasedGame, summary: GameSummary) -> tuple[State, ...]:
        """Every position of the game, from where it started to where its last action led: one more than its actions.

        **The seed is asked for where a chance is actually drawn, and not before.** An action with one outcome
        leads where it leads, and picking from a list of one gives that one whatever the generator says — so a
        game whose every action is settled replays exactly without any seed at all. Demanded up front, the
        requirement turned every game of a deterministic game into one that could not be shown: two hundred and
        sixty chess games, none of which draws a chance anywhere.

        Where an action really does have several outcomes and no seed was kept, this raises: there the seed is
        the only record of which way the game went, and guessing would be showing positions the game never
        reached."""
        chance = random.Random(summary.seeds[1]) if len(summary.seeds) > 1 else None
        state = game.start()
        states = [state]
        for action in summary.actions:
            acting = game.joint_actions(state)
            if len(acting) != 1:
                break
            player = game.players().names[acting[0][0]]
            outcomes = game.joint_outcomes(state, JointAction(((player, action),))).outcomes
            if not outcomes:
                break
            if len(outcomes) == 1:
                state = outcomes[0][0]
            elif chance is None:
                raise ValueError(
                    f"{summary.label} wasn't remembered with an outcome seed, and an action of it had "
                    f"{len(outcomes)} outcomes: it can't be played again"
                )
            else:
                state = chance.choices(
                    [outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes]
                )[0]
            states.append(state)
        return tuple(states)
