from openmind.agent.model.game_summary import GameSummary
from openmind.agent.model.model_description import ModelDescription
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.training.model.played_game import PlayedGame


class PlayedGameSummaryMapper:
    """A self-play game as it is remembered: its kind, round and number, the model each player played in the order of
    the players' names, and its record, worked out where the game came back to."""

    def to_summary(
        self,
        rbs: RuleBasedGame,
        game: PlayedGame,
        kind: str,
        round_number: int | None,
        number: int,
        models: tuple[ModelDescription, ...],
        record: str | None,
    ) -> GameSummary:
        seeds = tuple(seed for seed in (game.agent_seed, game.outcome_seed) if seed is not None)
        return GameSummary(
            rbs.context,
            kind,
            round_number,
            number,
            seeds,
            rbs.players().names,
            models,
            game.payoffs,
            len(game.actions),
            game.ending,
            record,
            game.time_control,
            game.seconds,
            game.budgets,
            game.clocks,
            game.flagged,
            game.actions,
        )
