from openmind.agent.model.domain import Domain
from openmind.agent.model.game_summary import GameSummary
from openmind.agent.model.model_description import ModelDescription
from openmind.evaluation.model.match_game import MatchGame
from openmind.timing.model.time_control import TimeControl


class MatchGameSummaryMapper:
    """A match's game as it is remembered: the evaluated model in its seat and the opponent's in the other, in the order
    of the players' names. A policy gives no budget, so a match's steps have none."""

    def to_summary(
        self,
        domain: Domain,
        game: MatchGame,
        kind: str,
        round_number: int | None,
        number: int,
        seat: int,
        evaluated: ModelDescription,
        opponent: ModelDescription,
        record: str | None,
        time_control: TimeControl | None,
    ) -> GameSummary:
        models = (evaluated, opponent) if seat == 0 else (opponent, evaluated)
        return GameSummary(
            domain.name,
            kind,
            round_number,
            number,
            (game.policy_seed, game.outcome_seed),
            domain.players.names,
            models,
            game.payoffs,
            game.plies,
            game.ending,
            record,
            time_control,
            game.seconds,
            (None,) * len(game.seconds),
            game.clocks,
            game.flagged,
        )
