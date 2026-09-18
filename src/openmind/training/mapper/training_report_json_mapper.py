import json

from openmind.evaluation.model.match_results import MatchResults
from openmind.timing.mapper.time_control_text_mapper import TimeControlTextMapper
from openmind.training.model.training_report import TrainingReport
from openmind.training.model.training_round import TrainingRound


class TrainingReportJsonMapper:
    """Maps a training report to JSON text: the settings, then every round's value rules, fits, rows and games."""

    def to_json(self, report: TrainingReport) -> str:
        settings = report.settings
        distillation = settings.distillation
        values = distillation.values
        return json.dumps(
            {
                "domain": report.domain,
                "created_at": report.created_at.isoformat(timespec="seconds"),
                "complete": report.complete,
                "settings": {
                    "rounds": settings.rounds,
                    "start_file": settings.start_file,
                    "games": distillation.games,
                    "held_out_games": distillation.held_out_games,
                    "iterations": distillation.iterations,
                    "seed": distillation.seed,
                    "target": distillation.target,
                    "time_control": None
                    if distillation.time_control is None
                    else TimeControlTextMapper().to_text(distillation.time_control),
                    "expected_steps": distillation.expected_steps,
                    "time_reserve": distillation.time_reserve,
                    "selection": distillation.selection,
                    "puct_exploration": distillation.puct_exploration,
                    "prior": distillation.prior,
                    "prior_temperature": distillation.prior_temperature,
                    "prices": list(values.prices),
                    "max_steps": values.max_steps,
                    "tolerance": values.tolerance,
                    "seconds": values.seconds,
                    "memory_bytes": values.memory_bytes,
                    "candidates": values.candidates,
                    "rollout_actions": settings.rollout_actions,
                    "rollout_limit": settings.rollout_limit,
                    "unfinished_payoff": settings.unfinished_payoff,
                    "evaluation_games": settings.evaluation_games,
                    "deduction": None
                    if settings.deduction is None
                    else {
                        "plies": settings.deduction.plies,
                        "seconds": settings.deduction.seconds,
                        "highest": settings.deduction.highest,
                    },
                },
                "rounds": [self._round(item) for item in report.rounds],
            },
            indent=2,
        )

    def _round(self, item: TrainingRound) -> dict[str, object]:
        return {
            "number": item.number,
            "context": item.context,
            "position_rules": [
                {"name": rule.name, "weight": rule.weight(item.context), "id": rule.id} for rule in item.rules
            ],
            "fits": [
                {
                    "price": fit.price,
                    "terms_kept": fit.terms_kept,
                    "steps": fit.steps,
                    "settled": fit.settled,
                    "training_loss": fit.training_loss,
                    "held_out_loss": fit.held_out_loss,
                }
                for fit in item.fits
            ],
            "chosen_price": None if item.chosen is None else item.chosen.price,
            "training_rows": item.training_rows,
            "held_out_rows": item.held_out_rows,
            "held_out_error": item.held_out_error,
            "baselines": [self._results(results) for results in item.baselines],
            "against_previous": None if item.against_previous is None else self._results(item.against_previous),
            "seconds": item.seconds,
        }

    def _results(self, results: MatchResults) -> dict[str, object]:
        return {
            "opponent": results.opponent,
            "games": results.games,
            "wins": results.wins,
            "draws": results.draws,
            "losses": results.losses,
            "time_control": None if results.time_control is None else TimeControlTextMapper().to_text(results.time_control),
            "wins_on_time": results.wins_on_time,
            "losses_on_time": results.losses_on_time,
        }
