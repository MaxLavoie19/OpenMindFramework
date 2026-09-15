import json

from openmind.evaluation.model.match_results import MatchResults
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
                    "ponder_positions": 0 if distillation.pondering is None else distillation.pondering.positions,
                },
                "rounds": [self._round(item) for item in report.rounds],
            },
            indent=2,
        )

    def _round(self, item: TrainingRound) -> dict[str, object]:
        base = item.value_base
        return {
            "number": item.number,
            "value_base": {
                "bias": base.bias,
                "low": base.low,
                "high": base.high,
                "rules": [{"term": rule.term.source, "weight": rule.weight} for rule in base.rules],
            },
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
            "pondering": None
            if item.pondering is None
            else {
                "positions": item.pondering.positions,
                "proven": item.pondering.proven,
                "seeds": item.pondering.seeds,
                "seeds_kept": item.pondering.seeds_kept,
                "seeds_in_rules": item.pondering.seeds_in_rules,
            },
        }

    def _results(self, results: MatchResults) -> dict[str, object]:
        return {
            "opponent": results.opponent,
            "games": results.games,
            "wins": results.wins,
            "draws": results.draws,
            "losses": results.losses,
        }
