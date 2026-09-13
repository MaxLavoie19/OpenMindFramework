import json

from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_report import EvaluationReport


class ReportJsonMapper:
    """Maps an evaluation report to JSON text."""

    def to_json(self, report: EvaluationReport) -> str:
        settings = report.settings
        rater = report.rater
        return json.dumps(
            {
                "domain": report.domain,
                "created_at": report.created_at.isoformat(timespec="seconds"),
                "rules_file": report.rules_file,
                "settings": {
                    "games": settings.games,
                    "iterations": settings.iterations,
                    "positions": settings.positions,
                    "budgets": list(settings.budgets),
                    "seed": settings.seed,
                },
                "baselines": [
                    {
                        "opponent": results.opponent,
                        "games": results.games,
                        "wins": results.wins,
                        "draws": results.draws,
                        "losses": results.losses,
                    }
                    for results in report.baselines
                ],
                "every_action_optimal": report.every_action_optimal,
                "agreement": [self._agreement(item) for item in report.agreement],
                "unguided_agreement": [self._agreement(item) for item in report.unguided_agreement],
                "rater": None
                if rater is None
                else {
                    "positions": rater.positions,
                    "distinguishing": rater.distinguishing,
                    "optimal": rater.optimal,
                    "mean_regret": rater.mean_regret,
                },
            },
            indent=2,
        )

    def _agreement(self, agreement: Agreement) -> dict[str, int | float]:
        return {
            "iterations": agreement.iterations,
            "positions": agreement.positions,
            "optimal": agreement.optimal,
            "optimal_visit_share": agreement.optimal_visit_share,
            "mean_regret": agreement.mean_regret,
            "seconds_per_choice": agreement.seconds_per_choice,
        }
