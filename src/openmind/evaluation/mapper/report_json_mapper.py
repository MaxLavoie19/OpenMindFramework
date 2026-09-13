import json

from openmind.evaluation.model.evaluation_report import EvaluationReport


class ReportJsonMapper:
    """Maps an evaluation report to JSON text."""

    def to_json(self, report: EvaluationReport) -> str:
        settings = report.settings
        return json.dumps(
            {
                "domain": report.domain,
                "created_at": report.created_at.isoformat(timespec="seconds"),
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
                "agreement": [
                    {"iterations": item.iterations, "positions": item.positions, "optimal": item.optimal}
                    for item in report.agreement
                ],
            },
            indent=2,
        )
