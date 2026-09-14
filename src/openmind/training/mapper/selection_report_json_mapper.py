import json

from openmind.evaluation.model.agreement import Agreement
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.training.model.selection_report import SelectionReport


class SelectionReportJsonMapper:
    """Maps a selection report to JSON text, rules as readable text."""

    def __init__(self, rule_text_mapper: RuleTextMapper) -> None:
        self._rule_text_mapper = rule_text_mapper

    def to_json(self, report: SelectionReport) -> str:
        settings, confirmation = report.settings, report.confirmation
        return json.dumps(
            {
                "domain": report.domain,
                "created_at": report.created_at.isoformat(),
                "candidates_file": report.candidates_file,
                "settings": {
                    "positions": settings.positions,
                    "reference_iterations": settings.reference_iterations,
                    "iterations": settings.iterations,
                    "guided_rollouts": settings.guided_rollouts,
                    "margin": settings.margin,
                    "confidence": settings.confidence,
                    "resamples": settings.resamples,
                    "seed": settings.seed,
                    "max_hours": settings.max_hours,
                },
                "candidates": report.candidates,
                "complete": report.complete,
                "selected": [self._rule_text_mapper.to_text(rule) for rule in report.selected.rules],
                "full": self._agreement(report.full),
                "subset": self._agreement(report.subset),
                "confirmation": None
                if confirmation is None
                else {
                    "positions": confirmation.positions,
                    "regret_difference": confirmation.regret_difference,
                    "upper_bound": confirmation.upper_bound,
                    "margin": confirmation.margin,
                    "holds": confirmation.holds,
                },
                "tests": [
                    {
                        "rule": self._rule_text_mapper.to_text(test.rule),
                        "pass": test.pass_number,
                        "regret_difference": test.regret_difference,
                        "upper_bound": test.upper_bound,
                        "optimal_difference": test.optimal_difference,
                        "removed": test.removed,
                        "free": test.free,
                        "seconds": test.seconds,
                    }
                    for test in report.tests
                ],
            },
            indent=2,
        )

    def _agreement(self, agreement: Agreement) -> dict[str, object]:
        return {
            "iterations": agreement.iterations,
            "positions": agreement.positions,
            "optimal": agreement.optimal,
            "optimal_visit_share": agreement.optimal_visit_share,
            "mean_regret": agreement.mean_regret,
            "seconds_per_choice": agreement.seconds_per_choice,
        }
