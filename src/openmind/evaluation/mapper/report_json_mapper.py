import json

from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_report import EvaluationReport
from openmind.timing.mapper.time_control_text_mapper import TimeControlTextMapper


class ReportJsonMapper:
    """Maps an evaluation report to JSON text."""

    def to_json(self, report: EvaluationReport) -> str:
        settings = report.settings
        rater, valuer = report.rater, report.valuer
        return json.dumps(
            {
                "domain": report.domain,
                "created_at": report.created_at.isoformat(timespec="seconds"),
                "rules_file": report.rules_file,
                "values_file": report.values_file,
                "settings": {
                    "games": settings.games,
                    "iterations": settings.iterations,
                    "positions": settings.positions,
                    "budgets": list(settings.budgets),
                    "seed": settings.seed,
                    "reference_iterations": settings.reference_iterations,
                    "guided_rollouts": settings.guided_rollouts,
                    "rollout_actions": settings.rollout_actions,
                    "rollout_limit": settings.rollout_limit,
                    "unfinished_payoff": settings.unfinished_payoff,
                    "time_control": None if settings.time_control is None else TimeControlTextMapper().to_text(settings.time_control),
                    "expected_steps": settings.expected_steps,
                    "time_reserve": settings.time_reserve,
                    "selection": settings.selection,
                    "puct_exploration": settings.puct_exploration,
                    "prior": settings.prior,
                    "prior_temperature": settings.prior_temperature,
                },
                "baselines": [
                    {
                        "opponent": results.opponent,
                        "games": results.games,
                        "wins": results.wins,
                        "draws": results.draws,
                        "losses": results.losses,
                        "time_control": None
                        if results.time_control is None
                        else TimeControlTextMapper().to_text(results.time_control),
                        "wins_on_time": results.wins_on_time,
                        "losses_on_time": results.losses_on_time,
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
                "valuer": None
                if valuer is None
                else {
                    "positions": valuer.positions,
                    "valued": valuer.valued,
                    "mean_absolute_error": valuer.mean_absolute_error,
                    "optimal": valuer.optimal,
                    "mean_regret": valuer.mean_regret,
                },
                "guidance_tests": [
                    {
                        "iterations": test.iterations,
                        "positions": test.positions,
                        "low_value_share_difference": test.low_value_share_difference,
                        "low_value_share_p": test.low_value_share_p,
                        "regret_difference": test.regret_difference,
                        "regret_p": test.regret_p,
                        "optimal_only_guided": test.optimal_only_guided,
                        "optimal_only_unguided": test.optimal_only_unguided,
                        "optimal_choice_p": test.optimal_choice_p,
                    }
                    for test in report.guidance_tests
                ],
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
