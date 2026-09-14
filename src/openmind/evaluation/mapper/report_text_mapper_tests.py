from dataclasses import replace
from datetime import datetime

from openmind.evaluation.mapper.report_text_mapper import ReportTextMapper
from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_report import EvaluationReport
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.model.guidance_test import GuidanceTest
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.model.rater_agreement import RaterAgreement

BASELINES = [
    "Against random: 2 games, 2 wins, 0 draws, 0 losses",
    "Against untrained MCTS: 2 games, 1 wins, 0 draws, 1 losses",
]


def report(
    agreement: tuple[Agreement, ...] = (),
    unguided_agreement: tuple[Agreement, ...] = (),
    rater: RaterAgreement | None = None,
    rules_file: str | None = None,
) -> EvaluationReport:
    return EvaluationReport(
        "tictactoe",
        datetime(2026, 9, 13, 19, 0, 0),
        rules_file,
        EvaluationSettings(games=2, iterations=10, positions=100, budgets=(10, 500), seed=1),
        (MatchResults("random", 2, 2, 0, 0), MatchResults("untrained MCTS", 2, 1, 0, 1)),
        12,
        agreement,
        unguided_agreement,
        rater,
    )


def test_guided_and_unguided_agreement_sit_side_by_side_followed_by_the_rules_alone() -> None:
    text = ReportTextMapper().to_text(
        report(
            (Agreement(10, 100, 89, 0.6123, 0.02, 0.003792), Agreement(500, 100, 99, 0.9, 0.004, 0.02734)),
            (Agreement(10, 100, 86, 0.55, 0.031, 0.003277), Agreement(500, 100, 99, 0.88, 0.006, 0.02361)),
            RaterAgreement(100, 38, 52.5, 0.12),
            "data/rbs/tictactoe/rules.json",
        )
    )

    assert text.splitlines() == [
        *BASELINES,
        "Guided by data/rbs/tictactoe/rules.json against unguided, on the same 100 positions (every action optimal in 12):",
        "iterations  optimal  visits on optimal    mean regret   seconds per choice",
        "        10  89 / 86      0.612 / 0.550  0.020 / 0.031  0.003792 / 0.003277",
        "       500  99 / 99      0.900 / 0.880  0.004 / 0.006  0.027340 / 0.023610",
        "Rules alone: ratings separate actions in 38 of 100 positions; a top-rated action is optimal in 52.5 of 100; "
        "mean regret 0.120",
    ]


def test_an_unguided_evaluation_has_a_single_agreement_table() -> None:
    text = ReportTextMapper().to_text(report((Agreement(10, 100, 86, 0.55, 0.031, 0.003277),)))

    assert text.splitlines() == [
        *BASELINES,
        "Agreement with perfect play on 100 positions (every action optimal in 12):",
        "iterations  optimal  visits on optimal  mean regret  seconds per choice",
        "        10       86              0.550        0.031            0.003277",
    ]


def test_paired_tests_follow_as_a_table_and_a_reference_search_is_named() -> None:
    text = ReportTextMapper().to_text(
        replace(
            report(
                (Agreement(10, 100, 89, 0.6123, 0.02, 0.003792),),
                (Agreement(10, 100, 86, 0.55, 0.031, 0.003277),),
                None,
                "rules.json",
            ),
            settings=EvaluationSettings(games=2, iterations=10, positions=100, budgets=(10,), seed=1, reference_iterations=2000),
            guidance_tests=(GuidanceTest(10, 100, -0.052, 0.0012, -0.004, 0.31, 7, 2, 0.18),),
        )
    )

    assert text.splitlines()[2:] == [
        "Guided by rules.json against unguided, on the same 100 positions "
        "(reference: 2000-iteration unguided searches; every action optimal in 12):",
        "iterations  optimal  visits on optimal    mean regret   seconds per choice",
        "        10  89 / 86      0.612 / 0.550  0.020 / 0.031  0.003792 / 0.003277",
        "Guided against unguided, paired by position (guided minus unguided; Wilcoxon and McNemar p-values):",
        "iterations  low-value visits       p  regret     p  optimal only guided / unguided     p",
        "        10            -0.052  0.0012  -0.004  0.31                           7 / 2  0.18",
    ]


def test_unguided_rollouts_are_named_in_the_heading() -> None:
    text = ReportTextMapper().to_text(
        replace(
            report(
                (Agreement(10, 100, 89, 0.6123, 0.02, 0.003792),),
                (Agreement(10, 100, 86, 0.55, 0.031, 0.003277),),
                None,
                "rules.json",
            ),
            settings=EvaluationSettings(games=2, iterations=10, positions=100, budgets=(10,), seed=1, guided_rollouts=False),
        )
    )

    assert text.splitlines()[2] == (
        "Guided by rules.json with unguided rollouts against unguided, on the same 100 positions "
        "(every action optimal in 12):"
    )


def test_skipped_agreement_is_said() -> None:
    assert ReportTextMapper().to_text(report()).splitlines() == [
        *BASELINES,
        "Agreement with perfect play skipped: no positions",
    ]
