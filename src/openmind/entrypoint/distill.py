import argparse
import logging
from datetime import datetime
from pathlib import Path

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.factory.domain_factory import create_domain
from openmind.expression.mapper.expression_json_mapper import ExpressionJsonMapper
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.rbs.constant.induction_constant import (
    DEFAULT_MAX_CONDITIONS,
    DEFAULT_MIN_GAIN,
    DEFAULT_MIN_RULE_VISITS,
    DEFAULT_MIN_VISITS,
)
from openmind.rbs.mapper.rule_base_json_mapper import RuleBaseJsonMapper
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.rbs.model.induction_settings import InductionSettings
from openmind.rbs.repository.rule_base_repository import RuleBaseRepository
from openmind.training.constant.training_constant import (
    DEFAULT_GAMES,
    DEFAULT_HELD_OUT_GAMES,
    DEFAULT_ITERATIONS,
    DEFAULT_SEED,
)
from openmind.training.factory.training_factory import create_distiller
from openmind.training.model.distillation_settings import DistillationSettings
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Distills rules from self-play, prints them with their measures, and saves the rule base."""
    parser = argparse.ArgumentParser(prog="openmind-distill", description="Distill rules from self-play.")
    parser.add_argument("domain", help="domain to distill, such as tictactoe")
    parser.add_argument(
        "--games", type=int, default=DEFAULT_GAMES, help=f"self-play games to learn from (default: {DEFAULT_GAMES})"
    )
    parser.add_argument(
        "--held-out-games",
        type=int,
        default=DEFAULT_HELD_OUT_GAMES,
        help=f"self-play games to measure the rules on (default: {DEFAULT_HELD_OUT_GAMES})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"MCTS iterations per self-play move (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--min-visits",
        type=int,
        default=DEFAULT_MIN_VISITS,
        help=f"visits a sample needs to count (default: {DEFAULT_MIN_VISITS})",
    )
    parser.add_argument(
        "--max-conditions",
        type=int,
        default=DEFAULT_MAX_CONDITIONS,
        help=f"conditions per rule at most (default: {DEFAULT_MAX_CONDITIONS})",
    )
    parser.add_argument(
        "--min-rule-visits",
        type=int,
        default=DEFAULT_MIN_RULE_VISITS,
        help=f"visits a rule needs (default: {DEFAULT_MIN_RULE_VISITS})",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=DEFAULT_MIN_GAIN,
        help=f"change in expected value a condition needs (default: {DEFAULT_MIN_GAIN})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING"),
        help="lowest level saved in the log (default: INFO)",
    )
    parser.add_argument(
        "--log-directory", default="data/log/distill", help="where logs are saved (default: data/log/distill)"
    )
    parser.add_argument(
        "--rules-directory", default="data/rbs", help="where rule bases are saved (default: data/rbs)"
    )
    arguments = parser.parse_args(argv)
    domain = create_domain(arguments.domain)
    induction = InductionSettings(
        arguments.min_visits, arguments.max_conditions, arguments.min_rule_visits, arguments.min_gain
    )
    settings = DistillationSettings(
        arguments.games, arguments.held_out_games, arguments.iterations, arguments.seed, induction
    )

    directory = Path(arguments.log_directory) / domain.name
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)-5s %(name)s %(message)s"))
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(arguments.log_level)
    try:
        result = create_distiller().distill(domain, AgentBuilder().with_exploration(EXPLORATION), settings)
        repository = RuleBaseRepository(RuleBaseJsonMapper(ExpressionJsonMapper()))
        path = repository.save(result.rule_base, Path(arguments.rules_directory), datetime.now())
        rule_text = RuleTextMapper(ExpressionTextMapper(VariableNameMapper()))
        for rule in result.rule_base.rules:
            print(rule_text.to_text(rule))
        print(f"Rules: {len(result.rule_base.rules)}, conditions per rule on average: {result.mean_conditions}")
        print(
            f"Samples: {result.training_samples} for training, {result.held_out_samples} held out; "
            f"rating error on held-out samples: {result.rating_error}"
        )
        print(f"Saved rules {path}")
        logger.info("Saved rules %s", path)
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()
