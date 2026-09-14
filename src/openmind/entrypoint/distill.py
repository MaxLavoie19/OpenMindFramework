import argparse
import logging
from datetime import datetime
from pathlib import Path

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.factory.domain_factory import create_domain
from openmind.parallel.constant.parallel_constant import DEFAULT_WORKERS
from openmind.rbs.constant.generation_constant import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_CONFIDENCE,
    DEFAULT_FALSE_DISCOVERY_RATE,
    DEFAULT_MAX_CONDITIONS,
    DEFAULT_MAX_OFFSET,
    DEFAULT_MIN_GAIN,
    DEFAULT_MIN_RULE_VISITS,
    DEFAULT_MIN_VISITS,
    DEFAULT_PATTERNS,
    DEFAULT_PERMUTATIONS,
    DEFAULT_SOLO_LIMIT,
)
from openmind.rbs.mapper.hypothesis_text_mapper import HypothesisTextMapper
from openmind.rbs.mapper.rule_base_json_mapper import RuleBaseJsonMapper
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.rbs.model.generation_settings import GenerationSettings
from openmind.rbs.repository.rule_base_repository import RuleBaseRepository
from openmind.training.constant.training_constant import (
    DEFAULT_GAMES,
    DEFAULT_HELD_OUT_GAMES,
    DEFAULT_ITERATIONS,
    DEFAULT_SEED,
)
from openmind.training.factory.training_factory import create_distiller
from openmind.training.model.distillation_settings import DistillationSettings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Generates rules from self-play, validates them on held-out games, prints them with their measures, and saves the
    rule base."""
    parser = argparse.ArgumentParser(prog="openmind-distill", description="Generate and validate rules from self-play.")
    parser.add_argument("domain", help="domain to distill, such as tictactoe")
    options: list[tuple[str, type, object, str]] = [
        ("--games", int, DEFAULT_GAMES, "self-play games to discover rules in"),
        ("--held-out-games", int, DEFAULT_HELD_OUT_GAMES, "self-play games to validate and measure rules on"),
        ("--iterations", int, DEFAULT_ITERATIONS, "MCTS iterations per self-play move"),
        ("--seed", int, DEFAULT_SEED, "random seed"),
        ("--min-visits", int, DEFAULT_MIN_VISITS, "visits an action needs in a state to count"),
        ("--max-conditions", int, DEFAULT_MAX_CONDITIONS, "conditions per rule at most"),
        ("--min-rule-visits", int, DEFAULT_MIN_RULE_VISITS, "visits the actions a rule matches need"),
        ("--min-gain", float, DEFAULT_MIN_GAIN, "difference in advantage a hypothesis needs in discovery"),
        ("--confidence", float, DEFAULT_CONFIDENCE, "confidence of the payoff bound that marks priority rules"),
        ("--beam-width", int, DEFAULT_BEAM_WIDTH, "hypotheses of each size kept and extended"),
        ("--max-offset", int, DEFAULT_MAX_OFFSET, "largest index offset near() reads from the variable an action sets"),
        ("--solo-limit", int, DEFAULT_SOLO_LIMIT, "own moves solo_distance() looks ahead"),
        ("--patterns", int, DEFAULT_PATTERNS, "winning moves probed for goal patterns"),
        ("--false-discovery-rate", float, DEFAULT_FALSE_DISCOVERY_RATE, "false discovery rate hypotheses are kept at"),
        ("--permutations", int, DEFAULT_PERMUTATIONS, "permutations of each validation test"),
        ("--workers", int, DEFAULT_WORKERS, "worker processes the self-play games run in"),
    ]
    for flag, kind, default, meaning in options:
        parser.add_argument(flag, type=kind, default=default, help=f"{meaning} (default: {default})")
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
    generation = GenerationSettings(
        arguments.min_visits,
        arguments.max_conditions,
        arguments.min_rule_visits,
        arguments.min_gain,
        arguments.confidence,
        arguments.beam_width,
        arguments.max_offset,
        arguments.solo_limit,
        arguments.patterns,
        arguments.false_discovery_rate,
        arguments.permutations,
    )
    settings = DistillationSettings(
        arguments.games, arguments.held_out_games, arguments.iterations, arguments.seed, generation
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
        logger.info("Running self-play in %d worker processes", arguments.workers)
        result = create_distiller(arguments.workers).distill(domain, AgentBuilder().with_exploration(EXPLORATION), settings)
        repository = RuleBaseRepository(RuleBaseJsonMapper())
        path = repository.save(result.rule_base, Path(arguments.rules_directory), datetime.now())
        rule_text, hypothesis_text = RuleTextMapper(), HypothesisTextMapper()
        for rule in result.rule_base.rules:
            print(rule_text.to_text(rule))
        print(f"Rules: {len(result.rule_base.rules)}, conditions per rule on average: {result.mean_conditions}")
        validated = [test for test in result.hypotheses if test.validated]
        print(
            f"Goal patterns: {len(result.patterns)}; hypotheses: {len(result.hypotheses)} tested, {len(validated)} "
            f"validated at a false discovery rate of {generation.false_discovery_rate}; {len(result.covered)} covered by "
            "a simpler rule"
        )
        for test in validated:
            print(f"  {hypothesis_text.to_text(test)}")
        for coverage in result.covered:
            print(f"  {rule_text.to_text(coverage.rule)}: covered by {rule_text.to_text(coverage.covering)}")
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
