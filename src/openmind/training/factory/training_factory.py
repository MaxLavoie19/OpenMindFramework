from openmind.training.builder.distiller_builder import DistillerBuilder
from openmind.training.builder.rule_selector_builder import RuleSelectorBuilder
from openmind.training.service.distiller import Distiller
from openmind.training.service.rule_selector import RuleSelector


def create_distiller(workers: int = 1) -> Distiller:
    """A distiller with its self-play and rule generator, running games and condition checks in that many worker
    processes, and the rater services it measures rules with."""
    return DistillerBuilder().with_workers(workers).build()


def create_rule_selector(workers: int = 1) -> RuleSelector:
    """A rule selector searching in that many worker processes."""
    return RuleSelectorBuilder().with_workers(workers).build()
