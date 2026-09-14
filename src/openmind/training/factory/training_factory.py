from openmind.training.builder.distiller_builder import DistillerBuilder
from openmind.training.builder.rule_selector_builder import RuleSelectorBuilder
from openmind.training.builder.value_distiller_builder import ValueDistillerBuilder
from openmind.training.service.distiller import Distiller
from openmind.training.service.rule_selector import RuleSelector
from openmind.training.service.value_distiller import ValueDistiller


def create_distiller(workers: int = 1) -> Distiller:
    """A distiller with its self-play and rule generator, running games and condition checks in that many worker
    processes, and the rater services it measures rules with."""
    return DistillerBuilder().with_workers(workers).build()


def create_rule_selector(workers: int = 1) -> RuleSelector:
    """A rule selector searching in that many worker processes."""
    return RuleSelectorBuilder().with_workers(workers).build()


def create_value_distiller(workers: int = 1) -> ValueDistiller:
    """A value distiller with its self-play and value generator, running games and term evaluations in that many worker
    processes, and the valuer services it measures value rules with."""
    return ValueDistillerBuilder().with_workers(workers).build()
