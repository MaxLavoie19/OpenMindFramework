from dataclasses import dataclass

from openmind.rbs.model.value_base import ValueBase
from openmind.training.model.rule_support import RuleSupport
from openmind.training.model.signal_record import SignalRecord


@dataclass(frozen=True, slots=True)
class SignalLibrary:
    """What a domain's training learned about signals, kept across rounds and runs: every signal's record, the rules the
    signals support, and by signal name the value base last fitted to each followed signal's targets, which an agent
    following that signal values positions with."""

    domain: str
    records: tuple[SignalRecord, ...] = ()
    supports: tuple[RuleSupport, ...] = ()
    value_bases: tuple[tuple[str, ValueBase], ...] = ()
