from openmind.agent.model.domain import Domain
from openmind.inference.constant.inference_constant import BEST, LOOK_AHEAD_VARIABLE, ME, OTHER, VIEW, WORST
from openmind.inference.model.deduction import Deduction
from openmind.inference.model.expression import Expression
from openmind.inference.model.pattern import Pattern
from openmind.inference.model.pattern_condition import PatternCondition
from openmind.inference.model.vocabulary import Vocabulary
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.value import Value


class DeductionInducer:
    """Candidate expressions from a proven deduction, scoped to what it touched, for the expression search to try first.
    A player is written relative to the one the deduction is for: `me`, or `other` for the next player.

    - The proof as a look-ahead: whether the player the proof pays the highest payoff gets it, read along the line, each
      ply the best over that player's actions and the worst over the other's. A win in 2 of mine reads "whatever the
      other player does, I can win next": `{view}.best(me, lambda v3: v3.worst(other, lambda v2: v2.best(me, lambda v1:
      v1.payoff[me] == 1.0)))`.
    - What the first action changed, as a pattern: every variable with whole-number indices it changed, read as it was
      before, at its offset from the first one, counted wherever they all hold together.

    A look-ahead or pattern that can't be written this way, such as a proof paying a third player, gives no seed."""

    def __init__(self, expression_generator: ExpressionGenerator, variable_name_mapper: VariableNameMapper) -> None:
        self._expression_generator = expression_generator
        self._variable_name_mapper = variable_name_mapper

    def seeds(
        self, domain: Domain, deduction: Deduction, vocabulary: Vocabulary, highest: float
    ) -> tuple[Expression, ...]:
        """The proof's look-ahead, then the first action's pattern; none for a deduction that proved nothing."""
        if not deduction.proven:
            return ()
        found = (self._proof(domain, deduction, highest), self._pattern(domain, deduction, vocabulary))
        return tuple(expression for expression in found if expression is not None)

    def _proof(self, domain: Domain, deduction: Deduction, highest: float) -> Expression | None:
        payoffs = deduction.payoffs or ()
        winners = [index for index, payoff in enumerate(payoffs) if payoff >= highest]
        if not winners:
            return None
        winner = domain.players.names[winners[0]]
        relative = self._relative(domain, deduction, winner)
        base, texts = self._variable_name_mapper.from_name(domain.players.payoffs[winners[0]])
        if relative is None or texts not in ((), (winner,)):
            return None
        reading = f"{VIEW}.{base}" if not texts else f"{VIEW}.{base}[{relative}]"
        expression: Expression | None = Expression(f"{reading} == {highest!r}", 1, 0)
        befores = (deduction.state, *(state for _, state in deduction.line[:-1]))
        for before in reversed(befores):
            mover = dict(before.variables)[domain.players.to_act]
            mover_relative = self._relative(domain, deduction, mover)
            if expression is None or mover_relative is None:
                return None
            kind = BEST if mover == winner else WORST
            variable = f"{LOOK_AHEAD_VARIABLE}{expression.plies + 1}"
            wanted = f"{VIEW}.{kind}({mover_relative}, lambda {variable}: {expression.template.replace(VIEW, variable)})"
            expression = next(
                (child for child in self._expression_generator.look_aheads(expression) if child.template == wanted), None
            )
        return expression

    def _pattern(self, domain: Domain, deduction: Deduction, vocabulary: Vocabulary) -> Expression | None:
        _, after = deduction.line[0]
        before = dict(deduction.state.variables)
        skipped = {domain.players.to_act, *domain.players.payoffs}
        changed: list[tuple[str, tuple[int, ...], str]] = []
        for name, value in after.variables:
            if name in skipped or name not in before or before[name] == value:
                continue
            base, texts = self._variable_name_mapper.from_name(name)
            rendered = self._rendered(domain, deduction, before[name])
            if not texts or base not in vocabulary.indices_by_base or rendered is None:
                continue
            if all(text.lstrip("-").isdecimal() for text in texts):
                changed.append((base, tuple(int(text) for text in texts), rendered))
        if not changed:
            return None
        anchor, origin, _ = changed[0]
        conditions = tuple(
            PatternCondition(base, tuple(index - start for index, start in zip(indices, origin, strict=True)), "==", rendered)
            for base, indices, rendered in changed
            if len(indices) == len(origin)
        )
        return self._expression_generator.pattern_expression(Pattern(anchor, conditions), vocabulary)

    def _relative(self, domain: Domain, deduction: Deduction, player: Value) -> str | None:
        names = domain.players.names
        if player == deduction.player:
            return ME
        if player == names[(names.index(deduction.player) + 1) % len(names)]:
            return OTHER
        return None

    def _rendered(self, domain: Domain, deduction: Deduction, value: Value) -> str | None:
        if value in domain.players.names:
            return self._relative(domain, deduction, value)
        return repr(value)
