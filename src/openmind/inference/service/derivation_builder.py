from openmind.inference.model.derivation import Derivation
from openmind.inference.model.derivation_step import GIVEN, DerivationStep
from openmind.inference.model.substitution import Substitution
from openmind.statement.model.clause import Clause


class DerivationBuilder:
    """Builds the record of how a conclusion was reached, out of the records of what it came from.

    Both ways of reasoning need this and neither should own it. Chaining forward and driving at a goal put clauses
    together by the same step, and a conclusion is worth no less for having been reached one way rather than the
    other — so the account of it is made the same way for both.

    The fiddly part it exists to get right: each parent numbers its own steps from one, so laying two of them end
    to end would leave several steps called 3 and premises pointing at whichever came first. Each parent's numbers
    are mapped to the new ones as its steps are taken, and its premises are remapped with them. A step both
    parents share is kept once, since a thing used twice was still only established once."""

    def given(self, clause: Clause) -> Derivation:
        """A clause taken as it stands, resting on itself."""
        return Derivation(clause, (DerivationStep(1, GIVEN, (), clause),), self.chances(clause, ()))

    def joined(
        self,
        made: Clause,
        rule: str,
        agreed: Substitution,
        *parents: Derivation,
    ) -> Derivation:
        """One derivation of that conclusion, out of the ones it was reached from."""
        steps: list[DerivationStep] = []
        seen: dict[str, int] = {}
        lasts: list[int] = []
        for parent in parents:
            renumbered: dict[int, int] = {}
            for step in parent.steps:
                key = f"{step.rule}|{step.clause.readable}"
                held = seen.get(key)
                if held is not None:
                    renumbered[step.number] = held
                    continue
                number = len(steps) + 1
                seen[key] = number
                renumbered[step.number] = number
                steps.append(
                    DerivationStep(
                        number,
                        step.rule,
                        tuple(renumbered.get(one, one) for one in step.premises),
                        step.clause,
                        step.substitution,
                    )
                )
            if parent.steps:
                lasts.append(renumbered[parent.steps[-1].number])
        steps.append(DerivationStep(len(steps) + 1, rule, tuple(lasts), made, agreed))
        chances = self.chances(made, tuple(one for parent in parents for one in parent.chances))
        return Derivation(made, tuple(steps), chances)

    def chances(self, clause: Clause, held: tuple[str, ...]) -> tuple[str, ...]:
        """The doubtful clauses a conclusion leans on, each named once.

        Named once and not counted, because what matters later is *which* ones it leans on. Two reasons for the
        same conclusion are two reasons only where they lean on different ones; leaning twice on the same doubtful
        clause is leaning on it once."""
        found = dict.fromkeys(held)
        if not clause.certain and clause.name:
            found.setdefault(clause.name)
        return tuple(found)
