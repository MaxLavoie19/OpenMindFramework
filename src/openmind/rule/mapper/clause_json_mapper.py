from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Functor, Number, Term, Variable


class ClauseJsonMapper:
    """Maps a clause to JSON-ready data and back.

    A clause is stored as what it is rather than as the text of it. Text would have to be parsed to be reasoned
    with again, and a rule that has to be parsed before it can be used is a rule that will one day fail to parse;
    this way the thing written down and the thing reasoned with are the same shape.

    Each term says which of the four it is by the key it carries, since a constant holding the name `"variable"` and
    a variable called the same thing must not come back as each other."""

    def clause_to_data(self, clause: Clause) -> dict[str, object]:
        return {
            "literals": [self.literal_to_data(one) for one in clause.literals],
            "probability": clause.probability,
            "name": clause.name,
        }

    def clause_from_data(self, data: dict[str, object]) -> Clause:
        literals = data.get("literals") or ()
        return Clause(
            tuple(self.literal_from_data(one) for one in literals),  # type: ignore[arg-type]
            float(data.get("probability", 1.0)),  # type: ignore[arg-type]
            str(data.get("name", "")),
        )

    def literal_to_data(self, literal: Literal) -> dict[str, object]:
        return {
            "predicate": literal.predicate,
            "arguments": [self.term_to_data(one) for one in literal.arguments],
            "negated": literal.negated,
        }

    def literal_from_data(self, data: dict[str, object]) -> Literal:
        arguments = data.get("arguments") or ()
        return Literal(
            str(data["predicate"]),
            tuple(self.term_to_term(one) for one in arguments),  # type: ignore[arg-type]
            bool(data.get("negated", False)),
        )

    def term_to_data(self, term: Term) -> dict[str, object]:
        if isinstance(term, Variable):
            return {"variable": term.name, "sort": term.sort}
        if isinstance(term, Number):
            return {"number": term.value}
        if isinstance(term, Functor):
            return {"functor": term.name, "arguments": [self.term_to_data(one) for one in term.arguments]}
        return {"constant": term.name}

    def term_to_term(self, data: dict[str, object]) -> Term:
        if "variable" in data:
            return Variable(str(data["variable"]), str(data.get("sort", "")))
        if "number" in data:
            return Number(data["number"])  # type: ignore[arg-type]
        if "functor" in data:
            arguments = data.get("arguments") or ()
            return Functor(str(data["functor"]), tuple(self.term_to_term(one) for one in arguments))  # type: ignore[arg-type]
        if "constant" in data:
            return Constant(data["constant"])  # type: ignore[arg-type]
        raise ValueError(f"A term has to say which kind it is; this one carries {sorted(data)}")
