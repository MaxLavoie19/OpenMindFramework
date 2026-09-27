from openmind.rule.mapper.clause_json_mapper import ClauseJsonMapper
from openmind.statement.model.consequence import Consequence
from openmind.statement.model.drawn import Drawn


class ConsequenceJsonMapper:
    """Maps what an action does to JSON-ready data and back.

    **What the predictor learns has never outlived the process that learned it.** Rules are kept, facts are kept,
    models are kept; a consequence — this action removes something, and the square it removes from is the row of
    where it started and the column of where it lands — was computed, used for the rest of the run, and dropped.
    So every run began by working out again what a move does, and nothing could be built from what an earlier run
    had found.

    Said as what it is rather than as the text of it, for the reason `ClauseJsonMapper` gives: a rule that has to
    be parsed before it can be used is a rule that will one day fail to parse.

    Its conditions are clauses and go through the clause mapper, because they are the same kind of thing as the
    conditions under which an action is refused — learned by the same machinery, and written down the same way.

    **And so are its drawings, now that a drawing is a term.** This kept its own map from a kind's name to its
    class, and read a drawing's fields off the dataclass, beside a mapper that already knew how to write down
    any term — and a third reading of the same thing, to put a record's value through the value mapper. Several
    ways of writing one thing is how the two halves of this system came to disagree about what a capture is. A
    drawing now goes through `term_to_data` like every other term, and a way of drawing added later is written
    down without this being told about it."""

    def __init__(self, clause_json_mapper: ClauseJsonMapper | None = None) -> None:
        self._clauses = ClauseJsonMapper() if clause_json_mapper is None else clause_json_mapper

    def to_data(self, consequence: Consequence) -> dict[str, object]:
        return {
            "change": consequence.change,
            "action": consequence.action,
            "model": consequence.model,
            "where": [self.drawn_to_data(one) for one in consequence.where],
            "onto": [self.drawn_to_data(one) for one in consequence.onto],
            "value": None if consequence.value is None else self.drawn_to_data(consequence.value),
            "when": [self._clauses.clause_to_data(one) for one in consequence.when],
            "order": consequence.order,
            "settled": consequence.settled,
        }

    def from_data(self, data: dict[str, object]) -> Consequence:
        return Consequence(
            str(data["change"]),
            str(data["action"]),
            str(data["model"]),
            tuple(self.drawn_from_data(one) for one in data.get("where") or ()),  # type: ignore[union-attr]
            tuple(self.drawn_from_data(one) for one in data.get("onto") or ()),  # type: ignore[union-attr]
            None if data.get("value") is None else self.drawn_from_data(data["value"]),  # type: ignore[arg-type]
            tuple(self._clauses.clause_from_data(one) for one in data.get("when") or ()),  # type: ignore[union-attr]
            int(data.get("order", 0)),  # type: ignore[arg-type]
            bool(data.get("settled", True)),
        )

    def drawn_to_data(self, drawn: Drawn) -> dict[str, object]:
        """One way of drawing a part from the action, written down as the term it is."""
        return self._clauses.term_to_data(drawn)

    def drawn_from_data(self, data: dict[str, object]) -> Drawn:
        """That drawing again."""
        return self._clauses.term_to_term(data)
