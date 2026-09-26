from dataclasses import fields

from openmind.rule.model.consequence import Consequence
from openmind.rule.model.drawn import Always, Asked, Column, Drawn, More, Other, Place, Row, Standing, Stepped
from openmind.rule.mapper.clause_json_mapper import ClauseJsonMapper

#: Every way one part of a change can be drawn from the action, by the name it is written down under.
#:
#: Named rather than numbered, so a drawing added later does not renumber the ones already written down, and a
#: file written today still reads after one is added.
DRAWINGS: dict[str, type] = {
    one.__name__: one for one in (Place, Stepped, Row, Column, Standing, Asked, Always, Other, More)
}

#: Which of them a drawing is, as it is written down.
DRAWN = "drawn"


class ConsequenceJsonMapper:
    """Maps what an action does to JSON-ready data and back.

    **What the predictor learns has never outlived the process that learned it.** Rules are kept, facts are kept,
    models are kept; a consequence — this action removes something, and the square it removes from is the row of
    where it started and the column of where it lands — was computed, used for the rest of the run, and dropped.
    So every run began by working out again what a move does, and nothing could be built from what an earlier run
    had found.

    Said as what it is rather than as the text of it, for the reason `ClauseJsonMapper` gives: a rule that has to
    be parsed before it can be used is a rule that will one day fail to parse. A drawing carries the name of its
    kind, since `Always("row")` and `Place("row", …)` must not come back as one another.

    Its conditions are clauses and go through the clause mapper, because they are the same kind of thing as the
    conditions under which an action is refused — learned by the same machinery, and written down the same way."""

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
        """One way of drawing a part from the action, with the name of which way it is.

        Its own fields and no more, read off the kind rather than listed here, so a drawing that gains a field
        is written down with it without this having to be told."""
        return {DRAWN: type(drawn).__name__, **{one.name: getattr(drawn, one.name) for one in fields(drawn)}}

    def drawn_from_data(self, data: dict[str, object]) -> Drawn:
        named = str(data.get(DRAWN, ""))
        kind = DRAWINGS.get(named)
        if kind is None:
            raise ValueError(f"No way of drawing a part is called {named!r}; there are {sorted(DRAWINGS)}")
        return kind(**{one.name: data[one.name] for one in fields(kind) if one.name in data})
