import json

from openmind.agent.model.game_summary import GameSummary
from openmind.agent.model.model_description import ModelDescription
from openmind.timing.mapper.time_control_text_mapper import TimeControlTextMapper
from openmind.timing.model.clock import Clock
from openmind.world.model.action import Action


class GameSummaryJsonMapper:
    """A game summary as JSON text and back. Each model is kept by name and id, its text being remembered once on its
    own; reading a summary back takes the models from the given descriptions, by id, and a model not among them raises
    ValueError. A time control is written as chess writes one."""

    def __init__(self, time_control_text_mapper: TimeControlTextMapper | None = None) -> None:
        self._time_controls = TimeControlTextMapper() if time_control_text_mapper is None else time_control_text_mapper

    def to_json(self, summary: GameSummary) -> str:
        return json.dumps(
            {
                "domain": summary.domain,
                "kind": summary.kind,
                "round": summary.round,
                "number": summary.number,
                "seeds": list(summary.seeds),
                "players": [
                    {"player": player, "model": model.name, "model_id": model.id, "payoff": payoff}
                    for player, model, payoff in zip(summary.players, summary.models, summary.payoffs, strict=True)
                ],
                "plies": summary.plies,
                "ending": summary.ending,
                "record": summary.record,
                "time_control": None if summary.time_control is None else self._time_controls.to_text(summary.time_control),
                "seconds": list(summary.seconds),
                "budgets": list(summary.budgets),
                "clocks": [
                    {"remaining": clock.remaining, "increment": clock.increment, "flagged": clock.flagged}
                    for clock in summary.clocks
                ],
                "flagged": summary.flagged,
                "actions": [{"name": action.name, "parameters": [list(pair) for pair in action.parameters]} for action in summary.actions],
            }
        )

    @staticmethod
    def players(text: str) -> list[dict[str, object]]:
        """A summary's players, each with its player, model, model id and payoff, without reading the rest."""
        return list(json.loads(text)["players"])

    def from_json(self, text: str, models: tuple[ModelDescription, ...]) -> GameSummary:
        data = json.loads(text)
        known = {model.id: model for model in models}
        players = data["players"]
        if missing := [item["model_id"] for item in players if item["model_id"] not in known]:
            raise ValueError(f"The game's model {missing[0]} isn't among the models given")
        control = data["time_control"]
        return GameSummary(
            data["domain"],
            data["kind"],
            data["round"],
            data["number"],
            tuple(data["seeds"]),
            tuple(item["player"] for item in players),
            tuple(known[item["model_id"]] for item in players),
            tuple(float(item["payoff"]) for item in players),
            data["plies"],
            data["ending"],
            data["record"],
            None if control is None else self._time_controls.from_text(control),
            tuple(data["seconds"]),
            tuple(data["budgets"]),
            tuple(Clock(item["remaining"], item["increment"], item["flagged"]) for item in data["clocks"]),
            data["flagged"],
            tuple(
                Action(item["name"], tuple((name, value) for name, value in item["parameters"]))
                for item in data.get("actions", ())
            ),
        )

