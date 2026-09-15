from openmind.rhetoric.constant.rhetoric_constant import AUDIENCE, IDENTITY
from openmind.rhetoric.model.distance import Distance
from openmind.rhetoric.model.position import Position
from openmind.rhetoric.model.speaker import Speaker


class DistanceMeasurer:
    """Measures the distances a speaker perceives, for every audience member and every question both positions of a
    kind answer:

    - identity: effective ethos minus the ethos projected to the member, problematic by the importance of the question
      to the speaker's effective ethos;
    - audience: projected ethos minus the member's pathos as perceived, problematic by the importance of the question to
      the member.

    A position without importance counts as importance 1 (it matters fully). Distances come in the order of the members
    in the projective ethos, identity before audience, and questions in the order of the first position."""

    def distances(self, speaker: Speaker) -> tuple[Distance, ...]:
        perceived = dict(speaker.projective_pathos)
        measured: list[Distance] = []
        for member, projected in speaker.projective_ethos:
            measured.extend(self._between(member, IDENTITY, speaker.effective_ethos.positions, projected.positions, first_matters=True))
            if member in perceived:
                measured.extend(self._between(member, AUDIENCE, projected.positions, perceived[member].positions, first_matters=False))
        return tuple(measured)

    def _between(
        self,
        member: str,
        kind: str,
        first: tuple[Position, ...],
        second: tuple[Position, ...],
        first_matters: bool,
    ) -> list[Distance]:
        seconds = {position.question: position for position in second}
        distances: list[Distance] = []
        for position in first:
            other = seconds.get(position.question)
            if other is None:
                continue
            value = position.answer - other.answer
            holder = position if first_matters else other
            importance = 1.0 if holder.importance is None else holder.importance
            distances.append(Distance(member, position.question, kind, value, abs(value) * importance))
        return distances
