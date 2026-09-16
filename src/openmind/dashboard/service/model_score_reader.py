from pathlib import Path

from openmind.agent.constant.agent_constant import DRAW, LOSS, MODEL_KEYWORD, WIN
from openmind.dashboard.model.model_score import ModelScore
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base


class ModelScoreReader:
    """Reads what every model's games came to from a domain's knowledge base, as `GameMemory` remembers them."""

    def scores(self, directory: Path, domain: str) -> tuple[ModelScore, ...]:
        """Every model remembered, the latest to play first; none without a knowledge base."""
        if not (directory / domain).is_dir():
            return ()
        knowledge_base = create_knowledge_base(domain, directory)
        scores: list[ModelScore] = []
        for model in knowledge_base.recall(keyword=MODEL_KEYWORD):
            if len(model.names) != 2:
                continue
            model_id, name = model.names
            outcomes = {outcome: knowledge_base.recall(name=model_id, keyword=outcome) for outcome in (WIN, DRAW, LOSS)}
            times = [record.provenance.when for records in outcomes.values() for record in records if record.provenance.when]
            wins, draws, losses = (len(outcomes[outcome]) for outcome in (WIN, DRAW, LOSS))
            last = max(times).replace(microsecond=0).isoformat(sep=" ") if times else "none"
            scores.append(ModelScore(name, model_id, wins + draws + losses, wins, draws, losses, last))
        return tuple(sorted(scores, key=lambda score: score.last_game, reverse=True))
