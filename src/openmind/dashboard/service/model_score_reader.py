from pathlib import Path

from openmind.agent.constant.agent_constant import DRAW, LOSS, MODEL_KEYWORD, WIN
from openmind.dashboard.model.model_score import ModelScore
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base


class ModelScoreReader:
    """Reads what every model's games came to from a domain's knowledge base, as `GameMemory` keeps them."""

    def scores(self, directory: Path, domain: str) -> tuple[ModelScore, ...]:
        """Every model remembered, the latest to play first; none without a knowledge base."""
        if not (directory / domain).is_dir():
            return ()
        knowledge_base = create_knowledge_base(domain, directory)
        scores: list[ModelScore] = []
        for model in knowledge_base.beliefs(tags=(("keyword", MODEL_KEYWORD),)):
            tags = dict(model.tags)
            model_id, name = str(tags["id"]), str(tags["name"])
            outcomes = {
                outcome: knowledge_base.beliefs(tags=(("keyword", outcome), ("model", model_id))) for outcome in (WIN, DRAW, LOSS)
            }
            times = [
                evidence.source.at
                for beliefs in outcomes.values()
                for belief in beliefs
                for evidence in belief.evidence
                if evidence.source.at is not None
            ]
            wins, draws, losses = (len(outcomes[outcome]) for outcome in (WIN, DRAW, LOSS))
            last = max(times).replace(microsecond=0).isoformat(sep=" ") if times else "none"
            scores.append(ModelScore(name, model_id, wins + draws + losses, wins, draws, losses, last))
        return tuple(sorted(scores, key=lambda score: score.last_game, reverse=True))
