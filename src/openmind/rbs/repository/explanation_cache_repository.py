import json
import re
from pathlib import Path


class ExplanationCacheRepository:
    """Keeps a language model's sentences by rule source, as JSON at <directory>/<domain>/<model>.json: a list of
    entries, each a rule's source, its literal reading and the model's sentence, sorted by source, so the pairs can
    later teach a model to translate. A domain or model name's characters other than letters, digits, dots and dashes
    become underscores in the path."""

    def path(self, directory: Path, domain: str, model: str) -> Path:
        return directory / self._safe(domain) / f"{self._safe(model)}.json"

    def load(self, path: Path) -> dict[str, tuple[str, str]]:
        """Every entry by source, as its reading and sentence; nothing when the file doesn't exist."""
        if not path.exists():
            return {}
        return {
            entry["source"]: (entry["reading"], entry["sentence"])
            for entry in json.loads(path.read_text(encoding="utf-8"))
        }

    def save(self, entries: dict[str, tuple[str, str]], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        document = [
            {"source": source, "reading": reading, "sentence": sentence}
            for source, (reading, sentence) in sorted(entries.items())
        ]
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path

    def _safe(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9.-]", "_", name)
