"""Registry of chunking strategies."""

from __future__ import annotations

from chunktuner.models import ChunkingStrategy


class StrategyRegistry:
    def __init__(self) -> None:
        self._by_name: dict[str, ChunkingStrategy] = {}

    def register(self, strategy: ChunkingStrategy) -> None:
        self._by_name[strategy.name] = strategy

    def get(self, name: str) -> ChunkingStrategy:
        if name not in self._by_name:
            raise KeyError(f"Unknown strategy: {name}")
        return self._by_name[name]

    def list(self, content_type: str | None = None) -> list[ChunkingStrategy]:
        strategies = list(self._by_name.values())
        if content_type is None:
            return strategies
        return [s for s in strategies if content_type in s.supported_content_types]

    def names(self, content_type: str | None = None) -> list[str]:
        return [s.name for s in self.list(content_type)]
