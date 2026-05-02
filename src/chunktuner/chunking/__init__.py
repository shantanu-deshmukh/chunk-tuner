from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.chunking.fixed_tokens import FixedTokenStrategy
from chunktuner.chunking.recursive_character import RecursiveCharacterStrategy
from chunktuner.chunking.registry import StrategyRegistry


def build_default_registry(encoding: str = "cl100k_base") -> StrategyRegistry:
    """Alias for ``build_full_registry`` (backward compatible name)."""
    return build_full_registry(encoding)


default_registry = build_full_registry()

__all__ = [
    "StrategyRegistry",
    "default_registry",
    "build_default_registry",
    "build_full_registry",
    "FixedTokenStrategy",
    "RecursiveCharacterStrategy",
]
