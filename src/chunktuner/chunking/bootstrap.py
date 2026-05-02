"""Construct a ``StrategyRegistry`` with all built-in strategies (best-effort optional deps)."""

from __future__ import annotations

from chunktuner.chunking.fixed_tokens import FixedTokenStrategy
from chunktuner.chunking.registry import StrategyRegistry
from chunktuner.chunking.recursive_character import RecursiveCharacterStrategy


def build_full_registry(encoding: str = "cl100k_base") -> StrategyRegistry:
    reg = StrategyRegistry()
    reg.register(FixedTokenStrategy(encoding_name=encoding))
    reg.register(RecursiveCharacterStrategy(encoding_name=encoding))
    try:
        from chunktuner.chunking.semantic import SemanticStrategy

        reg.register(SemanticStrategy(encoding_name=encoding))
    except ImportError:
        pass
    try:
        from chunktuner.chunking.markdown_semantic import MarkdownSemanticStrategy

        reg.register(MarkdownSemanticStrategy(encoding_name=encoding))
    except ImportError:
        pass
    try:
        from chunktuner.chunking.pdf_structural import PdfStructuralStrategy

        reg.register(PdfStructuralStrategy(encoding_name=encoding))
    except ImportError:
        pass
    try:
        from chunktuner.chunking.structural_semantic import StructuralSemanticStrategy

        reg.register(StructuralSemanticStrategy(encoding_name=encoding))
    except ImportError:
        pass
    try:
        from chunktuner.chunking.late_chunking import LateChunkingStrategy

        reg.register(LateChunkingStrategy(encoding_name=encoding))
    except ImportError:
        pass
    try:
        from chunktuner.chunking.code_ast import CodeASTStrategy

        reg.register(CodeASTStrategy(encoding_name=encoding))
    except ImportError:
        pass
    try:
        from chunktuner.chunking.code_window import CodeWindowStrategy

        reg.register(CodeWindowStrategy(encoding_name=encoding))
    except ImportError:
        pass
    try:
        from chunktuner.chunking.agentic import AgenticStrategy

        reg.register(AgenticStrategy())
    except ImportError:
        pass
    return reg
