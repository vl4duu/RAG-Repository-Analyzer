"""Top-level package for the RAG Repository Analyzer backend logic.

This file ensures that the "src" directory is treated as a regular Python
package across diverse runtimes (including some production environments)
so that absolute and relative imports like `from src.rag_service import RAGService`
and `from .metadata_index import MetadataIndex` work reliably.
"""

__all__ = []
