"""
Legato.Pit RAG (Retrieval-Augmented Generation) Module

Provides SQLite-based vector storage and semantic search for the LEGATO knowledge base.
"""

from .database import init_db, get_db_path
from .embedding_service import EmbeddingService
from .context_builder import ContextBuilder

__all__ = [
    'init_db',
    'get_db_path',
    'EmbeddingService',
    'ContextBuilder',
]
