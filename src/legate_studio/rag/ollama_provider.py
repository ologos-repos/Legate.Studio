"""
Ollama Embedding Provider

Uses local Ollama for generating embeddings.
Supports models like nomic-embed-text, mxbai-embed-large, etc.
"""

import os
import logging
from typing import List, Optional

import requests

from .embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama local embedding provider."""

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_HOST = "http://localhost:11434"

    # Common model dimensions (can be updated as needed)
    DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: Optional[str] = None,
    ):
        """Initialize the Ollama embedding provider.

        Args:
            model: Ollama model to use for embeddings
            host: Ollama API host (defaults to OLLAMA_HOST env var or localhost)
        """
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)
        self._dimension = self.DIMENSIONS.get(model, 768)  # Default assumption

        logger.info(f"Ollama embedding provider initialized: {model} @ {self.host}")

    def create_embedding(self, text: str) -> List[float]:
        """Generate an embedding using local Ollama.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            RuntimeError: If the API call fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot create embedding for empty text")

        try:
            response = requests.post(
                f"{self.host}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text,
                },
                timeout=60,  # Local models can be slower
            )
            response.raise_for_status()
            data = response.json()

            embedding = data.get("embedding", [])
            if not embedding:
                raise RuntimeError("No embedding returned from Ollama")

            # Update dimension if we got a different size
            if len(embedding) != self._dimension:
                logger.info(f"Updating {self.model} dimension: {self._dimension} -> {len(embedding)}")
                self._dimension = len(embedding)

            logger.debug(f"Generated Ollama embedding with {len(embedding)} dimensions")
            return embedding

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.host}. Is it running?")
            raise RuntimeError(f"Ollama not available at {self.host}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama embedding request failed: {e}")
            raise RuntimeError(f"Failed to create embedding: {e}")

    def model_identifier(self) -> str:
        """Return the provider:model identifier."""
        return f"ollama:{self.model}"

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def is_available(self) -> bool:
        """Check if Ollama is available and the model is loaded."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            return self.model in model_names

        except requests.exceptions.RequestException:
            return False
