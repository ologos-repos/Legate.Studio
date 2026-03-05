"""
OpenAI Embedding Provider

Uses OpenAI's text-embedding-3-small model for generating embeddings.
"""

import os
import logging
from typing import List, Optional

import requests

from .embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using the embeddings API."""

    EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
    DEFAULT_MODEL = "text-embedding-3-small"
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """Initialize the OpenAI embedding provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use for embeddings
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")

        self.model = model
        self._dimension = self.DIMENSIONS.get(model, 1536)
        logger.info(f"OpenAI embedding provider initialized with model: {model}")

    def create_embedding(self, text: str) -> List[float]:
        """Generate an embedding using OpenAI API.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            RuntimeError: If the API call fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot create embedding for empty text")

        # Truncate very long text (OpenAI has token limits)
        # text-embedding-3-small supports 8191 tokens
        max_chars = 30000  # Rough approximation
        if len(text) > max_chars:
            logger.warning(f"Truncating text from {len(text)} to {max_chars} chars")
            text = text[:max_chars]

        try:
            response = requests.post(
                self.EMBEDDING_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": text,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            embedding = data["data"][0]["embedding"]
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding

        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI embedding request failed: {e}")
            raise RuntimeError(f"Failed to create embedding: {e}")

    def model_identifier(self) -> str:
        """Return the provider:model identifier."""
        return f"openai:{self.model}"

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batched API calls.

        OpenAI supports up to 2048 texts per request, but we batch at 100
        for reliability and memory efficiency.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call (default 100)

        Returns:
            List of embedding vectors in the same order as input texts

        Raises:
            RuntimeError: If any API call fails
        """
        if not texts:
            return []

        all_embeddings = []
        max_chars = 30000

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Truncate long texts
            processed_batch = []
            for text in batch:
                if not text or not text.strip():
                    processed_batch.append(" ")  # OpenAI requires non-empty
                elif len(text) > max_chars:
                    logger.warning(f"Truncating text from {len(text)} to {max_chars} chars")
                    processed_batch.append(text[:max_chars])
                else:
                    processed_batch.append(text)

            try:
                response = requests.post(
                    self.EMBEDDING_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": processed_batch,
                    },
                    timeout=60,  # Longer timeout for batch
                )
                response.raise_for_status()
                data = response.json()

                # OpenAI returns embeddings with index field, sort by index
                batch_embeddings = sorted(data["data"], key=lambda x: x["index"])
                for item in batch_embeddings:
                    all_embeddings.append(item["embedding"])

                logger.info(f"Generated batch of {len(batch)} embeddings ({i+len(batch)}/{len(texts)})")

            except requests.exceptions.RequestException as e:
                logger.error(f"OpenAI batch embedding request failed: {e}")
                raise RuntimeError(f"Failed to create batch embeddings: {e}")

        return all_embeddings
