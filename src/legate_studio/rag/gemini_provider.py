"""
Gemini Embedding Provider

Uses Google's text-embedding-004 model (768 dimensions) for generating embeddings.
This is the default embedding provider — it requires only a GEMINI_API_KEY.

Dimension note: text-embedding-004 produces 768-dim vectors vs OpenAI's 1536-dim.
Existing embeddings stored in the codex_embeddings table will be incompatible after
switching providers. Run EmbeddingService.regenerate_all_embeddings() to re-embed.
"""

import logging
import os
import time

from .embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)

# Max characters to send — Gemini has a token limit; 30k chars is a safe approximation
_MAX_CHARS = 30_000


def _retry_on_quota(func, *args, max_retries: int = 3, **kwargs):
    """Retry a Gemini API call on 429 or 5xx errors with exponential backoff."""
    import google.api_core.exceptions as gexc

    delay = 2.0
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except gexc.ResourceExhausted:
            # 429 — rate limit / quota exceeded
            if attempt < max_retries - 1:
                logger.warning(f"Gemini quota exceeded, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise
        except gexc.ServiceUnavailable:
            # 503 — service temporarily unavailable
            if attempt < max_retries - 1:
                logger.warning(f"Gemini service unavailable, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise
        except gexc.InternalServerError:
            # 500
            if attempt < max_retries - 1:
                logger.warning(f"Gemini internal error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embedding provider using text-embedding-004 (768 dimensions).

    This is the preferred default provider. Requires GEMINI_API_KEY env var
    (or pass api_key to the constructor).

    IMPORTANT — dimension change:
        OpenAI text-embedding-3-small → 1536 dimensions
        Gemini text-embedding-004     → 768 dimensions

    Switching providers requires re-embedding all stored vectors. Use
    EmbeddingService.regenerate_all_embeddings() after deployment.
    """

    MODEL = "models/text-embedding-004"
    MODEL_SHORT = "text-embedding-004"
    DIMENSIONS = 768

    def __init__(self, api_key: str | None = None):
        """Initialize the Gemini embedding provider.

        Args:
            api_key: Google Gemini API key. Falls back to GEMINI_API_KEY env var.

        Raises:
            ValueError: If no API key is available.
            ImportError: If google-generativeai is not installed.
        """
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai is required for GeminiEmbeddingProvider. "
                "Install it with: uv add google-generativeai"
            ) from e

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided and GEMINI_API_KEY env var not set"
            )

        genai.configure(api_key=self.api_key)
        self._genai = genai
        logger.info(f"GeminiEmbeddingProvider initialized with model: {self.MODEL_SHORT}")

    def create_embedding(self, text: str) -> list[float]:
        """Generate an embedding using the Gemini text-embedding-004 model.

        Args:
            text: The text to embed (empty/whitespace-only text raises ValueError)

        Returns:
            List of 768 floats representing the embedding vector

        Raises:
            ValueError: If text is empty
            RuntimeError: If the API call fails after retries
        """
        if not text or not text.strip():
            raise ValueError("Cannot create embedding for empty text")

        # Truncate very long text
        if len(text) > _MAX_CHARS:
            logger.warning(f"Truncating text from {len(text)} to {_MAX_CHARS} chars for Gemini")
            text = text[:_MAX_CHARS]

        try:
            result = _retry_on_quota(
                self._genai.embed_content,
                model=self.MODEL,
                content=text,
                task_type="retrieval_document",
            )
            embedding = result["embedding"]
            logger.debug(f"Generated Gemini embedding with {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            logger.error(f"Gemini embedding request failed: {e}")
            raise RuntimeError(f"Failed to create Gemini embedding: {e}") from e

    def model_identifier(self) -> str:
        """Return the provider:model identifier."""
        return f"gemini:{self.MODEL_SHORT}"

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (768 for text-embedding-004)."""
        return self.DIMENSIONS

    def create_embeddings_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Generate embeddings for multiple texts in batched API calls.

        Gemini's embed_content accepts a list of strings as the content parameter.
        We batch at 100 for reliability and memory efficiency.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call (default 100)

        Returns:
            List of 768-dim embedding vectors in the same order as input texts

        Raises:
            RuntimeError: If any API call fails
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Truncate and sanitize
            processed: list[str] = []
            for text in batch:
                if not text or not text.strip():
                    processed.append(" ")  # non-empty placeholder
                elif len(text) > _MAX_CHARS:
                    logger.warning(f"Truncating text from {len(text)} to {_MAX_CHARS} chars")
                    processed.append(text[:_MAX_CHARS])
                else:
                    processed.append(text)

            try:
                result = _retry_on_quota(
                    self._genai.embed_content,
                    model=self.MODEL,
                    content=processed,
                    task_type="retrieval_document",
                )
                # When content is a list, result["embedding"] is a list of lists
                batch_embeddings = result["embedding"]
                all_embeddings.extend(batch_embeddings)
                logger.info(
                    f"Generated Gemini batch of {len(batch)} embeddings "
                    f"({i + len(batch)}/{len(texts)})"
                )

            except Exception as e:
                logger.error(f"Gemini batch embedding request failed at offset {i}: {e}")
                raise RuntimeError(f"Failed to create Gemini batch embeddings: {e}") from e

        return all_embeddings
