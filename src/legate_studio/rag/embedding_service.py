"""
Embedding Service

Core RAG logic including:
- Embedding storage and retrieval
- Similarity search
- Correlation checking
"""

import struct
import logging
import sqlite3
from typing import List, Dict, Optional, Tuple
from threading import Lock

from .embedding_provider import EmbeddingProvider
from .database import get_connection

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for managing embeddings and similarity search."""

    def __init__(self, provider: EmbeddingProvider, db_conn: Optional[sqlite3.Connection] = None):
        """Initialize the embedding service.

        Args:
            provider: The embedding provider to use
            db_conn: Optional database connection (creates new if not provided)
        """
        self.provider = provider
        self.conn = db_conn or get_connection()
        self._lock = Lock()

        logger.info(f"EmbeddingService initialized with {provider.model_identifier()}")

    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Convert float list to binary blob for storage.

        Uses little-endian float32 format (same as Llore).
        """
        return struct.pack(f'<{len(embedding)}f', *embedding)

    def _deserialize_embedding(self, blob: bytes) -> List[float]:
        """Convert binary blob back to float list."""
        count = len(blob) // 4  # 4 bytes per float32
        return list(struct.unpack(f'<{count}f', blob))

    def store_embedding(
        self,
        entry_id: int,
        entry_type: str,
        embedding: List[float],
    ) -> bool:
        """Store an embedding in the database.

        Args:
            entry_id: The ID of the entry (from knowledge_entries or project_entries)
            entry_type: 'knowledge' or 'project'
            embedding: The embedding vector

        Returns:
            True if successful
        """
        blob = self._serialize_embedding(embedding)
        version = self.provider.model_identifier()

        with self._lock:
            try:
                self.conn.execute(
                    """
                    INSERT INTO embeddings (entry_id, entry_type, embedding, vector_version)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(entry_id, entry_type, vector_version)
                    DO UPDATE SET embedding = excluded.embedding, updated_at = CURRENT_TIMESTAMP
                    """,
                    (entry_id, entry_type, blob, version),
                )
                self.conn.commit()
                logger.debug(f"Stored embedding for {entry_type}:{entry_id}")
                return True

            except sqlite3.Error as e:
                logger.error(f"Failed to store embedding: {e}")
                return False

    def get_embedding(self, entry_id: int, entry_type: str = 'knowledge') -> Optional[List[float]]:
        """Retrieve an embedding from the database.

        Args:
            entry_id: The entry ID
            entry_type: 'knowledge' or 'project'

        Returns:
            The embedding vector or None if not found
        """
        version = self.provider.model_identifier()

        row = self.conn.execute(
            """
            SELECT embedding FROM embeddings
            WHERE entry_id = ? AND entry_type = ? AND vector_version = ?
            """,
            (entry_id, entry_type, version),
        ).fetchone()

        if row:
            return self._deserialize_embedding(row[0])
        return None

    def generate_and_store(
        self,
        entry_id: int,
        entry_type: str,
        text: str,
    ) -> Optional[List[float]]:
        """Generate an embedding for text and store it.

        Args:
            entry_id: The entry ID
            entry_type: 'knowledge' or 'project'
            text: The text to embed

        Returns:
            The embedding vector or None if failed
        """
        try:
            embedding = self.provider.create_embedding(text)
            self.store_embedding(entry_id, entry_type, embedding)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score between -1 and 1
        """
        if len(a) != len(b):
            raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")

        # Use float64 for precision (like Llore)
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)

        # Clamp to [-1, 1] to handle floating point errors
        return max(-1.0, min(1.0, similarity))

    def find_similar(
        self,
        query_text: str,
        entry_type: str = 'knowledge',
        limit: int = 10,
        threshold: float = 0.4,
    ) -> List[Dict]:
        """Find entries similar to the query text.

        Args:
            query_text: Text to search for
            entry_type: 'knowledge' or 'project'
            limit: Maximum results to return
            threshold: Minimum similarity score

        Returns:
            List of dicts with entry info and similarity scores
        """
        # Generate query embedding
        try:
            query_embedding = self.provider.create_embedding(query_text)
        except Exception as e:
            logger.error(f"Failed to create query embedding: {e}")
            return []

        version = self.provider.model_identifier()

        # Get all embeddings for comparison
        if entry_type == 'knowledge':
            rows = self.conn.execute(
                """
                SELECT e.entry_id, e.embedding, k.entry_id as eid, k.title, k.category, k.content
                FROM embeddings e
                JOIN knowledge_entries k ON e.entry_id = k.id
                WHERE e.entry_type = 'knowledge' AND e.vector_version = ?
                """,
                (version,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT e.entry_id, e.embedding, p.project_id as eid, p.title, p.status, p.description
                FROM embeddings e
                JOIN project_entries p ON e.entry_id = p.id
                WHERE e.entry_type = 'project' AND e.vector_version = ?
                """,
                (version,),
            ).fetchall()

        # Calculate similarities
        results = []
        for row in rows:
            stored_embedding = self._deserialize_embedding(row['embedding'])
            similarity = self.cosine_similarity(query_embedding, stored_embedding)

            if similarity >= threshold:
                results.append({
                    'id': row['entry_id'],
                    'entry_id': row['eid'],
                    'title': row['title'],
                    'category': row['category'] if entry_type == 'knowledge' else row['status'],
                    'content': row['content'] if entry_type == 'knowledge' else row['description'],
                    'similarity': similarity,
                })

        # Sort by similarity (descending) and limit
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    def correlate(
        self,
        title: str,
        content: str,
        threshold_skip: float = 0.90,
        threshold_suggest: float = 0.70,
    ) -> Dict:
        """Check if similar content already exists.

        Args:
            title: Entry title
            content: Entry content
            threshold_skip: Score above this = SKIP (likely duplicate)
            threshold_suggest: Score above this = SUGGEST (needs review)

        Returns:
            Dict with action (CREATE/SUGGEST/SKIP), score, and matches
        """
        # Combine title and content for embedding
        text = f"Title: {title}\n\nContent: {content}"

        similar = self.find_similar(text, limit=5, threshold=threshold_suggest)

        if not similar:
            return {
                'action': 'CREATE',
                'score': 0.0,
                'matches': [],
            }

        top_score = similar[0]['similarity']

        if top_score >= threshold_skip:
            action = 'SKIP'
        elif top_score >= threshold_suggest:
            action = 'SUGGEST'
        else:
            action = 'CREATE'

        return {
            'action': action,
            'score': top_score,
            'matches': [
                {
                    'entry_id': m['entry_id'],
                    'title': m['title'],
                    'similarity': round(m['similarity'], 3),
                }
                for m in similar
            ],
        }

    def get_entries_without_embeddings(self, entry_type: str = 'knowledge') -> List[Tuple[int, str]]:
        """Find entries that don't have embeddings yet.

        Returns:
            List of (id, text) tuples for entries needing embeddings
        """
        version = self.provider.model_identifier()

        if entry_type == 'knowledge':
            rows = self.conn.execute(
                """
                SELECT k.id, k.title, k.content
                FROM knowledge_entries k
                LEFT JOIN embeddings e ON k.id = e.entry_id
                    AND e.entry_type = 'knowledge'
                    AND e.vector_version = ?
                WHERE e.id IS NULL
                """,
                (version,),
            ).fetchall()
            return [(r['id'], f"Title: {r['title']}\n\nContent: {r['content']}") for r in rows]

        else:
            rows = self.conn.execute(
                """
                SELECT p.id, p.title, p.description
                FROM project_entries p
                LEFT JOIN embeddings e ON p.id = e.entry_id
                    AND e.entry_type = 'project'
                    AND e.vector_version = ?
                WHERE e.id IS NULL
                """,
                (version,),
            ).fetchall()
            return [(r['id'], f"Title: {r['title']}\n\nDescription: {r['description'] or ''}") for r in rows]

    def generate_missing_embeddings(self, entry_type: str = 'knowledge', delay: float = 0.1, batch_size: int = 100) -> int:
        """Generate embeddings for all entries that don't have them.

        Uses batch API calls for efficiency when processing many entries.

        Args:
            entry_type: 'knowledge' or 'project'
            delay: Seconds to wait between batch API calls (not per-item)
            batch_size: Number of entries to process per batch API call

        Returns:
            Number of embeddings generated
        """
        import time

        entries = self.get_entries_without_embeddings(entry_type)
        if not entries:
            logger.info(f"No missing embeddings for {entry_type}")
            return 0

        logger.info(f"Generating embeddings for {len(entries)} {entry_type} entries in batches of {batch_size}")
        count = 0

        # Process in batches
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            entry_ids = [e[0] for e in batch]
            texts = [e[1] for e in batch]

            try:
                # Use batch embedding API
                embeddings = self.provider.create_embeddings_batch(texts)

                # Store all embeddings from this batch
                version = self.provider.model_identifier()
                with self._lock:
                    for entry_id, embedding in zip(entry_ids, embeddings):
                        blob = self._serialize_embedding(embedding)
                        self.conn.execute(
                            """
                            INSERT INTO embeddings (entry_id, entry_type, embedding, vector_version)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(entry_id, entry_type, vector_version)
                            DO UPDATE SET embedding = excluded.embedding, updated_at = CURRENT_TIMESTAMP
                            """,
                            (entry_id, entry_type, blob, version),
                        )
                    self.conn.commit()

                count += len(embeddings)
                logger.info(f"Generated batch of {len(embeddings)} embeddings ({count}/{len(entries)} total)")

                # Delay between batches (not between individual items)
                if delay > 0 and i + batch_size < len(entries):
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch starting at {i}: {e}")
                # Continue with next batch instead of failing entirely
                continue

        return count

    # ============ Enhanced Search ============

    def keyword_search(
        self,
        query: str,
        entry_type: str = 'knowledge',
        limit: int = 20,
    ) -> List[Dict]:
        """Search entries using keyword matching on title, content, and tags.

        Args:
            query: Search query
            entry_type: 'knowledge' or 'project'
            limit: Maximum results

        Returns:
            List of matching entries with match_type='keyword'
        """
        if entry_type != 'knowledge':
            return []

        # Split query into terms and create LIKE patterns
        terms = [t.strip().lower() for t in query.split() if len(t.strip()) >= 2]
        if not terms:
            return []

        results = []

        for term in terms:
            pattern = f'%{term}%'

            rows = self.conn.execute(
                """
                SELECT id, entry_id, title, category, content, domain_tags, key_phrases
                FROM knowledge_entries
                WHERE LOWER(title) LIKE ?
                   OR LOWER(content) LIKE ?
                   OR LOWER(domain_tags) LIKE ?
                   OR LOWER(key_phrases) LIKE ?
                LIMIT ?
                """,
                (pattern, pattern, pattern, pattern, limit * 2),
            ).fetchall()

            for row in rows:
                # Count how many terms match for rough scoring
                text = f"{row['title']} {row['content']} {row['domain_tags'] or ''} {row['key_phrases'] or ''}".lower()
                term_hits = sum(1 for t in terms if t in text)
                score = term_hits / len(terms)  # 0.0 to 1.0

                results.append({
                    'id': row['id'],
                    'entry_id': row['entry_id'],
                    'title': row['title'],
                    'category': row['category'],
                    'content': row['content'],
                    'similarity': score,
                    'match_type': 'keyword',
                })

        # Deduplicate by entry_id, keeping highest score
        seen = {}
        for r in results:
            eid = r['entry_id']
            if eid not in seen or r['similarity'] > seen[eid]['similarity']:
                seen[eid] = r

        results = sorted(seen.values(), key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    def hybrid_search(
        self,
        query: str,
        entry_type: str = 'knowledge',
        limit: int = 10,
        semantic_threshold: float = 0.25,
        keyword_boost: float = 0.15,
        include_low_confidence: bool = True,
    ) -> Dict:
        """Hybrid search combining semantic and keyword matching.

        Args:
            query: Search query
            entry_type: 'knowledge' or 'project'
            limit: Maximum high-confidence results
            semantic_threshold: Minimum semantic similarity for high-confidence
            keyword_boost: Boost added when keyword also matches
            include_low_confidence: Whether to return low-confidence bucket

        Returns:
            Dict with 'results' (high confidence) and 'maybe_related' (low confidence)
        """
        # Get semantic results with low threshold to capture more
        semantic_results = self.find_similar(
            query_text=query,
            entry_type=entry_type,
            limit=limit * 3,  # Get more than we need
            threshold=0.15,   # Very low threshold to not miss anything
        )

        # Get keyword results
        keyword_results = self.keyword_search(
            query=query,
            entry_type=entry_type,
            limit=limit * 2,
        )

        # Build a map of entry_id -> best result with combined scoring
        combined = {}

        # Add semantic results
        for r in semantic_results:
            eid = r['entry_id']
            combined[eid] = {
                **r,
                'semantic_score': r['similarity'],
                'keyword_score': 0.0,
                'match_types': ['semantic'],
            }

        # Merge keyword results
        for r in keyword_results:
            eid = r['entry_id']
            if eid in combined:
                # Entry found by both - boost the score
                combined[eid]['keyword_score'] = r['similarity']
                combined[eid]['match_types'].append('keyword')
                # Combined score with keyword boost
                combined[eid]['similarity'] = min(1.0,
                    combined[eid]['semantic_score'] + (r['similarity'] * keyword_boost)
                )
            else:
                # Only found by keyword
                combined[eid] = {
                    **r,
                    'semantic_score': 0.0,
                    'keyword_score': r['similarity'],
                    'match_types': ['keyword'],
                }

        # Sort by combined similarity
        all_results = sorted(combined.values(), key=lambda x: x['similarity'], reverse=True)

        # Split into high-confidence and low-confidence
        high_confidence = []
        low_confidence = []

        for r in all_results:
            # High confidence if semantic score is above threshold
            # OR if keyword score is strong (> 0.5 = more than half the terms matched)
            if r.get('semantic_score', 0) >= semantic_threshold or r.get('keyword_score', 0) >= 0.5:
                if len(high_confidence) < limit:
                    high_confidence.append(r)
                elif include_low_confidence:
                    low_confidence.append(r)
            elif include_low_confidence:
                low_confidence.append(r)

        # Limit low confidence results
        low_confidence = low_confidence[:limit]

        return {
            'results': high_confidence,
            'maybe_related': low_confidence if include_low_confidence else [],
            'total_found': len(all_results),
        }

    def expand_query(self, query: str, max_expansions: int = 5) -> List[str]:
        """Expand query with synonyms and related terms using Claude.

        Args:
            query: Original query
            max_expansions: Maximum additional queries to generate

        Returns:
            List of expanded queries (includes original)
        """
        import os
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic package not installed, skipping query expansion")
            return [query]

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return [query]

        try:
            client = anthropic.Anthropic(api_key=api_key)

            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": f"""Generate {max_expansions} alternative search queries for: "{query}"

Return ONLY the alternative queries, one per line. Include synonyms, related concepts, and rephrased versions. Be concise."""
                }]
            )

            text = response.content[0].text
            expansions = [line.strip() for line in text.strip().split('\n') if line.strip()]
            expansions = expansions[:max_expansions]

            # Include original query first
            return [query] + expansions

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]

    def search_with_expansion(
        self,
        query: str,
        entry_type: str = 'knowledge',
        limit: int = 10,
        expand: bool = True,
    ) -> Dict:
        """Search with optional query expansion for better recall.

        Args:
            query: Search query
            entry_type: 'knowledge' or 'project'
            limit: Maximum results
            expand: Whether to expand query with related terms

        Returns:
            Dict with results and metadata
        """
        queries = self.expand_query(query) if expand else [query]

        # Run hybrid search for each query variant
        all_results = {}

        for q in queries:
            result = self.hybrid_search(
                query=q,
                entry_type=entry_type,
                limit=limit,
                include_low_confidence=True,
            )

            # Merge results, keeping best scores
            for r in result['results'] + result['maybe_related']:
                eid = r['entry_id']
                if eid not in all_results or r['similarity'] > all_results[eid]['similarity']:
                    all_results[eid] = r

        # Sort and split
        sorted_results = sorted(all_results.values(), key=lambda x: x['similarity'], reverse=True)

        high = [r for r in sorted_results if r.get('semantic_score', 0) >= 0.25 or r.get('keyword_score', 0) >= 0.5]
        low = [r for r in sorted_results if r not in high]

        return {
            'results': high[:limit],
            'maybe_related': low[:limit],
            'queries_used': queries,
            'total_found': len(sorted_results),
        }
