"""
Motif Processor - Core processing logic for transcripts

Processes transcripts through:
1. Parse - Split transcript into threads
2. Classify - Categorize via Claude API (Haiku 4.5)
3. Correlate - Check existing entries for similarity
4. Extract - Generate markdown artifacts via Claude
5. Write - Commit to user's Library repo

Can be used by:
- Background worker (async jobs)
- MCP server (sync processing for short content)
- Direct API calls
"""

import json
import logging
import re
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Model for classification and extraction
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# Prompts embedded from Conduct
CLASSIFIER_PROMPT = '''You are classifying segments of a voice transcript for the LEGATO system.

## Core Principle

**Everything becomes a Note first.** All threads are classified as KNOWLEDGE and stored in the Library. If a thread describes something that needs implementation, it is flagged with `needs_chord: true` for escalation.

## Your Task

For each thread:

1. **Categorize** as one of:
   - EPIPHANY: Major breakthrough or insight (rare - genuine "aha" moments)
   - CONCEPT: Technical definition, explanation, or implementation idea
   - REFLECTION: Personal thought, observation, or musing
   - GLIMMER: A captured moment - photographing a feeling. Poetic, evocative, sensory.
   - REMINDER: Note to self about something to remember
   - WORKLOG: Summary of work already completed

2. **Determine if it needs a Chord** (`needs_chord`):
   - Set `true` if this describes something to build/implement
   - Provide `chord_name` (slug-friendly) when true
   - Leave `false` for pure knowledge/reflection

3. **Extract metadata**:
   - Domain tags (2-5 relevant topics)
   - Key phrases (distinctive terms)
   - Title/summary (one line)

## Classification Signals

### Pure Knowledge (needs_chord: false):
- Past tense reflection ("I realized...", "I've been thinking...")
- Conceptual explanation ("The way X works is...")
- Insight framing ("What's interesting is...")
- Pure notes to self ("I need to remember...")
- Observations with no call to action

### Needs Chord (needs_chord: true):
- Future tense intent ("I want to build...", "We should create...")
- Direct commands/requests ("Create a...", "Make a...", "Build a...")
- Technical specification ("It should have X, Y, Z features...")
- Implementation details ("Using Python, we could...")
- Repository/codebase references ("repo", "project", "app", "tool")

### Chord Scope (when needs_chord is true):
- `chord_scope: note` - Single feature, <1 day work, simple
- `chord_scope: chord` - Multiple components, >1 day work, complex

## Output Format

Return JSON array - ALL items are type KNOWLEDGE:

```json
[
  {
    "id": "thread-001",
    "type": "KNOWLEDGE",
    "knowledge_category": "epiphany",
    "title": "Oracle Machines as AI Intuition Framework",
    "description": "Insight connecting Turing's oracle machines to modern AI behavior",
    "domain_tags": ["ai", "turing", "theory", "intuition"],
    "key_phrases": ["oracle machine", "intuition engine", "ordinal logic"],
    "needs_chord": false
  }
]
```

## Important Notes

- **Everything is KNOWLEDGE** - there is no PROJECT type
- Implementation ideas are CONCEPT or EPIPHANY with `needs_chord: true`
- Be conservative with EPIPHANY - reserve for genuinely novel insights
- GLIMMER never has `needs_chord: true`
- Chord names should be lowercase, hyphenated slugs
'''

EXTRACTOR_PROMPT = '''You are extracting structured knowledge artifacts from classified transcript threads for the LEGATO system.

## Your Task

Given a classified KNOWLEDGE thread, produce a complete markdown artifact suitable for the Legate.Library.

## Artifact Structure

Return ONLY the markdown content (no code fences), starting with the frontmatter:

---
id: library.{category}.{slug}
title: "{title}"
category: {category}
created: {timestamp}
source_transcript: {source}
domain_tags: {tags_as_yaml_list}
key_phrases: {phrases_as_yaml_list}
needs_chord: {true/false}
chord_name: {if_applicable}
chord_scope: {if_applicable}
---

# {Title}

{Main content - well-structured prose derived from transcript}

## Key Points

- {Bullet point summary}
- {Another key point}

## Context

{Any relevant background or context from the transcript}

## Category Guidelines

### EPIPHANY
- Focus on the breakthrough insight itself
- Explain why this is significant

### CONCEPT
- Define the concept precisely
- Provide examples where relevant

### REFLECTION
- Capture the personal perspective
- Preserve the authentic voice

### GLIMMER
- State the seed idea clearly
- Keep it brief but evocative

### REMINDER
- Be specific about the action
- Make it actionable

### WORKLOG
- Summarize what was accomplished
- Keep it factual

## Slug Generation

Generate slugs that are:
- Lowercase with hyphens
- 2-5 words capturing the essence
- Example: "oracle-machines-intuition" not "my-thought-about-ai"
'''


class MotifProcessor:
    """Processes a transcript through all stages."""

    def __init__(self, job_id: str, user_id: str, app=None):
        """Initialize the processor.

        Args:
            job_id: The processing job ID
            user_id: The user's ID (for API keys and DB access)
            app: Flask app (for app context in workers)
        """
        self.job_id = job_id
        self.user_id = user_id
        self.app = app
        self._anthropic_client = None
        self.source_id = None

    def process(self, transcript: str, source_id: str = None) -> List[str]:
        """Run full processing pipeline.

        Args:
            transcript: The transcript text to process
            source_id: Optional source identifier

        Returns:
            List of created entry IDs
        """
        self.source_id = source_id or f"pit-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        try:
            # Stage 1: Parse
            self._update_stage('parsing', 5)
            threads = self._parse_threads(transcript)

            if not threads:
                self._mark_complete([])
                return []

            self._create_thread_records(threads)

            # Stage 2: Classify
            self._update_stage('classifying', 10)
            self._classify_threads(threads)

            # Stage 3: Correlate
            self._update_stage('correlating', 50)
            self._correlate_threads()

            # Stage 4: Extract
            self._update_stage('extracting', 70)
            self._extract_threads()

            # Stage 5: Write
            self._update_stage('writing', 85)
            entry_ids = self._write_threads()

            # Stage 6: Generate embeddings
            self._update_stage('embedding', 92)
            self._generate_embeddings()

            # Stage 7: Queue chords for approval
            self._update_stage('queueing', 97)
            self._queue_chord_entries()

            # Complete
            self._mark_complete(entry_ids)
            return entry_ids

        except Exception as e:
            logger.error(f"Job {self.job_id} processing failed: {e}")
            self._mark_failed(str(e))
            raise

    def _parse_threads(self, transcript: str) -> List[Dict]:
        """Parse transcript into threads (simple text splitting)."""
        # Split on double newlines or "---" separators
        sections = re.split(r'\n\n\n+|^---+$', transcript, flags=re.MULTILINE)

        threads = []
        for i, section in enumerate(sections):
            content = section.strip()
            if content and len(content) > 20:  # Skip very short fragments
                threads.append({
                    'thread_id': f'thread-{i+1:03d}',
                    'thread_index': i,
                    'content': content,
                })

        logger.info(f"Job {self.job_id}: Parsed {len(threads)} threads from transcript")
        return threads

    def _create_thread_records(self, threads: List[Dict]):
        """Create thread records in database."""
        from .rag.database import init_db

        db = init_db()

        for thread in threads:
            db.execute("""
                INSERT INTO processing_threads (job_id, thread_id, thread_index, raw_content, status)
                VALUES (?, ?, ?, ?, 'pending')
            """, (self.job_id, thread['thread_id'], thread['thread_index'], thread['content']))

        # Update job with thread count
        db.execute("""
            UPDATE processing_jobs
            SET threads_total = ?, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
        """, (len(threads), self.job_id))

        db.commit()

    def _classify_threads(self, threads: List[Dict]):
        """Classify all threads via Claude API."""
        api_key = self._get_user_api_key()
        categories = self._get_user_categories()

        # Build prompt with user categories if custom
        system_prompt = CLASSIFIER_PROMPT
        if categories:
            category_desc = "\n".join([
                f"   - {c['name'].upper()}: {c.get('description', c['display_name'])}"
                for c in categories
            ])
            system_prompt = system_prompt.replace(
                "1. **Categorize** as one of:",
                f"1. **Categorize** as one of (user's categories):\n{category_desc}\n\n   Or standard categories:"
            )

        # Build threads JSON for classification
        threads_json = json.dumps([
            {'id': t['thread_id'], 'content': t['content']}
            for t in threads
        ], indent=2)

        response = self._call_claude(
            system=system_prompt,
            user=f"Classify these threads:\n\n{threads_json}",
            api_key=api_key
        )

        # Parse response
        try:
            # Handle potential markdown code fences
            response_text = response.strip()
            if response_text.startswith('```'):
                response_text = re.sub(r'^```\w*\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)

            # Extract JSON array - model may include extra text before/after
            # Find the outermost [ and ] brackets
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx + 1]

            classifications = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse classification response: {e}")
            logger.error(f"Response was: {response[:500]}")
            raise ValueError(f"Invalid classification response: {e}")

        self._update_thread_classifications(classifications)

    def _update_thread_classifications(self, classifications: List[Dict]):
        """Update thread records with classification results."""
        from .rag.database import init_db

        db = init_db()

        for cls in classifications:
            thread_id = cls.get('id')
            if not thread_id:
                continue

            db.execute("""
                UPDATE processing_threads
                SET category = ?,
                    title = ?,
                    description = ?,
                    domain_tags = ?,
                    key_phrases = ?,
                    needs_chord = ?,
                    chord_name = ?,
                    chord_scope = ?,
                    status = 'classified',
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ? AND thread_id = ?
            """, (
                cls.get('knowledge_category', 'reflection'),
                cls.get('title', ''),
                cls.get('description', ''),
                json.dumps(cls.get('domain_tags', [])),
                json.dumps(cls.get('key_phrases', [])),
                1 if cls.get('needs_chord') else 0,
                cls.get('chord_name'),
                cls.get('chord_scope'),
                self.job_id,
                thread_id
            ))

        db.commit()
        self._update_progress()

    def _correlate_threads(self):
        """Check each thread against existing entries."""
        from .rag.database import init_db, get_user_db_path

        shared_db = init_db()

        # Get user's database for similarity check
        user_db_path = get_user_db_path(self.user_id)
        if not user_db_path or not user_db_path.exists():
            logger.warning(f"User DB not found for {self.user_id}, skipping correlation")
            # Mark all as CREATE
            shared_db.execute("""
                UPDATE processing_threads
                SET correlation_action = 'CREATE',
                    correlation_score = 0.0,
                    status = 'correlated',
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ? AND status = 'classified'
            """, (self.job_id,))
            shared_db.commit()
            self._update_progress()
            return

        user_db = sqlite3.connect(str(user_db_path))
        user_db.row_factory = sqlite3.Row

        threads = shared_db.execute("""
            SELECT * FROM processing_threads
            WHERE job_id = ? AND status = 'classified'
        """, (self.job_id,)).fetchall()

        for thread in threads:
            try:
                # Simple title-based similarity check
                # (Full embedding-based correlation would require OpenAI key)
                action, target_id, score = self._simple_correlate(thread, user_db)

                shared_db.execute("""
                    UPDATE processing_threads
                    SET correlation_action = ?,
                        correlation_entry_id = ?,
                        correlation_score = ?,
                        status = 'correlated',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ? AND thread_id = ?
                """, (action, target_id, score, self.job_id, thread['thread_id']))

            except Exception as e:
                logger.error(f"Correlation failed for {thread['thread_id']}: {e}")
                # Default to CREATE on error
                shared_db.execute("""
                    UPDATE processing_threads
                    SET correlation_action = 'CREATE',
                        correlation_score = 0.0,
                        status = 'correlated',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ? AND thread_id = ?
                """, (self.job_id, thread['thread_id']))

        shared_db.commit()
        user_db.close()
        self._update_progress()

    def _simple_correlate(self, thread: Dict, user_db) -> Tuple[str, Optional[str], float]:
        """Simple correlation based on title similarity.

        Returns:
            Tuple of (action, target_entry_id, score)
        """
        title = thread['title'] or ''
        if not title:
            return 'CREATE', None, 0.0

        # Find entries with similar titles
        similar = user_db.execute("""
            SELECT entry_id, title, chord_status, chord_repo
            FROM knowledge_entries
            WHERE title LIKE ? OR title LIKE ?
            LIMIT 5
        """, (f"%{title[:20]}%", f"%{title[-20:]}%")).fetchall()

        if not similar:
            return 'CREATE', None, 0.0

        # Simple word overlap scoring
        title_words = set(title.lower().split())
        best_score = 0.0
        best_match = None

        for entry in similar:
            entry_words = set((entry['title'] or '').lower().split())
            if not entry_words:
                continue
            overlap = len(title_words & entry_words)
            score = overlap / max(len(title_words), len(entry_words))
            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= 0.95:
            return 'SKIP', best_match['entry_id'], best_score
        elif best_score >= 0.80:
            # Check if needs_chord and similar has active chord
            if thread['needs_chord'] and best_match.get('chord_status') == 'active':
                return 'QUEUE', best_match['entry_id'], best_score
            return 'APPEND', best_match['entry_id'], best_score
        else:
            return 'CREATE', None, best_score

    def _extract_threads(self):
        """Generate markdown artifacts for threads marked CREATE."""
        api_key = self._get_user_api_key()

        from .rag.database import init_db
        db = init_db()

        threads = db.execute("""
            SELECT * FROM processing_threads
            WHERE job_id = ? AND status = 'correlated' AND correlation_action = 'CREATE'
        """, (self.job_id,)).fetchall()

        for thread in threads:
            try:
                # Build extraction context
                thread_data = {
                    'category': thread['category'],
                    'title': thread['title'],
                    'description': thread['description'],
                    'content': thread['raw_content'],
                    'domain_tags': json.loads(thread['domain_tags'] or '[]'),
                    'key_phrases': json.loads(thread['key_phrases'] or '[]'),
                    'needs_chord': bool(thread['needs_chord']),
                    'chord_name': thread['chord_name'],
                    'chord_scope': thread['chord_scope'],
                    'source': self.source_id,
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                }

                prompt = f"""Extract a knowledge artifact from this classified thread:

Category: {thread_data['category']}
Title: {thread_data['title']}
Description: {thread_data['description']}
Domain Tags: {thread_data['domain_tags']}
Key Phrases: {thread_data['key_phrases']}
Needs Chord: {thread_data['needs_chord']}
Chord Name: {thread_data['chord_name'] or 'N/A'}
Source: {thread_data['source']}
Timestamp: {thread_data['timestamp']}

Raw Content:
{thread_data['content']}

Generate the complete markdown artifact with frontmatter."""

                markdown = self._call_claude(
                    system=EXTRACTOR_PROMPT,
                    user=prompt,
                    api_key=api_key
                )

                # Generate entry ID
                entry_id = self._generate_entry_id(thread['category'], thread['title'])

                db.execute("""
                    UPDATE processing_threads
                    SET extracted_markdown = ?,
                        entry_id = ?,
                        status = 'extracted',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ? AND thread_id = ?
                """, (markdown, entry_id, self.job_id, thread['thread_id']))

            except Exception as e:
                logger.error(f"Extraction failed for {thread['thread_id']}: {e}")
                db.execute("""
                    UPDATE processing_threads
                    SET status = 'failed', error_message = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ? AND thread_id = ?
                """, (str(e), self.job_id, thread['thread_id']))

        db.commit()
        self._update_progress()

    def _write_threads(self) -> List[str]:
        """Write extracted artifacts to user's Library repo."""
        from .auth import get_user_installation_token
        from .rag.github_service import create_file
        from .rag.database import init_db, get_user_db_path

        # Get installation token for user's Library
        token = get_user_installation_token(self.user_id, 'library')
        if not token:
            raise ValueError("No installation token for user's Library")

        # Get Library repo
        shared_db = init_db()
        repo_row = shared_db.execute("""
            SELECT repo_full_name FROM user_repos
            WHERE user_id = ? AND repo_type = 'library'
        """, (self.user_id,)).fetchone()

        if not repo_row:
            raise ValueError("User has no Library repo configured")

        library_repo = repo_row['repo_full_name']

        # Get user's legato DB for saving entries
        user_db_path = get_user_db_path(self.user_id)
        user_db = sqlite3.connect(str(user_db_path))
        user_db.row_factory = sqlite3.Row

        threads = shared_db.execute("""
            SELECT * FROM processing_threads
            WHERE job_id = ? AND status = 'extracted'
        """, (self.job_id,)).fetchall()

        entry_ids = []

        for thread in threads:
            try:
                # Build file path
                category = thread['category'] or 'general'
                slug = self._generate_slug(thread['title'])
                date_prefix = datetime.utcnow().strftime('%Y-%m-%d')
                file_path = f"{category}s/{date_prefix}-{slug}.md"

                # Commit to GitHub
                create_file(
                    repo=library_repo,
                    path=file_path,
                    content=thread['extracted_markdown'],
                    message=f"Add {thread['title']} via Legato.Pit",
                    token=token
                )

                # Save to user's local DB
                self._save_entry_to_user_db(user_db, thread, file_path)

                entry_ids.append(thread['entry_id'])

                shared_db.execute("""
                    UPDATE processing_threads
                    SET file_path = ?,
                        status = 'written',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ? AND thread_id = ?
                """, (file_path, self.job_id, thread['thread_id']))

                logger.info(f"Wrote {thread['entry_id']} to {library_repo}/{file_path}")

            except Exception as e:
                logger.error(f"Write failed for {thread['thread_id']}: {e}")
                shared_db.execute("""
                    UPDATE processing_threads
                    SET status = 'failed', error_message = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ? AND thread_id = ?
                """, (str(e), self.job_id, thread['thread_id']))

        shared_db.commit()
        user_db.commit()
        user_db.close()

        return entry_ids

    def _save_entry_to_user_db(self, user_db, thread: Dict, file_path: str):
        """Save the knowledge entry to user's local database."""
        user_db.execute("""
            INSERT INTO knowledge_entries (
                entry_id, title, category, content, file_path,
                source_transcript, needs_chord, chord_name, chord_scope,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(entry_id) DO UPDATE SET
                title = excluded.title,
                content = excluded.content,
                file_path = excluded.file_path,
                updated_at = CURRENT_TIMESTAMP
        """, (
            thread['entry_id'],
            thread['title'],
            thread['category'],
            thread['extracted_markdown'],
            file_path,
            self.source_id,
            thread['needs_chord'],
            thread['chord_name'],
            thread['chord_scope'],
        ))

    def _get_user_api_key(self) -> str:
        """Get user's Anthropic API key (platform key for managed tier, or BYOK)."""
        from .core import get_api_key_for_user

        api_key = get_api_key_for_user(self.user_id, 'anthropic')
        if not api_key:
            raise ValueError("User has no Anthropic API key configured. Please add your API key in Settings.")
        return api_key

    def _get_user_categories(self) -> List[Dict]:
        """Get user's custom categories."""
        from .rag.database import get_user_db_path

        user_db_path = get_user_db_path(self.user_id)
        if not user_db_path or not user_db_path.exists():
            return []

        try:
            user_db = sqlite3.connect(str(user_db_path))
            user_db.row_factory = sqlite3.Row
            rows = user_db.execute("""
                SELECT name, display_name, description
                FROM user_categories
                WHERE user_id = ? AND is_active = 1
                ORDER BY sort_order
            """, (self.user_id,)).fetchall()
            user_db.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def _call_claude(self, system: str, user: str, api_key: str) -> str:
        """Make a call to Claude API."""
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}]
        )

        return response.content[0].text

    def _generate_entry_id(self, category: str, title: str) -> str:
        """Generate a unique entry ID."""
        slug = self._generate_slug(title)
        return f"library.{category}.{slug}"

    def _generate_slug(self, title: str) -> str:
        """Generate a URL-friendly slug from title."""
        if not title:
            return secrets.token_hex(4)

        # Lowercase and replace non-alphanumeric with hyphens
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower())
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        # Limit length
        if len(slug) > 50:
            slug = slug[:50].rsplit('-', 1)[0]
        # Ensure not empty
        if not slug:
            slug = secrets.token_hex(4)
        return slug

    def _update_stage(self, stage: str, progress: int):
        """Update job stage and progress."""
        from .rag.database import init_db

        db = init_db()
        db.execute("""
            UPDATE processing_jobs
            SET current_stage = ?, progress_pct = ?, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
        """, (stage, progress, self.job_id))
        db.commit()

    def _update_progress(self):
        """Update progress based on thread completion counts."""
        from .rag.database import init_db

        db = init_db()

        stats = db.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status IN ('written', 'skipped') THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM processing_threads
            WHERE job_id = ?
        """, (self.job_id,)).fetchone()

        db.execute("""
            UPDATE processing_jobs
            SET threads_completed = ?,
                threads_failed = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
        """, (stats['completed'] or 0, stats['failed'] or 0, self.job_id))
        db.commit()

    def _mark_complete(self, entry_ids: List[str]):
        """Mark job as completed."""
        from .rag.database import init_db

        db = init_db()
        db.execute("""
            UPDATE processing_jobs
            SET status = 'completed',
                current_stage = 'complete',
                progress_pct = 100,
                result_entry_ids = ?,
                completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
        """, (','.join(entry_ids) if entry_ids else '', self.job_id))
        db.commit()

        logger.info(f"Job {self.job_id} completed with {len(entry_ids)} entries")

    def _mark_failed(self, error: str):
        """Mark job as failed."""
        from .rag.database import init_db

        db = init_db()
        db.execute("""
            UPDATE processing_jobs
            SET status = 'failed',
                error_message = ?,
                completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
        """, (error, self.job_id))
        db.commit()

    def _generate_embeddings(self):
        """Generate embeddings for entries that don't have them.

        Uses user's OpenAI API key to generate embeddings for newly created
        knowledge entries so they appear in the graph.
        """
        from .rag.database import get_user_db_path
        from .rag.embedding_service import EmbeddingService
        from .rag.openai_provider import OpenAIEmbeddingProvider
        from .auth import get_user_api_key

        # Get user's OpenAI API key
        openai_key = get_user_api_key(self.user_id, 'openai')
        if not openai_key:
            logger.warning(f"Job {self.job_id}: No OpenAI API key configured, skipping embedding generation")
            return

        user_db_path = get_user_db_path(self.user_id)
        if not user_db_path or not user_db_path.exists():
            logger.warning(f"Job {self.job_id}: No user DB found, skipping embedding generation")
            return

        try:
            user_db = sqlite3.connect(str(user_db_path))
            user_db.row_factory = sqlite3.Row

            provider = OpenAIEmbeddingProvider(api_key=openai_key)
            embedding_service = EmbeddingService(provider, user_db)

            count = embedding_service.generate_missing_embeddings('knowledge', delay=0.1)

            logger.info(f"Job {self.job_id}: Generated {count} embedding(s)")

            user_db.close()

        except Exception as e:
            logger.error(f"Job {self.job_id}: Failed to generate embeddings: {e}")

    def _queue_chord_entries(self):
        """Queue entries with needs_chord=true for user approval.

        After writing entries to the Library, check if any have needs_chord=1
        and queue them in agent_queue for approval before spawning repos.
        """
        from .rag.database import init_agents_db, get_user_db_path
        from .agents import import_chords_from_library

        user_db_path = get_user_db_path(self.user_id)
        if not user_db_path or not user_db_path.exists():
            logger.warning(f"Job {self.job_id}: No user DB found, skipping chord queueing")
            return

        user_db = sqlite3.connect(str(user_db_path))
        user_db.row_factory = sqlite3.Row

        agents_db = init_agents_db()

        try:
            stats = import_chords_from_library(user_db, agents_db, self.user_id)

            if stats['queued'] > 0:
                logger.info(f"Job {self.job_id}: Queued {stats['queued']} chord(s) for approval")
            if stats['multi_note_chords'] > 0:
                logger.info(f"Job {self.job_id}: Created {stats['multi_note_chords']} multi-note chord(s)")
        except Exception as e:
            logger.error(f"Job {self.job_id}: Failed to queue chords: {e}")
        finally:
            user_db.close()


def process_motif_sync(transcript: str, user_id: str, source_id: str = None) -> Dict:
    """Process a motif synchronously (for short content or MCP).

    Args:
        transcript: The transcript text
        user_id: The user's ID
        source_id: Optional source identifier

    Returns:
        Dict with job_id, status, entry_ids, etc.
    """
    from .rag.database import init_db
    from datetime import datetime, timedelta

    job_id = f"job-{secrets.token_hex(8)}"

    # Create job record with locked_until set to prevent background worker from claiming
    # (worker checks for locked_until IS NULL OR locked_until < now to identify crashed jobs)
    db = init_db()
    lock_until = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    db.execute("""
        INSERT INTO processing_jobs (job_id, user_id, input_content, source_id, status, worker_id, locked_until, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'processing', 'sync', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """, (job_id, user_id, transcript, source_id, lock_until))
    db.commit()

    try:
        processor = MotifProcessor(job_id, user_id)
        entry_ids = processor.process(transcript, source_id)

        return {
            'success': True,
            'job_id': job_id,
            'status': 'completed',
            'entry_ids': entry_ids,
        }
    except Exception as e:
        # Re-raise StaleInstallationError so API can handle re-auth
        from .auth import StaleInstallationError
        if isinstance(e, StaleInstallationError):
            raise
        logger.error(f"Sync processing failed: {e}")
        return {
            'success': False,
            'job_id': job_id,
            'status': 'failed',
            'error': str(e),
        }
