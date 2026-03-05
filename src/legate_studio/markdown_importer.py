"""
Markdown Import Classifier

Imports a ZIP of markdown files, classifies them into categories,
and adjusts their frontmatter - WITHOUT parsing them as transcripts.

This is a "classify and sort" operation, not a "parse and extract" operation.
The markdown content is preserved as-is; only frontmatter is updated.

Usage:
    1. Upload ZIP file via /import/upload
    2. Review classified files via /import/preview/<job_id>
    3. Confirm import via /import/confirm/<job_id>
"""

import io
import json
import logging
import re
import secrets
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Model for classification
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# Classification prompt for existing markdown files
# Unlike CLASSIFIER_PROMPT in motif_processor.py, this doesn't parse transcripts
# It classifies existing, complete markdown documents
MARKDOWN_CLASSIFIER_PROMPT = '''You are classifying existing markdown documents for the LEGATO knowledge library.

## Your Task

For each markdown document provided, determine:

1. **Category** - The most appropriate category for this content:
   - EPIPHANY: Major breakthrough, insight, or realization (rare - genuine "aha" moments)
   - CONCEPT: Technical definition, explanation, implementation idea, or reference material
   - REFLECTION: Personal thought, observation, journal entry, or musing
   - GLIMMER: A captured moment - poetic, evocative, sensory. Brief impressions.
   - REMINDER: Note to self, todo, or something to remember
   - WORKLOG: Summary of work completed, progress report, changelog
   - Or use a custom category if user-defined categories are provided

2. **Metadata** to extract:
   - title: A clear, descriptive title (use existing H1 or derive from content)
   - description: Brief 1-2 sentence summary
   - domain_tags: 2-5 relevant topic tags (lowercase, hyphenated)
   - key_phrases: Distinctive terms or concepts mentioned

3. **Chord Assessment** (whether this needs implementation):
   - needs_chord: true only if this describes something to BUILD or IMPLEMENT
   - chord_name: slug-friendly name for the project (if needs_chord)
   - chord_scope: "note" (simple, <1 day) or "chord" (complex, multi-day)

## Classification Signals

### Category Selection:
- Contains code, technical specs, or definitions → CONCEPT
- Past tense reflection, personal insights → REFLECTION
- Major realization, paradigm shift → EPIPHANY
- Brief, poetic, evocative → GLIMMER
- Action items, todos, reminders → REMINDER
- Progress reports, completed work → WORKLOG

### Needs Chord Signals:
- Future tense intent ("I want to build...", "We should create...")
- Project specifications or feature lists
- Implementation details with no existing code
- Repository/tool creation requests

### Pure Knowledge (needs_chord: false):
- Explanations, definitions, documentation
- Personal reflections or observations
- Completed work summaries
- Reference material

## Output Format

Return a JSON array with one object per file:

```json
[
  {
    "file_path": "original/path/in/zip.md",
    "category": "concept",
    "title": "Understanding Oracle Machines",
    "description": "Exploration of Turing's oracle machines and their relevance to AI",
    "domain_tags": ["ai", "turing", "computation", "theory"],
    "key_phrases": ["oracle machine", "halting problem", "computability"],
    "needs_chord": false,
    "chord_name": null,
    "chord_scope": null
  }
]
```

## Important Notes

- Preserve the original file path for reference
- Be conservative with EPIPHANY - most content is CONCEPT or REFLECTION
- GLIMMER is for brief, evocative pieces, not long technical documents
- If existing frontmatter has a category, consider it but reclassify if clearly wrong
- Generate meaningful domain_tags that would aid in searching/filtering
'''


@dataclass
class ImportedFile:
    """Represents a single markdown file from the import."""
    original_path: str
    content: str
    existing_frontmatter: Dict = field(default_factory=dict)
    existing_body: str = ""

    # Classification results (filled after Claude classification)
    category: str = ""
    title: str = ""
    description: str = ""
    domain_tags: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    needs_chord: bool = False
    chord_name: Optional[str] = None
    chord_scope: Optional[str] = None

    # Generated fields
    entry_id: str = ""
    target_path: str = ""

    # Status
    status: str = "pending"  # pending, classified, confirmed, written, error
    error: Optional[str] = None


@dataclass
class ImportJob:
    """Represents an import job with multiple files."""
    job_id: str
    user_id: str
    created_at: str
    files: List[ImportedFile] = field(default_factory=list)
    status: str = "pending"  # pending, classifying, classified, writing, completed, failed
    error: Optional[str] = None

    # Category mode options
    category_mode: str = "auto"  # 'auto' (classify into existing) or 'new' (use new category)
    new_category: Optional[str] = None  # Category name when mode is 'new'


def parse_frontmatter(content: str) -> Tuple[Dict, str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: Raw markdown file content

    Returns:
        Tuple of (frontmatter dict, body content)
    """
    frontmatter = {}
    body = content

    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            try:
                for line in parts[1].strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        value = value.strip().strip('"\'')
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.lower() == 'null':
                            value = None
                        # Handle arrays like [tag1, tag2]
                        elif value.startswith('[') and value.endswith(']'):
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                pass
                        frontmatter[key.strip()] = value
                body = parts[2].strip()
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter: {e}")

    return frontmatter, body


def generate_frontmatter(file: ImportedFile, source_id: str) -> str:
    """Generate YAML frontmatter string for a classified file.

    Args:
        file: The ImportedFile with classification data
        source_id: Source identifier for tracking

    Returns:
        YAML frontmatter string (without --- delimiters)
    """
    lines = []

    # Core fields
    lines.append(f'id: {file.entry_id}')
    lines.append(f'title: "{file.title}"')
    lines.append(f'category: {file.category}')
    lines.append(f'created: {datetime.utcnow().isoformat()}Z')
    lines.append(f'source_import: {source_id}')

    # Tags and phrases
    if file.domain_tags:
        lines.append(f'domain_tags: {json.dumps(file.domain_tags)}')
    if file.key_phrases:
        lines.append(f'key_phrases: {json.dumps(file.key_phrases)}')

    # Chord fields
    lines.append(f'needs_chord: {str(file.needs_chord).lower()}')
    if file.needs_chord:
        if file.chord_name:
            lines.append(f'chord_name: {file.chord_name}')
        if file.chord_scope:
            lines.append(f'chord_scope: {file.chord_scope}')

    # Preserve any existing frontmatter fields not covered above
    preserved_keys = {'author', 'date', 'updated', 'source', 'references', 'related'}
    for key, value in file.existing_frontmatter.items():
        if key in preserved_keys:
            if isinstance(value, str):
                lines.append(f'{key}: "{value}"')
            elif isinstance(value, list):
                lines.append(f'{key}: {json.dumps(value)}')
            else:
                lines.append(f'{key}: {value}')

    return '\n'.join(lines)


def generate_slug(title: str) -> str:
    """Generate a URL-safe slug from title."""
    if not title:
        return secrets.token_hex(4)
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower())[:50].strip('-')
    return slug or secrets.token_hex(4)


def generate_entry_id(category: str, title: str) -> str:
    """Generate a canonical entry ID."""
    slug = generate_slug(title)
    return f"library.{category}.{slug}"


def generate_target_path(category: str, title: str) -> str:
    """Generate target file path in the library."""
    slug = generate_slug(title)
    date_prefix = datetime.utcnow().strftime('%Y-%m-%d')
    return f"{category}/{date_prefix}-{slug}.md"


def extract_zip(zip_content: bytes) -> List[ImportedFile]:
    """Extract markdown files from a ZIP archive.

    Args:
        zip_content: Raw ZIP file bytes

    Returns:
        List of ImportedFile objects
    """
    files = []

    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        for name in zf.namelist():
            # Skip directories and non-markdown files
            if name.endswith('/') or not name.endswith('.md'):
                continue

            # Skip macOS metadata files
            if '/__MACOSX/' in name or name.startswith('__MACOSX/'):
                continue

            # Skip README files
            basename = Path(name).name
            if basename.lower() in ('readme.md', 'description.md'):
                continue

            try:
                content = zf.read(name).decode('utf-8')
                frontmatter, body = parse_frontmatter(content)

                files.append(ImportedFile(
                    original_path=name,
                    content=content,
                    existing_frontmatter=frontmatter,
                    existing_body=body,
                ))

            except Exception as e:
                logger.warning(f"Failed to read {name}: {e}")
                files.append(ImportedFile(
                    original_path=name,
                    content="",
                    status="error",
                    error=str(e),
                ))

    return files


def classify_files(
    files: List[ImportedFile],
    api_key: str,
    user_categories: List[Dict] = None
) -> List[ImportedFile]:
    """Classify a list of markdown files using Claude.

    Args:
        files: List of ImportedFile objects to classify
        api_key: Anthropic API key
        user_categories: Optional list of user-defined categories

    Returns:
        The same list with classification data filled in
    """
    import anthropic

    # Filter out already errored files
    pending_files = [f for f in files if f.status != "error"]

    if not pending_files:
        return files

    # Build prompt with file contents
    system_prompt = MARKDOWN_CLASSIFIER_PROMPT

    # Add user categories if provided
    if user_categories:
        category_desc = "\n".join([
            f"   - {c['name'].upper()}: {c.get('description', c['display_name'])}"
            for c in user_categories
        ])
        system_prompt = system_prompt.replace(
            "1. **Category** - The most appropriate category for this content:",
            f"1. **Category** - The most appropriate category (user's custom categories):\n{category_desc}\n\n   Or standard categories:"
        )

    # Build file contents for classification
    files_data = []
    for f in pending_files:
        # Include a preview of content (first 2000 chars to manage token usage)
        preview = f.existing_body[:2000]
        if len(f.existing_body) > 2000:
            preview += "\n\n[...content truncated...]"

        files_data.append({
            "file_path": f.original_path,
            "existing_frontmatter": f.existing_frontmatter,
            "content_preview": preview,
        })

    user_message = f"Classify these {len(files_data)} markdown files:\n\n{json.dumps(files_data, indent=2)}"

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )

        response_text = response.content[0].text.strip()

        # Handle markdown code fences
        if response_text.startswith('```'):
            response_text = re.sub(r'^```\w*\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)

        # Extract JSON array
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']')
        if start_idx != -1 and end_idx != -1:
            response_text = response_text[start_idx:end_idx + 1]

        classifications = json.loads(response_text)

        # Build lookup by file path
        class_by_path = {c['file_path']: c for c in classifications}

        # Apply classifications
        for f in pending_files:
            if f.original_path in class_by_path:
                c = class_by_path[f.original_path]
                f.category = c.get('category', 'concept').lower()
                f.title = c.get('title', Path(f.original_path).stem)
                f.description = c.get('description', '')
                f.domain_tags = c.get('domain_tags', [])
                f.key_phrases = c.get('key_phrases', [])
                f.needs_chord = c.get('needs_chord', False)
                f.chord_name = c.get('chord_name')
                f.chord_scope = c.get('chord_scope')

                # Generate derived fields
                f.entry_id = generate_entry_id(f.category, f.title)
                f.target_path = generate_target_path(f.category, f.title)
                f.status = "classified"
            else:
                # File not in response - mark as error
                f.status = "error"
                f.error = "Not returned in classification response"

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse classification response: {e}")
        for f in pending_files:
            f.status = "error"
            f.error = f"Classification parse error: {e}"
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        for f in pending_files:
            f.status = "error"
            f.error = str(e)

    return files


def build_final_markdown(file: ImportedFile, source_id: str) -> str:
    """Build the final markdown content with updated frontmatter.

    Args:
        file: The classified ImportedFile
        source_id: Source identifier for tracking

    Returns:
        Complete markdown string with new frontmatter
    """
    frontmatter = generate_frontmatter(file, source_id)
    return f"---\n{frontmatter}\n---\n\n{file.existing_body}"


class MarkdownImporter:
    """Handles the full import workflow for markdown files."""

    def __init__(self, user_id: str, app=None):
        """Initialize the importer.

        Args:
            user_id: The user's ID
            app: Flask app for context
        """
        self.user_id = user_id
        self.app = app

    def create_job(self, zip_content: bytes) -> ImportJob:
        """Create an import job from a ZIP file.

        Args:
            zip_content: Raw ZIP file bytes

        Returns:
            ImportJob with extracted files
        """
        job_id = f"import-{secrets.token_hex(8)}"

        files = extract_zip(zip_content)

        job = ImportJob(
            job_id=job_id,
            user_id=self.user_id,
            created_at=datetime.utcnow().isoformat() + 'Z',
            files=files,
            status="pending",
        )

        logger.info(f"Created import job {job_id} with {len(files)} files")
        return job

    def classify_job(self, job: ImportJob) -> ImportJob:
        """Run classification on all files in the job.

        Supports two modes:
        - 'auto': Use Claude to classify into existing categories
        - 'new': Assign all files to a single new category (no Claude needed)

        Args:
            job: The ImportJob to classify

        Returns:
            The job with classification data filled in
        """
        job.status = "classifying"

        try:
            if job.category_mode == 'new' and job.new_category:
                # New category mode - no Claude needed
                self._classify_to_new_category(job)
            else:
                # Auto mode - use Claude classification
                from .core import get_api_key_for_user

                api_key = get_api_key_for_user(self.user_id, 'anthropic')
                if not api_key:
                    job.status = "failed"
                    job.error = "No Anthropic API key configured"
                    return job

                # Get user's custom categories
                user_categories = self._get_user_categories()
                classify_files(job.files, api_key, user_categories)

            # Check if all files classified successfully
            classified_count = sum(1 for f in job.files if f.status == "classified")
            error_count = sum(1 for f in job.files if f.status == "error")

            if classified_count > 0:
                job.status = "classified"
            else:
                job.status = "failed"
                job.error = f"All {error_count} files failed classification"

            logger.info(f"Job {job.job_id}: {classified_count} classified, {error_count} errors")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Job {job.job_id} classification failed: {e}")

        return job

    def _classify_to_new_category(self, job: ImportJob):
        """Assign all files to a single new category without Claude.

        Extracts title from filename or content, generates minimal metadata.
        """
        category = job.new_category.lower().replace(' ', '-')

        for f in job.files:
            if f.status == "error":
                continue

            try:
                # Extract title from existing frontmatter or filename
                title = f.existing_frontmatter.get('title')
                if not title:
                    # Extract from first H1 in content
                    import re
                    match = re.search(r'^#\s+(.+)$', f.existing_body, re.MULTILINE)
                    if match:
                        title = match.group(1).strip()
                    else:
                        # Fall back to filename
                        title = Path(f.original_path).stem
                        title = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', title)
                        title = title.replace('-', ' ').replace('_', ' ').title()

                # Use existing tags if present
                tags = f.existing_frontmatter.get('domain_tags', [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(',')]

                f.category = category
                f.title = title
                f.description = f.existing_frontmatter.get('description', '')
                f.domain_tags = tags if tags else []
                f.key_phrases = f.existing_frontmatter.get('key_phrases', [])
                f.needs_chord = f.existing_frontmatter.get('needs_chord', False)
                f.chord_name = f.existing_frontmatter.get('chord_name')
                f.chord_scope = f.existing_frontmatter.get('chord_scope')

                # Generate derived fields
                f.entry_id = generate_entry_id(f.category, f.title)
                f.target_path = generate_target_path(f.category, f.title)
                f.status = "classified"

            except Exception as e:
                f.status = "error"
                f.error = str(e)

    def write_files(self, job: ImportJob, source_id: str = None) -> List[str]:
        """Write classified files to the user's Library.

        Args:
            job: The classified ImportJob
            source_id: Optional source identifier

        Returns:
            List of created entry IDs
        """
        from .auth import get_user_installation_token
        from .rag.github_service import create_file
        from .rag.database import init_db, get_user_db_path
        import sqlite3

        source_id = source_id or f"import-{job.job_id}"

        # Get installation token for user's Library
        token = get_user_installation_token(self.user_id, 'library')
        if not token:
            raise ValueError("No installation token for user's Library")

        # Get Library repo
        shared_db = init_db()
        repo_row = shared_db.execute(
            "SELECT repo_full_name FROM user_repos WHERE user_id = ? AND repo_type = 'library'",
            (self.user_id,)
        ).fetchone()

        if not repo_row:
            raise ValueError("User has no Library repo configured")

        library_repo = repo_row['repo_full_name']

        # Get user's legato DB for saving entries
        user_db_path = get_user_db_path(self.user_id)
        user_db = sqlite3.connect(str(user_db_path))
        user_db.row_factory = sqlite3.Row

        entry_ids = []
        job.status = "writing"

        for file in job.files:
            if file.status != "classified":
                continue

            try:
                # Build final markdown
                markdown = build_final_markdown(file, source_id)

                # Commit to GitHub
                create_file(
                    repo=library_repo,
                    path=file.target_path,
                    content=markdown,
                    message=f"Import {file.title} from {source_id}",
                    token=token
                )

                # Save to user's local DB
                self._save_entry(user_db, file, source_id)

                entry_ids.append(file.entry_id)
                file.status = "written"

                logger.info(f"Wrote {file.entry_id} to {library_repo}/{file.target_path}")

            except Exception as e:
                logger.error(f"Failed to write {file.original_path}: {e}")
                file.status = "error"
                file.error = str(e)

        user_db.commit()
        user_db.close()

        # Generate embeddings for imported entries
        if entry_ids:
            self._generate_embeddings(entry_ids)

        job.status = "completed"
        return entry_ids

    def _generate_embeddings(self, entry_ids: List[str]):
        """Generate embeddings for the imported entries.

        Args:
            entry_ids: List of entry IDs to generate embeddings for
        """
        import os
        from .rag.database import get_user_db_path
        from .core import get_api_key_for_user
        import sqlite3

        try:
            openai_key = get_api_key_for_user(self.user_id, 'openai')
            if not openai_key:
                logger.warning("No OpenAI API key - skipping embedding generation")
                return

            from .rag.openai_provider import OpenAIEmbeddingProvider
            from .rag.embedding_service import EmbeddingService

            user_db_path = get_user_db_path(self.user_id)
            user_db = sqlite3.connect(str(user_db_path))
            user_db.row_factory = sqlite3.Row

            provider = OpenAIEmbeddingProvider(api_key=openai_key)
            embedding_service = EmbeddingService(provider, user_db)

            # Generate embeddings for the imported entries
            count = embedding_service.generate_missing_embeddings('knowledge', delay=0.05)
            logger.info(f"Generated {count} embeddings for imported entries")

            user_db.close()

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")

    def _get_user_categories(self) -> List[Dict]:
        """Get user's custom categories."""
        from .rag.database import get_user_db_path
        import sqlite3

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

    def _save_entry(self, user_db, file: ImportedFile, source_id: str):
        """Save a knowledge entry to user's local database."""
        user_db.execute("""
            INSERT INTO knowledge_entries (
                entry_id, title, category, content, file_path,
                source_transcript, needs_chord, chord_name, chord_scope,
                domain_tags, key_phrases,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(entry_id) DO UPDATE SET
                title = excluded.title,
                content = excluded.content,
                file_path = excluded.file_path,
                updated_at = CURRENT_TIMESTAMP
        """, (
            file.entry_id,
            file.title,
            file.category,
            file.existing_body,
            file.target_path,
            source_id,
            1 if file.needs_chord else 0,
            file.chord_name,
            file.chord_scope,
            json.dumps(file.domain_tags) if file.domain_tags else None,
            json.dumps(file.key_phrases) if file.key_phrases else None,
        ))


# ============ Job Storage ============
# In-memory storage for import jobs (could be moved to DB for persistence)

_import_jobs: Dict[str, ImportJob] = {}


def store_job(job: ImportJob):
    """Store a job in memory."""
    _import_jobs[job.job_id] = job


def get_job(job_id: str, user_id: str = None) -> Optional[ImportJob]:
    """Retrieve a job by ID, optionally verifying ownership."""
    job = _import_jobs.get(job_id)
    if job and user_id and job.user_id != user_id:
        return None
    return job


def delete_job(job_id: str):
    """Delete a job from storage."""
    _import_jobs.pop(job_id, None)


def list_user_jobs(user_id: str) -> List[ImportJob]:
    """List all jobs for a user."""
    return [j for j in _import_jobs.values() if j.user_id == user_id]
