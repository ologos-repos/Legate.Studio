"""
Library Sync

Synchronizes content from Legate.Library GitHub repository into the local SQLite database.
Supports both GitHub API fetching and local filesystem sync.
"""

import os
import re
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


def _parse_frontmatter_date(date_value) -> Optional[str]:
    """Parse a frontmatter date value to ISO format string.

    Handles various formats:
    - ISO strings: "2024-01-15T10:30:00Z"
    - Date strings: "2024-01-15"
    - Datetime objects
    - None

    Returns:
        ISO format string or None
    """
    if not date_value:
        return None

    if isinstance(date_value, datetime):
        return date_value.isoformat()

    if isinstance(date_value, str):
        # Already looks like ISO format
        if 'T' in date_value or len(date_value) == 10:
            return date_value
        # Try parsing common formats
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d']:
            try:
                return datetime.strptime(date_value, fmt).isoformat()
            except ValueError:
                continue

    return None


def parse_markdown_frontmatter(content: str) -> Tuple[Dict, str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: Raw markdown file content

    Returns:
        Tuple of (frontmatter dict, body content)
    """
    frontmatter = {}
    body = content

    # Check for YAML frontmatter (--- delimited)
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            try:
                # Simple YAML parsing (key: value)
                for line in parts[1].strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        value = value.strip().strip('"\'')
                        # Parse boolean values
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.lower() == 'null':
                            value = None
                        frontmatter[key.strip()] = value
                body = parts[2].strip()
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter: {e}")

    return frontmatter, body


def extract_title_from_content(content: str, filename: str) -> str:
    """Extract title from markdown content or filename.

    Args:
        content: Markdown content
        filename: Original filename

    Returns:
        Extracted title
    """
    # Try to find first H1 heading
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Fall back to filename
    name = Path(filename).stem
    # Remove date prefix if present (e.g., 2026-01-08-title)
    name = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', name)
    # Convert kebab-case to title case
    return name.replace('-', ' ').title()


# Category normalization map - shared between functions
CATEGORY_MAP = {
    # Singular (canonical)
    'concept': 'concept',
    'epiphany': 'epiphany',
    'reflection': 'reflection',
    'glimmer': 'glimmer',
    'reminder': 'reminder',
    'worklog': 'worklog',
    'tech-thought': 'tech-thought',
    'research-topic': 'research-topic',
    'theology': 'theology',
    'agent-thought': 'agent-thought',
    'article-idea': 'article-idea',
    'writing': 'writing',
    # Plural (legacy)
    'concepts': 'concept',
    'epiphanies': 'epiphany',
    'reflections': 'reflection',
    'glimmers': 'glimmer',
    'reminders': 'reminder',
    'worklogs': 'worklog',
    'tech-thoughts': 'tech-thought',
    'research-topics': 'research-topic',
    'theologies': 'theology',
    'agent-thoughts': 'agent-thought',
    'article-ideas': 'article-idea',
    'writings': 'writing',
    # Typos
    'epiphanys': 'epiphany',
    'theologys': 'theology',
    'tech-thoughtss': 'tech-thought',
    'research-topicss': 'research-topic',
    # Other
    'procedures': 'procedure',
    'procedure': 'procedure',
    'references': 'reference',
    'reference': 'reference',
}


def normalize_category(category: str) -> str:
    """Normalize a category name to its canonical singular form.

    Args:
        category: Category string (may be plural, have typos, etc.)

    Returns:
        Canonical category string (singular form)
    """
    if not category:
        return 'general'
    return CATEGORY_MAP.get(category.lower(), category.lower())


def categorize_from_path(path: str) -> str:
    """Determine category from file path.

    Normalizes folder names to canonical singular category names.
    Handles both old plural folders and new singular folders.

    Args:
        path: File path (e.g., "concept/file.md" or "concepts/file.md")

    Returns:
        Canonical category string (singular form)
    """
    parts = Path(path).parts
    if len(parts) > 0:
        folder = parts[0].lower()
        return CATEGORY_MAP.get(folder, folder)
    return 'general'


def compute_content_hash(content: str) -> str:
    """Compute a stable hash of content for deduplication and integrity.

    Args:
        content: The markdown body content (after frontmatter)

    Returns:
        First 16 characters of SHA256 hash
    """
    # Normalize: strip whitespace, lowercase for comparison stability
    normalized = content.strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def generate_slug(title: str) -> str:
    """Generate a URL-safe slug from a title.

    Args:
        title: The note title

    Returns:
        Slug like "my-note-title"
    """
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower())[:50].strip('-')
    return slug or 'untitled'


def generate_entry_id(category: str, title: str, content_hash: str = None) -> str:
    """Generate a canonical entry ID in the standard format.

    Args:
        category: Entry category (singular form like 'concept')
        title: Entry title
        content_hash: Optional content hash to append for disambiguation

    Returns:
        Entry ID like "library.concept.my-note-title" or
        "library.concept.my-note-title-abc123" if disambiguated
    """
    slug = generate_slug(title)
    # Use singular category form for consistency
    base_id = f"library.{category}.{slug}"
    if content_hash:
        # Append first 6 chars of hash to disambiguate
        return f"{base_id}-{content_hash[:6]}"
    return base_id


class LibrarySync:
    """Synchronizes Legate.Library content to SQLite database."""

    def __init__(self, db_conn, embedding_service=None):
        """Initialize the sync service.

        Args:
            db_conn: SQLite database connection
            embedding_service: Optional EmbeddingService for generating embeddings
        """
        self.conn = db_conn
        self.embedding_service = embedding_service

    def sync_from_github(
        self,
        repo: str = "bobbyhiddn/Legate.Library",
        token: Optional[str] = None,
        branch: str = "main",
    ) -> Dict:
        """Sync content from GitHub repository.

        Args:
            repo: GitHub repo in "owner/repo" format
            token: GitHub PAT for API access
            branch: Branch to sync from

        Returns:
            Dict with sync statistics
        """
        # Token must be provided by caller - no fallback to SYSTEM_PAT for security
        if not token:
            raise ValueError("GitHub token required for sync - callers must provide user token")

        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.github+json',
        }

        stats = {
            'files_found': 0,
            'entries_created': 0,
            'entries_updated': 0,
            'errors': 0,
            'embeddings_generated': 0,
        }

        try:
            # Get default branch from repo info
            repo_info_url = f"https://api.github.com/repos/{repo}"
            repo_response = requests.get(repo_info_url, headers=headers, timeout=10)
            if repo_response.ok:
                repo_data = repo_response.json()
                branch = repo_data.get('default_branch', branch)
                # Check if repo is empty (no commits)
                if repo_data.get('size', 0) == 0:
                    logger.warning(f"Repository {repo} appears to be empty (no commits)")
                    stats['errors'] = 0
                    stats['message'] = 'Repository is empty - please add initial content'
                    return stats

            # Get repository tree
            tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
            response = requests.get(tree_url, headers=headers, timeout=30)
            response.raise_for_status()
            tree_data = response.json()

            # Filter for markdown files (exclude README.md and description.md)
            md_files = [
                item for item in tree_data.get('tree', [])
                if item['type'] == 'blob' and item['path'].endswith('.md')
                and not item['path'].startswith('.')
                and item['path'] != 'README.md'
                and not item['path'].endswith('/description.md')
            ]

            stats['files_found'] = len(md_files)
            logger.info(f"Found {len(md_files)} markdown files in {repo}")

            # Build set of current file paths for move detection
            current_github_paths = {item['path'] for item in md_files}

            # Process each file
            for item in md_files:
                try:
                    result = self._process_github_file(
                        repo, item['path'], item['sha'], headers,
                        current_github_paths=current_github_paths
                    )
                    if result == 'created':
                        stats['entries_created'] += 1
                    elif result == 'updated':
                        stats['entries_updated'] += 1
                except Exception as e:
                    logger.error(f"Error processing {item['path']}: {e}")
                    stats['errors'] += 1

            # Generate embeddings only if we created/updated entries
            if self.embedding_service and (stats['entries_created'] > 0 or stats['entries_updated'] > 0):
                stats['embeddings_generated'] = self.embedding_service.generate_missing_embeddings(
                    'knowledge', delay=0.1
                )

            # Also sync assets from */assets/ folders
            asset_files = [
                item for item in tree_data.get('tree', [])
                if item['type'] == 'blob'
                and '/assets/' in item['path']
                and not item['path'].endswith('.gitkeep')
                and not item['path'].startswith('.')
            ]

            stats['assets_found'] = len(asset_files)
            stats['assets_created'] = 0
            stats['assets_updated'] = 0

            for item in asset_files:
                try:
                    result = self._process_asset_file(item['path'], item.get('size', 0))
                    if result == 'created':
                        stats['assets_created'] += 1
                    elif result == 'updated':
                        stats['assets_updated'] += 1
                except Exception as e:
                    logger.error(f"Error processing asset {item['path']}: {e}")
                    stats['errors'] += 1

            if stats['assets_found'] > 0:
                logger.info(f"Synced {stats['assets_created']} new, {stats['assets_updated']} updated assets")

            # Log sync
            self._log_sync(repo, branch, stats)

            return stats

        except requests.RequestException as e:
            logger.error(f"GitHub API error: {e}")
            raise

    def _process_github_file(
        self,
        repo: str,
        path: str,
        sha: str,
        headers: Dict,
        current_github_paths: Optional[set] = None,
    ) -> str:
        """Process a single file from GitHub.

        Args:
            repo: GitHub repo in "owner/repo" format
            path: Path to the file in the repo
            sha: Git SHA of the file
            headers: HTTP headers for GitHub API
            current_github_paths: Set of all current file paths in GitHub (for move detection)

        Returns:
            'created', 'updated', or 'skipped'
        """
        # Fetch file content
        content_url = f"https://api.github.com/repos/{repo}/contents/{path}"
        response = requests.get(content_url, headers=headers, timeout=30)
        response.raise_for_status()

        file_data = response.json()
        import base64
        content = base64.b64decode(file_data['content']).decode('utf-8')

        # Parse frontmatter and content
        frontmatter, body = parse_markdown_frontmatter(content)

        # Extract metadata
        title = frontmatter.get('title') or extract_title_from_content(body, path)
        raw_category = frontmatter.get('category') or categorize_from_path(path)
        # Always normalize category to handle plurals, typos, etc.
        category = normalize_category(raw_category)

        # Compute content hash for integrity/deduplication
        content_hash = compute_content_hash(body)

        # Always generate canonical entry_id from normalized category
        # Don't trust frontmatter ID as it may have bad category (e.g., research-topicss)
        entry_id = generate_entry_id(category, title)

        # Check for entry_id collision with different file_path
        # This handles long titles that truncate to the same slug
        collision = self.conn.execute(
            "SELECT id, file_path FROM knowledge_entries WHERE entry_id = ? AND file_path != ?",
            (entry_id, path)
        ).fetchone()
        moved_from_entry_id = None  # Track if this is a file move
        if collision:
            old_file_path = collision['file_path']
            # Check if this is a file move: old path no longer exists in GitHub
            if current_github_paths and old_file_path not in current_github_paths:
                # This is a file move! The old file doesn't exist in GitHub anymore.
                # Update the existing entry's file_path instead of creating a duplicate.
                logger.info(f"File move detected: {old_file_path} -> {path}")
                moved_from_entry_id = collision['id']
            else:
                # True collision: both paths exist, disambiguate with content hash
                logger.info(f"Entry ID collision detected for {path}, disambiguating with content hash")
                entry_id = generate_entry_id(category, title, content_hash)

        # Extract chord fields
        needs_chord = 1 if frontmatter.get('needs_chord') else 0
        chord_name = frontmatter.get('chord_name')
        chord_scope = frontmatter.get('chord_scope')
        chord_id = frontmatter.get('chord_id')
        chord_status = frontmatter.get('chord_status')
        chord_repo = frontmatter.get('chord_repo')

        # Extract source transcript for tracking
        source_transcript = frontmatter.get('source_transcript')

        # Extract task fields from frontmatter
        task_status = frontmatter.get('task_status')
        due_date = frontmatter.get('due_date')

        # Validate task_status if present
        valid_task_statuses = {'pending', 'in_progress', 'blocked', 'done'}
        if task_status and task_status not in valid_task_statuses:
            logger.warning(f"Invalid task_status '{task_status}' in {path}, ignoring")
            task_status = None

        # Extract created/updated dates from frontmatter (use as source of truth)
        created_at = frontmatter.get('created') or frontmatter.get('created_at')
        updated_at = frontmatter.get('updated') or frontmatter.get('updated_at')

        # Parse date strings to ISO format if needed
        created_at = _parse_frontmatter_date(created_at)
        updated_at = _parse_frontmatter_date(updated_at)

        # Extract topic tags - stored as JSON strings
        import json
        domain_tags_raw = frontmatter.get('domain_tags')
        key_phrases_raw = frontmatter.get('key_phrases')

        # Parse JSON arrays or keep as-is if already parsed
        if isinstance(domain_tags_raw, str) and domain_tags_raw.startswith('['):
            try:
                domain_tags = json.dumps(json.loads(domain_tags_raw))
            except json.JSONDecodeError:
                domain_tags = domain_tags_raw
        elif isinstance(domain_tags_raw, list):
            domain_tags = json.dumps(domain_tags_raw)
        else:
            domain_tags = None

        if isinstance(key_phrases_raw, str) and key_phrases_raw.startswith('['):
            try:
                key_phrases = json.dumps(json.loads(key_phrases_raw))
            except json.JSONDecodeError:
                key_phrases = key_phrases_raw
        elif isinstance(key_phrases_raw, list):
            key_phrases = json.dumps(key_phrases_raw)
        else:
            key_phrases = None

        # Extract subfolder from path (e.g., "rocks-cry-outs/chapters/file.md" -> "chapters")
        path_parts = path.split('/')
        if len(path_parts) >= 3:  # folder/subfolder/file.md
            # Check if the middle part is actually a subfolder (not just the category folder)
            subfolder = path_parts[-2] if path_parts[-2] != path_parts[0] else None
        else:
            subfolder = None

        # Check if entry exists by file_path (more reliable than entry_id)
        existing = self.conn.execute(
            "SELECT id, entry_id FROM knowledge_entries WHERE file_path = ?",
            (path,)
        ).fetchone()

        # Handle file move: if we detected a move, treat it as an update to the existing entry
        if moved_from_entry_id and not existing:
            existing = {'id': moved_from_entry_id, 'entry_id': entry_id}

        if existing:
            # Update existing entry (including entry_id if changed)
            # Chord status logic:
            # - If frontmatter sets needs_chord: false → clear chord fields (no chord needed)
            # - If frontmatter sets needs_chord: true AND has explicit chord_status → use frontmatter
            # - If frontmatter sets needs_chord: true AND no chord_status → check DB, preserve 'pending' or 'active'
            #   (GitHub frontmatter update may not have propagated yet after agent approval)
            # For file moves, query by id; for normal updates, query by file_path
            if moved_from_entry_id:
                existing_data = self.conn.execute(
                    "SELECT chord_status, chord_repo, chord_id, subfolder FROM knowledge_entries WHERE id = ?",
                    (moved_from_entry_id,)
                ).fetchone()
            else:
                existing_data = self.conn.execute(
                    "SELECT chord_status, chord_repo, chord_id, subfolder FROM knowledge_entries WHERE file_path = ?",
                    (path,)
                ).fetchone()

            # Chord status logic:
            # 1. Frontmatter has explicit chord_status/chord_repo → use it
            # 2. DB has active/pending chord → preserve it (frontmatter may lag behind)
            # 3. needs_chord=true but no status → ready to be queued
            # 4. needs_chord=false and no chord → nothing needed
            if chord_status or chord_repo:
                # Frontmatter explicitly sets chord info - use it
                final_chord_status = chord_status
                final_chord_repo = chord_repo
                final_chord_id = chord_id
            elif existing_data and existing_data['chord_status'] in ('pending', 'active', 'rejected'):
                # Preserve existing chord status from DB
                # (GitHub frontmatter may not have propagated yet after agent approval/rejection)
                final_chord_status = existing_data['chord_status']
                final_chord_repo = existing_data['chord_repo']
                final_chord_id = existing_data['chord_id']
            elif existing_data and existing_data['chord_repo']:
                # Has a chord_repo but no status - keep it
                final_chord_status = existing_data['chord_status']
                final_chord_repo = existing_data['chord_repo']
                final_chord_id = existing_data['chord_id']
            elif needs_chord:
                # Needs a chord but doesn't have one yet - ready to be queued
                final_chord_status = None
                final_chord_repo = None
                final_chord_id = None
            else:
                # No chord needed and no existing chord - clear fields
                final_chord_status = None
                final_chord_repo = None
                final_chord_id = None

            # For file moves, update file_path and subfolder; otherwise just update by file_path
            if moved_from_entry_id:
                self.conn.execute(
                    """
                    UPDATE knowledge_entries
                    SET entry_id = ?, title = ?, category = ?, content = ?,
                        file_path = ?, subfolder = ?,
                        needs_chord = ?, chord_name = ?, chord_scope = ?,
                        chord_id = ?, chord_status = ?, chord_repo = ?,
                        domain_tags = ?, key_phrases = ?, source_transcript = ?,
                        task_status = ?, due_date = ?, content_hash = ?,
                        updated_at = COALESCE(?, CURRENT_TIMESTAMP)
                    WHERE id = ?
                    """,
                    (entry_id, title, category, body,
                     path, subfolder,
                     needs_chord, chord_name, chord_scope,
                     final_chord_id, final_chord_status, final_chord_repo,
                     domain_tags, key_phrases, source_transcript,
                     task_status, due_date, content_hash, updated_at, moved_from_entry_id)
                )
                self.conn.commit()
                logger.info(f"Updated (file moved): {entry_id} - {title}")
                return 'updated'
            else:
                self.conn.execute(
                    """
                    UPDATE knowledge_entries
                    SET entry_id = ?, title = ?, category = ?, content = ?,
                        subfolder = ?,
                        needs_chord = ?, chord_name = ?, chord_scope = ?,
                        chord_id = ?, chord_status = ?, chord_repo = ?,
                        domain_tags = ?, key_phrases = ?, source_transcript = ?,
                        task_status = ?, due_date = ?, content_hash = ?,
                        updated_at = COALESCE(?, CURRENT_TIMESTAMP)
                    WHERE file_path = ?
                    """,
                    (entry_id, title, category, body,
                     subfolder,
                     needs_chord, chord_name, chord_scope,
                     final_chord_id, final_chord_status, final_chord_repo,
                     domain_tags, key_phrases, source_transcript,
                     task_status, due_date, content_hash, updated_at, path)
                )
                self.conn.commit()
                logger.debug(f"Updated: {entry_id} - {title}" + (f" [task:{task_status}]" if task_status else ""))
                return 'updated'
        else:
            # Create new entry - use frontmatter dates if available
            self.conn.execute(
                """
                INSERT INTO knowledge_entries
                (entry_id, title, category, content, file_path, subfolder,
                 needs_chord, chord_name, chord_scope, chord_id, chord_status, chord_repo,
                 domain_tags, key_phrases, source_transcript,
                 task_status, due_date, content_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), COALESCE(?, CURRENT_TIMESTAMP))
                """,
                (entry_id, title, category, body, path, subfolder,
                 needs_chord, chord_name, chord_scope, chord_id, chord_status, chord_repo,
                 domain_tags, key_phrases, source_transcript,
                 task_status, due_date, content_hash, created_at, updated_at)
            )
            self.conn.commit()
            logger.info(f"Created: {entry_id} - {title}" + (" [needs_chord]" if needs_chord else "") + (f" [task:{task_status}]" if task_status else "") + (f" [subfolder:{subfolder}]" if subfolder else ""))
            return 'created'

    def sync_from_filesystem(self, library_path: str) -> Dict:
        """Sync content from local filesystem.

        Args:
            library_path: Path to Legate.Library directory

        Returns:
            Dict with sync statistics
        """
        library_path = Path(library_path)
        if not library_path.exists():
            raise ValueError(f"Library path does not exist: {library_path}")

        stats = {
            'files_found': 0,
            'entries_created': 0,
            'entries_updated': 0,
            'errors': 0,
            'embeddings_generated': 0,
        }

        # Find all markdown files (exclude README.md and description.md)
        md_files = list(library_path.glob('**/*.md'))
        md_files = [f for f in md_files if f.name != 'README.md' and f.name != 'description.md' and not f.name.startswith('.')]

        stats['files_found'] = len(md_files)
        logger.info(f"Found {len(md_files)} markdown files in {library_path}")

        # Build set of current file paths for move detection
        current_local_paths = {str(f.relative_to(library_path)) for f in md_files}

        for file_path in md_files:
            try:
                result = self._process_local_file(
                    file_path, library_path,
                    current_local_paths=current_local_paths
                )
                if result == 'created':
                    stats['entries_created'] += 1
                elif result == 'updated':
                    stats['entries_updated'] += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                stats['errors'] += 1

        # Generate embeddings only if we created/updated entries
        if self.embedding_service and (stats['entries_created'] > 0 or stats['entries_updated'] > 0):
            stats['embeddings_generated'] = self.embedding_service.generate_missing_embeddings(
                'knowledge', delay=0.1
            )

        # Log sync
        self._log_sync(str(library_path), 'filesystem', stats)

        return stats

    def _process_local_file(
        self,
        file_path: Path,
        base_path: Path,
        current_local_paths: Optional[set] = None,
    ) -> str:
        """Process a single local file.

        Args:
            file_path: Path to the file
            base_path: Base path of the library
            current_local_paths: Set of all current file paths in the library (for move detection)

        Returns:
            'created', 'updated', or 'skipped'
        """
        content = file_path.read_text(encoding='utf-8')
        relative_path = str(file_path.relative_to(base_path))

        # Parse frontmatter and content
        frontmatter, body = parse_markdown_frontmatter(content)

        # Extract metadata
        title = frontmatter.get('title') or extract_title_from_content(body, file_path.name)
        raw_category = frontmatter.get('category') or categorize_from_path(relative_path)
        # Always normalize category to handle plurals, typos, etc.
        category = normalize_category(raw_category)

        # Compute content hash for integrity/deduplication
        content_hash = compute_content_hash(body)

        # Always generate canonical entry_id from normalized category
        # Don't trust frontmatter ID as it may have bad category (e.g., research-topicss)
        entry_id = generate_entry_id(category, title)

        # Check for entry_id collision with different file_path
        # This handles long titles that truncate to the same slug
        collision = self.conn.execute(
            "SELECT id, file_path FROM knowledge_entries WHERE entry_id = ? AND file_path != ?",
            (entry_id, relative_path)
        ).fetchone()
        moved_from_entry_id = None  # Track if this is a file move
        if collision:
            old_file_path = collision['file_path']
            # Check if this is a file move: old path no longer exists in filesystem
            if current_local_paths and old_file_path not in current_local_paths:
                # This is a file move! The old file doesn't exist anymore.
                # Update the existing entry's file_path instead of creating a duplicate.
                logger.info(f"File move detected: {old_file_path} -> {relative_path}")
                moved_from_entry_id = collision['id']
            else:
                # True collision: both paths exist, disambiguate with content hash
                logger.info(f"Entry ID collision detected for {relative_path}, disambiguating with content hash")
                entry_id = generate_entry_id(category, title, content_hash)

        # Extract chord fields
        needs_chord = 1 if frontmatter.get('needs_chord') else 0
        chord_name = frontmatter.get('chord_name')
        chord_scope = frontmatter.get('chord_scope')
        chord_id = frontmatter.get('chord_id')
        chord_status = frontmatter.get('chord_status')
        chord_repo = frontmatter.get('chord_repo')

        # Extract source transcript for tracking
        source_transcript = frontmatter.get('source_transcript')

        # Extract task fields from frontmatter
        task_status = frontmatter.get('task_status')
        due_date = frontmatter.get('due_date')

        # Validate task_status if present
        valid_task_statuses = {'pending', 'in_progress', 'blocked', 'done'}
        if task_status and task_status not in valid_task_statuses:
            logger.warning(f"Invalid task_status '{task_status}' in {relative_path}, ignoring")
            task_status = None

        # Extract created/updated dates from frontmatter (use as source of truth)
        created_at = frontmatter.get('created') or frontmatter.get('created_at')
        updated_at = frontmatter.get('updated') or frontmatter.get('updated_at')

        # Parse date strings to ISO format if needed
        created_at = _parse_frontmatter_date(created_at)
        updated_at = _parse_frontmatter_date(updated_at)

        # Extract topic tags - stored as JSON strings
        import json
        domain_tags_raw = frontmatter.get('domain_tags')
        key_phrases_raw = frontmatter.get('key_phrases')

        if isinstance(domain_tags_raw, str) and domain_tags_raw.startswith('['):
            try:
                domain_tags = json.dumps(json.loads(domain_tags_raw))
            except json.JSONDecodeError:
                domain_tags = domain_tags_raw
        elif isinstance(domain_tags_raw, list):
            domain_tags = json.dumps(domain_tags_raw)
        else:
            domain_tags = None

        if isinstance(key_phrases_raw, str) and key_phrases_raw.startswith('['):
            try:
                key_phrases = json.dumps(json.loads(key_phrases_raw))
            except json.JSONDecodeError:
                key_phrases = key_phrases_raw
        elif isinstance(key_phrases_raw, list):
            key_phrases = json.dumps(key_phrases_raw)
        else:
            key_phrases = None

        # Extract subfolder from path (e.g., "rocks-cry-outs/chapters/file.md" -> "chapters")
        path_parts = relative_path.split('/')
        if len(path_parts) >= 3:  # folder/subfolder/file.md
            # Check if the middle part is actually a subfolder (not just the category folder)
            subfolder = path_parts[-2] if path_parts[-2] != path_parts[0] else None
        else:
            subfolder = None

        # Check if entry exists by file_path (more reliable than entry_id)
        existing = self.conn.execute(
            "SELECT id, entry_id FROM knowledge_entries WHERE file_path = ?",
            (relative_path,)
        ).fetchone()

        # Handle file move: if we detected a move, treat it as an update to the existing entry
        if moved_from_entry_id and not existing:
            existing = {'id': moved_from_entry_id, 'entry_id': entry_id}

        if existing:
            # Same chord status logic as GitHub sync
            # For file moves, query by id; for normal updates, query by file_path
            if moved_from_entry_id:
                existing_data = self.conn.execute(
                    "SELECT chord_status, chord_repo, chord_id, subfolder FROM knowledge_entries WHERE id = ?",
                    (moved_from_entry_id,)
                ).fetchone()
            else:
                existing_data = self.conn.execute(
                    "SELECT chord_status, chord_repo, chord_id, subfolder FROM knowledge_entries WHERE file_path = ?",
                    (relative_path,)
                ).fetchone()

            # Chord status logic - preserve existing chord relationships
            if chord_status or chord_repo:
                # Frontmatter explicitly sets chord info - use it
                final_chord_status = chord_status
                final_chord_repo = chord_repo
                final_chord_id = chord_id
            elif existing_data and existing_data['chord_status'] in ('pending', 'active', 'rejected'):
                # Preserve existing chord status from DB
                final_chord_status = existing_data['chord_status']
                final_chord_repo = existing_data['chord_repo']
                final_chord_id = existing_data['chord_id']
            elif existing_data and existing_data['chord_repo']:
                # Has a chord_repo but no status - keep it
                final_chord_status = existing_data['chord_status']
                final_chord_repo = existing_data['chord_repo']
                final_chord_id = existing_data['chord_id']
            elif needs_chord:
                # Needs a chord but doesn't have one yet
                final_chord_status = None
                final_chord_repo = None
                final_chord_id = None
            else:
                # No chord needed and no existing chord
                final_chord_status = None
                final_chord_repo = None
                final_chord_id = None

            # For file moves, update file_path and subfolder; otherwise just update by file_path
            if moved_from_entry_id:
                self.conn.execute(
                    """
                    UPDATE knowledge_entries
                    SET entry_id = ?, title = ?, category = ?, content = ?,
                        file_path = ?, subfolder = ?,
                        needs_chord = ?, chord_name = ?, chord_scope = ?,
                        chord_id = ?, chord_status = ?, chord_repo = ?,
                        domain_tags = ?, key_phrases = ?, source_transcript = ?,
                        task_status = ?, due_date = ?, content_hash = ?,
                        updated_at = COALESCE(?, CURRENT_TIMESTAMP)
                    WHERE id = ?
                    """,
                    (entry_id, title, category, body,
                     relative_path, subfolder,
                     needs_chord, chord_name, chord_scope,
                     final_chord_id, final_chord_status, final_chord_repo,
                     domain_tags, key_phrases, source_transcript,
                     task_status, due_date, content_hash, updated_at, moved_from_entry_id)
                )
                self.conn.commit()
                logger.info(f"Updated (file moved): {entry_id} - {title}")
                return 'updated'
            else:
                self.conn.execute(
                    """
                    UPDATE knowledge_entries
                    SET entry_id = ?, title = ?, category = ?, content = ?,
                        subfolder = ?,
                        needs_chord = ?, chord_name = ?, chord_scope = ?,
                        chord_id = ?, chord_status = ?, chord_repo = ?,
                        domain_tags = ?, key_phrases = ?, source_transcript = ?,
                        task_status = ?, due_date = ?, content_hash = ?,
                        updated_at = COALESCE(?, CURRENT_TIMESTAMP)
                    WHERE file_path = ?
                    """,
                    (entry_id, title, category, body,
                     subfolder,
                     needs_chord, chord_name, chord_scope,
                     final_chord_id, final_chord_status, final_chord_repo,
                     domain_tags, key_phrases, source_transcript,
                     task_status, due_date, content_hash, updated_at, relative_path)
                )
                self.conn.commit()
                logger.debug(f"Updated: {entry_id} - {title}" + (f" [task:{task_status}]" if task_status else ""))
                return 'updated'
        else:
            # Create new entry - use frontmatter dates if available
            self.conn.execute(
                """
                INSERT INTO knowledge_entries
                (entry_id, title, category, content, file_path, subfolder,
                 needs_chord, chord_name, chord_scope, chord_id, chord_status, chord_repo,
                 domain_tags, key_phrases, source_transcript,
                 task_status, due_date, content_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), COALESCE(?, CURRENT_TIMESTAMP))
                """,
                (entry_id, title, category, body, relative_path, subfolder,
                 needs_chord, chord_name, chord_scope, chord_id, chord_status, chord_repo,
                 domain_tags, key_phrases, source_transcript,
                 task_status, due_date, content_hash, created_at, updated_at)
            )
            self.conn.commit()
            logger.info(f"Created: {entry_id} - {title}" + (" [needs_chord]" if needs_chord else "") + (f" [task:{task_status}]" if task_status else "") + (f" [subfolder:{subfolder}]" if subfolder else ""))
            return 'created'

    def _log_sync(self, source: str, branch: str, stats: Dict):
        """Log sync operation to database."""
        self.conn.execute(
            """
            INSERT INTO sync_log (source, commit_sha, entries_synced, status)
            VALUES (?, ?, ?, ?)
            """,
            (
                source,
                branch,
                stats['entries_created'] + stats['entries_updated'],
                'success' if stats['errors'] == 0 else 'partial'
            )
        )
        self.conn.commit()

    def get_sync_status(self) -> Dict:
        """Get the latest sync status."""
        row = self.conn.execute(
            """
            SELECT source, commit_sha, entries_synced, status, synced_at
            FROM sync_log
            ORDER BY synced_at DESC
            LIMIT 1
            """
        ).fetchone()

        if row:
            return dict(row)
        return {'status': 'never_synced'}

    def sync_assets_from_github(
        self,
        repo: str = "bobbyhiddn/Legate.Library",
        token: Optional[str] = None,
        branch: str = "main",
    ) -> Dict:
        """Sync assets from GitHub repository.

        Scans for files in */assets/ folders and indexes them in the database.
        Does not download the actual files - just tracks metadata.

        Args:
            repo: GitHub repo in "owner/repo" format
            token: GitHub PAT for API access
            branch: Branch to sync from

        Returns:
            Dict with sync statistics
        """
        import mimetypes

        if not token:
            raise ValueError("GitHub token required for sync")

        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.github+json',
        }

        stats = {
            'assets_found': 0,
            'assets_created': 0,
            'assets_updated': 0,
            'errors': 0,
        }

        try:
            # Get repository tree
            tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
            response = requests.get(tree_url, headers=headers, timeout=30)
            response.raise_for_status()
            tree_data = response.json()

            # Filter for files in */assets/ folders
            # Pattern: category/assets/filename (not .gitkeep)
            asset_files = [
                item for item in tree_data.get('tree', [])
                if item['type'] == 'blob'
                and '/assets/' in item['path']
                and not item['path'].endswith('.gitkeep')
                and not item['path'].startswith('.')
            ]

            stats['assets_found'] = len(asset_files)
            logger.info(f"Found {len(asset_files)} asset files in {repo}")

            for item in asset_files:
                try:
                    result = self._process_asset_file(item['path'], item.get('size', 0))
                    if result == 'created':
                        stats['assets_created'] += 1
                    elif result == 'updated':
                        stats['assets_updated'] += 1
                except Exception as e:
                    logger.error(f"Error processing asset {item['path']}: {e}")
                    stats['errors'] += 1

            return stats

        except requests.RequestException as e:
            logger.error(f"GitHub API error during asset sync: {e}")
            raise

    def _process_asset_file(self, file_path: str, file_size: int) -> str:
        """Process a single asset file from GitHub tree.

        Args:
            file_path: Path like "concept/assets/image-abc123.png"
            file_size: File size in bytes

        Returns:
            'created', 'updated', or 'skipped'
        """
        import mimetypes
        import secrets

        # Parse the path to extract category and filename
        # Expected format: category/assets/filename.ext
        parts = Path(file_path).parts
        if len(parts) < 3 or parts[-2] != 'assets':
            logger.debug(f"Skipping non-asset path: {file_path}")
            return 'skipped'

        category = parts[0]
        filename = parts[-1]

        # Determine MIME type from filename
        mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'

        # Check if asset already exists by file_path
        existing = self.conn.execute(
            "SELECT asset_id FROM library_assets WHERE file_path = ?",
            (file_path,)
        ).fetchone()

        if existing:
            # Update existing asset
            self.conn.execute(
                """
                UPDATE library_assets
                SET file_size = ?, mime_type = ?, updated_at = CURRENT_TIMESTAMP
                WHERE file_path = ?
                """,
                (file_size, mime_type, file_path)
            )
            self.conn.commit()
            logger.debug(f"Updated asset: {file_path}")
            return 'updated'
        else:
            # Create new asset record
            asset_id = f"asset-{secrets.token_hex(6)}"
            self.conn.execute(
                """
                INSERT INTO library_assets
                (asset_id, category, filename, file_path, mime_type, file_size)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (asset_id, category, filename, file_path, mime_type, file_size)
            )
            self.conn.commit()
            logger.info(f"Created asset: {asset_id} -> {file_path}")
            return 'created'
