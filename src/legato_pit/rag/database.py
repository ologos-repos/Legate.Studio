"""
SQLite Database Setup for Legato.Pit

Version: 2026-01-10-v2 (chord ontology update)

Split database architecture:
- legato.db: Knowledge entries, embeddings, sync tracking (RAG data)
- agents.db: Agent queue for project spawns
- chat.db: Chat sessions and messages

Archive databases (for future):
- agents_archive.db: Completed/old agent records
- chat_archive.db: Old chat sessions
"""

import os
import sqlite3
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Track which databases have been initialized to avoid re-running migrations
# This prevents lock contention when multiple threads/workers call init_db
_initialized_dbs: set = set()
_init_lock = threading.Lock()

# Default database path - can be overridden by FLY_VOLUME_PATH
DEFAULT_DB_DIR = Path(__file__).parent.parent.parent.parent / "data"
FLY_VOLUME_PATH = os.environ.get("FLY_VOLUME_PATH", "/data")


def get_db_dir() -> Path:
    """Get the database directory, preferring Fly volume if available."""
    if os.path.exists(FLY_VOLUME_PATH) and os.access(FLY_VOLUME_PATH, os.W_OK):
        return Path(FLY_VOLUME_PATH)

    # Fallback to local data directory
    DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DB_DIR


def get_db_path(db_name: str = "legato.db") -> Path:
    """Get path for a specific database file."""
    return get_db_dir() / db_name


def get_user_db_path(user_id: str) -> Path:
    """Get path for a user-specific database.

    In multi-tenant mode, each user gets their own database file.
    This ensures complete data isolation between users.

    Args:
        user_id: The user's unique ID

    Returns:
        Path to user's database file (e.g., legato_abc123.db)
    """
    # Sanitize user_id to prevent path traversal
    safe_id = "".join(c for c in user_id if c.isalnum() or c in '-_')
    return get_db_dir() / f"legato_{safe_id}.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a database connection with proper settings."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Enable foreign keys and WAL mode for better concurrency
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    # CRITICAL: Use FULL sync to ensure data is durably written to disk
    # before commit returns. This is essential for multi-worker deployments
    # on Fly.io where workers need to see each other's writes immediately.
    # NORMAL can leave data in OS buffers, causing visibility issues.
    conn.execute("PRAGMA synchronous = FULL")
    # Wait up to 30 seconds for locks instead of failing immediately
    conn.execute("PRAGMA busy_timeout = 30000")

    return conn


def checkpoint_all_databases():
    """Checkpoint all databases to ensure WAL changes are written to disk.

    This is important for data persistence when running on Fly.io with
    auto_stop_machines enabled.
    """
    for db_name in ['legato.db', 'agents.db', 'chat.db']:
        try:
            path = get_db_path(db_name)
            if path.exists():
                conn = sqlite3.connect(str(path))
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.close()
                logger.info(f"Checkpointed {db_name}")
        except Exception as e:
            logger.error(f"Failed to checkpoint {db_name}: {e}")


# ============ Legato DB (Knowledge/Embeddings) ============

def init_db(db_path: Optional[Path] = None, user_id: Optional[str] = None) -> sqlite3.Connection:
    """Initialize legato database with knowledge entries and embeddings.

    In multi-tenant mode, pass user_id to get a user-specific database.
    In single-tenant mode, uses the shared legato.db.

    This is the main RAG database containing:
    - knowledge_entries: Library knowledge artifacts
    - project_entries: Lab project metadata
    - embeddings: Vector embeddings for similarity search
    - transcript_hashes: Deduplication fingerprints
    - sync_log: Sync tracking
    - pipeline_runs: Pipeline run tracking

    Args:
        db_path: Optional explicit path (overrides user_id)
        user_id: User ID for multi-tenant isolation
    """
    if db_path:
        path = db_path
    elif user_id:
        path = get_user_db_path(user_id)
    else:
        path = get_db_path("legato.db")

    conn = get_connection(path)

    # Check if this database has already been initialized in this process
    # This prevents running migrations multiple times and causing lock contention
    db_key = str(path)
    with _init_lock:
        if db_key in _initialized_dbs:
            return conn
        _initialized_dbs.add(db_key)

    cursor = conn.cursor()

    # Knowledge entries from Library
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            category TEXT,
            content TEXT NOT NULL,
            source_thread TEXT,
            source_transcript TEXT,
            file_path TEXT,
            needs_chord INTEGER DEFAULT 0,
            chord_name TEXT,
            chord_scope TEXT,
            chord_id TEXT,
            chord_status TEXT,
            chord_repo TEXT,
            domain_tags TEXT,
            key_phrases TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add chord columns to existing tables (migration)
    try:
        cursor.execute("ALTER TABLE knowledge_entries ADD COLUMN needs_chord INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        cursor.execute("ALTER TABLE knowledge_entries ADD COLUMN chord_name TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE knowledge_entries ADD COLUMN chord_scope TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE knowledge_entries ADD COLUMN chord_id TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE knowledge_entries ADD COLUMN chord_status TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE knowledge_entries ADD COLUMN chord_repo TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE knowledge_entries ADD COLUMN domain_tags TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE knowledge_entries ADD COLUMN key_phrases TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE knowledge_entries ADD COLUMN subfolder TEXT")
    except sqlite3.OperationalError:
        pass

    # Project entries from Lab
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS project_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT UNIQUE NOT NULL,
            repo_name TEXT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'active',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Vector embeddings (supports multiple providers)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER NOT NULL,
            entry_type TEXT NOT NULL DEFAULT 'knowledge',
            embedding BLOB NOT NULL,
            vector_version TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(entry_id, entry_type, vector_version)
        )
    """)

    # Transcript fingerprints for deduplication
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcript_hashes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE NOT NULL,
            transcript_id TEXT,
            thread_count INTEGER,
            processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Sync tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sync_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            commit_sha TEXT,
            entries_synced INTEGER DEFAULT 0,
            status TEXT,
            synced_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Pipeline run tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            details TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id, stage)
        )
    """)

    # User-defined categories
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'default',
            name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            description TEXT,
            folder_name TEXT NOT NULL,
            color TEXT DEFAULT '#6366f1',
            sort_order INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, name)
        )
    """)

    # Add color column if it doesn't exist (migration for existing databases)
    try:
        cursor.execute("ALTER TABLE user_categories ADD COLUMN color TEXT DEFAULT '#6366f1'")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: Fix NULL colors for existing categories
    # This ensures existing categories get their proper default colors
    default_color_map = {cat[0]: cat[5] for cat in DEFAULT_CATEGORIES}
    for name, color in default_color_map.items():
        cursor.execute(
            "UPDATE user_categories SET color = ? WHERE name = ? AND (color IS NULL OR color = '')",
            (color, name)
        )
    # For any other categories without colors, set a default
    cursor.execute(
        "UPDATE user_categories SET color = '#6366f1' WHERE color IS NULL OR color = ''"
    )

    # Migration: Fix incorrectly pluralized folder_names (e.g., 'prisms' → 'prism')
    # Old code naively added 's' to category names for folder_name. DB defaults use singular.
    # Fix: where folder_name == name + 's', set folder_name = name (singular).
    cursor.execute("""
        UPDATE user_categories
        SET folder_name = name, updated_at = CURRENT_TIMESTAMP
        WHERE folder_name = name || 's'
          AND NOT (name LIKE '%s')
    """)

    # OAuth clients (Dynamic Client Registration - RFC 7591)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oauth_clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT UNIQUE NOT NULL,
            client_name TEXT,
            redirect_uris TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # OAuth authorization codes (short-lived, one-time use)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oauth_auth_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            client_id TEXT NOT NULL,
            github_user_id INTEGER NOT NULL,
            github_login TEXT NOT NULL,
            code_challenge TEXT,
            scope TEXT DEFAULT 'mcp:read mcp:write',
            redirect_uri TEXT NOT NULL,
            expires_at DATETIME NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # OAuth sessions (for refresh tokens)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oauth_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL,
            github_user_id INTEGER NOT NULL,
            github_login TEXT NOT NULL,
            refresh_token TEXT UNIQUE,
            expires_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Note links for explicit relationships between notes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS note_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_entry_id TEXT NOT NULL,
            target_entry_id TEXT NOT NULL,
            link_type TEXT DEFAULT 'related',
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT DEFAULT 'mcp-claude',
            UNIQUE(source_entry_id, target_entry_id, link_type)
        )
    """)

    # Library assets (images, files) stored in category assets folders
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS library_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            mime_type TEXT,
            file_size INTEGER,
            alt_text TEXT,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_library_assets_category ON library_assets(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_library_assets_asset_id ON library_assets(asset_id)")

    # Pipeline processing jobs for motif/transcript processing
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL,
            job_type TEXT NOT NULL DEFAULT 'motif',
            status TEXT NOT NULL DEFAULT 'pending',
            input_content TEXT NOT NULL,
            input_format TEXT DEFAULT 'markdown',
            result_entry_ids TEXT,
            error_message TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME
        )
    """)

    # Migration: Add task_status column to knowledge_entries if it doesn't exist
    try:
        cursor.execute("ALTER TABLE knowledge_entries ADD COLUMN task_status TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: Add due_date column to knowledge_entries if it doesn't exist
    try:
        cursor.execute("ALTER TABLE knowledge_entries ADD COLUMN due_date DATE")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: Add content_hash column for deduplication and integrity verification
    try:
        cursor.execute("ALTER TABLE knowledge_entries ADD COLUMN content_hash TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Create indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_entries(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_entry_id ON knowledge_entries(entry_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_needs_chord ON knowledge_entries(needs_chord, chord_status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_task_status ON knowledge_entries(task_status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_entry ON embeddings(entry_id, entry_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcript_hash ON transcript_hashes(content_hash)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_categories_user ON user_categories(user_id, is_active)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_note_links_source ON note_links(source_entry_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_note_links_target ON note_links(target_entry_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_jobs_status ON processing_jobs(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_content_hash ON knowledge_entries(content_hash)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_oauth_clients ON oauth_clients(client_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_oauth_auth_codes ON oauth_auth_codes(code)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_oauth_sessions ON oauth_sessions(refresh_token)")

    # ============ Job Queue Enhancements for Motif Processing ============

    # Migration: Add user_id to processing_jobs for multi-tenant support
    for column, col_type in [
        ('user_id', 'TEXT'),
        ('current_stage', 'TEXT'),
        ('progress_pct', 'INTEGER DEFAULT 0'),
        ('threads_total', 'INTEGER DEFAULT 0'),
        ('threads_completed', 'INTEGER DEFAULT 0'),
        ('threads_failed', 'INTEGER DEFAULT 0'),
        ('worker_id', 'TEXT'),
        ('locked_until', 'DATETIME'),
        ('started_at', 'DATETIME'),
        ('retry_count', 'INTEGER DEFAULT 0'),
        ('source_id', 'TEXT'),
    ]:
        try:
            cursor.execute(f"ALTER TABLE processing_jobs ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Processing threads table - tracks individual threads within a job
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            thread_index INTEGER NOT NULL,
            raw_content TEXT NOT NULL,

            -- Classification results
            category TEXT,
            title TEXT,
            description TEXT,
            domain_tags TEXT,
            key_phrases TEXT,
            needs_chord INTEGER DEFAULT 0,
            chord_name TEXT,
            chord_scope TEXT,

            -- Correlation results
            correlation_action TEXT,
            correlation_entry_id TEXT,
            correlation_score REAL,

            -- Extraction results
            extracted_markdown TEXT,
            entry_id TEXT,
            file_path TEXT,

            -- Status tracking
            status TEXT DEFAULT 'pending',
            error_message TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(job_id, thread_id),
            FOREIGN KEY (job_id) REFERENCES processing_jobs(job_id)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_threads_job ON processing_threads(job_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_threads_status ON processing_threads(job_id, status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_jobs_locked ON processing_jobs(status, locked_until)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_jobs_user ON processing_jobs(user_id)")

    # ============ Multi-Tenant Tables ============

    # Core user record (authenticated via GitHub)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            github_id INTEGER UNIQUE NOT NULL,
            github_login TEXT NOT NULL,
            email TEXT,
            tier TEXT DEFAULT 'free',
            tier_expires_at DATETIME,
            refresh_token_encrypted BLOB,
            settings TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration: add name column to users
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN name TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add avatar_url column to users
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add refresh_token_encrypted to users if missing
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN refresh_token_encrypted BLOB")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add oauth_token_encrypted to users for repo management
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN oauth_token_encrypted BLOB")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add oauth_token_expires_at to users
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN oauth_token_expires_at DATETIME")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add has_copilot flag to users (for gating Chords/Agents features)
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN has_copilot INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add copilot_checked_at timestamp to users
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN copilot_checked_at DATETIME")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add Stripe billing columns
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN stripe_customer_id TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE users ADD COLUMN stripe_subscription_id TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add trial tracking
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN trial_started_at DATETIME")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add beta flag for beta testers (get managed tier free)
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN is_beta INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # GitHub App installations (per-user scoped tokens)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS github_app_installations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            installation_id INTEGER UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            account_login TEXT,
            account_type TEXT,
            access_token_encrypted BLOB,
            token_expires_at DATETIME,
            permissions TEXT,
            suspended INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    # Migration: rename github_login to account_login if needed
    try:
        cursor.execute("ALTER TABLE github_app_installations ADD COLUMN account_login TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE github_app_installations ADD COLUMN account_type TEXT")
    except sqlite3.OperationalError:
        pass

    # User's API keys (encrypted, they provide for BYK tier)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            key_encrypted BLOB NOT NULL,
            key_hint TEXT,
            is_valid INTEGER DEFAULT 1,
            last_used_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, provider)
        )
    """)

    # Migration: add updated_at to user_api_keys
    try:
        cursor.execute("ALTER TABLE user_api_keys ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP")
    except sqlite3.OperationalError:
        pass

    # User's connected repos (Library, Conduct)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_repos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            repo_type TEXT NOT NULL,
            repo_full_name TEXT NOT NULL,
            installation_id INTEGER,
            is_primary INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, repo_type)
        )
    """)

    # Migration: add updated_at to user_repos
    try:
        cursor.execute("ALTER TABLE user_repos ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP")
    except sqlite3.OperationalError:
        pass

    # Audit log for security and debugging
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            action TEXT NOT NULL,
            resource_type TEXT,
            resource_id TEXT,
            details TEXT,
            ip_address TEXT,
            user_agent TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Usage metering for managed tier
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_meters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            period TEXT NOT NULL,
            meter_type TEXT NOT NULL,
            quantity INTEGER DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, period, meter_type)
        )
    """)

    # Usage events for detailed tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT NOT NULL,
            provider TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER,
            cost_microdollars INTEGER,
            metadata TEXT
        )
    """)

    # Pending repo additions (for retry when token expired)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pending_repo_additions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            repo_id INTEGER NOT NULL,
            repo_full_name TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, repo_id)
        )
    """)

    # ============ System Configuration Table (canonical schema) ============

    # Canonical system_config table with BOTH created_at AND updated_at.
    # This is the authoritative CREATE — crypto.py's CREATE TABLE is kept for
    # backward-compat isolation but this migration ensures the column set is complete.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration: add updated_at if this table was created by the old crypto.py schema
    # (which only had created_at). Safe no-op if column already exists.
    try:
        cursor.execute("ALTER TABLE system_config ADD COLUMN updated_at TEXT DEFAULT CURRENT_TIMESTAMP")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Multi-tenant indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_github ON users(github_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_login ON users(github_login)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_installations_user ON github_app_installations(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_api_keys_user ON user_api_keys(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_repos_user ON user_repos(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id, created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action, created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_meters_user ON usage_meters(user_id, period)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_user ON usage_events(user_id, timestamp)")

    conn.commit()
    logger.debug(f"Legato database initialized at {path}")

    return conn


# ============ Category Helpers ============

# (name, display_name, description, folder_name, sort_order, color)
# Using singular folder names for consistency with category names
DEFAULT_CATEGORIES = [
    ('epiphany', 'Epiphany', 'Major breakthrough or insight - genuine "aha" moments', 'epiphany', 1, '#f59e0b'),      # Amber
    ('concept', 'Concept', 'Technical definition, explanation, or implementation idea', 'concept', 2, '#6366f1'),     # Indigo
    ('reflection', 'Reflection', 'Personal thought, observation, or musing', 'reflection', 3, '#8b5cf6'),             # Violet
    ('glimmer', 'Glimmer', 'A captured moment - photographing a feeling. Poetic, evocative, sensory', 'glimmer', 4, '#ec4899'),  # Pink
    ('reminder', 'Reminder', 'Note to self about something to remember', 'reminder', 5, '#14b8a6'),                   # Teal
    ('worklog', 'Worklog', 'Summary of work already completed', 'worklog', 6, '#64748b'),                             # Slate
]


def seed_default_categories(conn: sqlite3.Connection, user_id: str = 'default') -> int:
    """Seed default categories for a user if they don't exist.

    Args:
        conn: Database connection
        user_id: User identifier

    Returns:
        Number of categories created
    """
    cursor = conn.cursor()
    created = 0

    for name, display_name, description, folder_name, sort_order, color in DEFAULT_CATEGORIES:
        try:
            cursor.execute("""
                INSERT INTO user_categories (user_id, name, display_name, description, folder_name, sort_order, color)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, name, display_name, description, folder_name, sort_order, color))
            created += 1
        except sqlite3.IntegrityError:
            pass  # Already exists

    conn.commit()
    if created > 0:
        logger.info(f"Seeded {created} default categories for user {user_id}")
    return created


def get_user_categories(conn: sqlite3.Connection, user_id: str = 'default') -> list[dict]:
    """Get all active categories for a user, seeding defaults if needed.

    Args:
        conn: Database connection
        user_id: User identifier

    Returns:
        List of category dictionaries with id, name, display_name, description, folder_name, sort_order, color
    """
    # Check if user has any categories
    count = conn.execute(
        "SELECT COUNT(*) FROM user_categories WHERE user_id = ? AND is_active = 1",
        (user_id,)
    ).fetchone()[0]

    if count == 0:
        seed_default_categories(conn, user_id)

    rows = conn.execute("""
        SELECT id, name, display_name, description, folder_name, sort_order, color
        FROM user_categories
        WHERE user_id = ? AND is_active = 1
        ORDER BY sort_order, name
    """, (user_id,)).fetchall()

    return [dict(row) for row in rows]


# ============ Agents DB ============

def init_agents_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize agents.db for agent queue management.

    Contains:
    - agent_queue: Pending project spawns awaiting approval
    """
    path = db_path or get_db_path("agents.db")
    conn = get_connection(path)

    # Check if already initialized to avoid lock contention
    db_key = str(path)
    with _init_lock:
        if db_key in _initialized_dbs:
            return conn
        _initialized_dbs.add(db_key)

    cursor = conn.cursor()

    # Agent queue for pending project spawns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            queue_id TEXT UNIQUE NOT NULL,
            project_name TEXT NOT NULL,
            project_type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            signal_json TEXT NOT NULL,
            tasker_body TEXT NOT NULL,
            source_transcript TEXT,
            related_entry_id TEXT,
            status TEXT DEFAULT 'pending',
            approved_by TEXT,
            approved_at DATETIME,
            spawn_result TEXT,
            user_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes (for columns that always exist)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_queue_status ON agent_queue(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_queue_source ON agent_queue(source_transcript)")

    # Migration: Add missing columns if they don't exist
    cursor.execute("PRAGMA table_info(agent_queue)")
    columns = [row[1] for row in cursor.fetchall()]

    if 'comments' not in columns:
        cursor.execute("ALTER TABLE agent_queue ADD COLUMN comments TEXT DEFAULT '[]'")
        logger.info("Added comments column to agent_queue")

    if 'user_id' not in columns:
        cursor.execute("ALTER TABLE agent_queue ADD COLUMN user_id TEXT")
        logger.info("Added user_id column to agent_queue")

    # Create user_id index after migration ensures column exists
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_queue_user ON agent_queue(user_id)")

    # Sync history to track processed workflow runs (persists even when queue is cleared)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sync_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            item_id TEXT NOT NULL,
            processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id, item_id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sync_history_run ON sync_history(run_id)")

    # Background jobs for async operations (reset, bulk sync, etc.)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS background_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL,
            job_type TEXT NOT NULL,
            user_id TEXT,
            status TEXT DEFAULT 'pending',
            progress_current INTEGER DEFAULT 0,
            progress_total INTEGER DEFAULT 0,
            progress_message TEXT,
            result_json TEXT,
            error_message TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            started_at DATETIME,
            completed_at DATETIME
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_background_jobs_user ON background_jobs(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_background_jobs_status ON background_jobs(status)")

    conn.commit()
    logger.debug(f"Agents database initialized at {path}")

    return conn


# ============ Chat DB ============

def init_chat_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize chat.db for chat sessions and messages.

    Contains:
    - chat_sessions: User chat sessions
    - chat_messages: Individual messages with context
    """
    path = db_path or get_db_path("chat.db")
    conn = get_connection(path)

    # Check if already initialized to avoid lock contention
    db_key = str(path)
    with _init_lock:
        if db_key in _initialized_dbs:
            return conn
        _initialized_dbs.add(db_key)

    cursor = conn.cursor()

    # Chat sessions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            title TEXT,
            user_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Chat messages with context tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            context_used TEXT,
            model_used TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id)")

    conn.commit()
    logger.debug(f"Chat database initialized at {path}")

    return conn


def backup_to_tigris(conn: sqlite3.Connection, bucket_name: str, db_name: str = "legato") -> bool:
    """Backup a single database to Tigris S3-compatible storage.

    Args:
        conn: Database connection to backup
        bucket_name: Tigris bucket name
        db_name: Name prefix for backup file (e.g., 'legato', 'agents', 'chat')
    """
    import boto3
    from datetime import datetime

    try:
        # Get Tigris credentials from environment
        s3 = boto3.client(
            's3',
            endpoint_url=os.environ.get('AWS_ENDPOINT_URL_S3'),
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'auto')
        )

        # Create backup
        db_dir = get_db_dir()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = db_dir / f"{db_name}_backup_{timestamp}.db"

        # SQLite online backup
        backup_conn = sqlite3.connect(str(backup_path))
        conn.backup(backup_conn)
        backup_conn.close()

        # Upload to Tigris
        key = f"backups/{backup_path.name}"
        s3.upload_file(str(backup_path), bucket_name, key)

        # Clean up local backup
        backup_path.unlink()

        logger.info(f"Database backed up to Tigris: {key}")
        return True

    except Exception as e:
        logger.error(f"Backup to Tigris failed: {e}")
        return False


def backup_all_to_tigris(bucket_name: str) -> dict:
    """Backup all databases to Tigris S3-compatible storage.

    Returns:
        Dict with backup status for each database
    """
    results = {}

    # Backup each database
    for db_name, init_func in [
        ("legato", init_db),
        ("agents", init_agents_db),
        ("chat", init_chat_db),
    ]:
        try:
            conn = init_func()
            success = backup_to_tigris(conn, bucket_name, db_name)
            results[db_name] = {"success": success}
        except Exception as e:
            results[db_name] = {"success": False, "error": str(e)}

    return results


# ============ Flask Request-Scoped Database Access ============

def get_user_legato_db():
    """Get the legato database for the current authenticated user.

    SECURITY: Each user MUST have their own isolated database.
    This function requires a valid user_id from:
    - Flask session (web routes)
    - g.mcp_user (MCP routes)

    Must be called within a Flask request context.

    Returns:
        sqlite3.Connection to user's legato database

    Raises:
        ValueError: If no user_id is present (security violation)
    """
    from flask import g, session

    # Check if we already have a connection for this request
    if 'user_legato_db' in g:
        return g.user_legato_db

    user_id = None

    # Try session first (web routes)
    user = session.get('user')
    if user and user.get('user_id'):
        user_id = user['user_id']

    # Try MCP context (API routes)
    if not user_id and hasattr(g, 'mcp_user') and g.mcp_user:
        mcp_user_id = g.mcp_user.get('user_id')
        github_id = g.mcp_user.get('github_id')

        # CRITICAL: Look up canonical user_id by github_id to prevent stale JWT issues
        # The JWT may contain an old user_id if the user record was recreated
        if github_id:
            shared_db = init_db()  # Shared database has users table
            canonical = shared_db.execute(
                "SELECT user_id FROM users WHERE github_id = ?", (github_id,)
            ).fetchone()
            if canonical:
                canonical_user_id = canonical['user_id']
                if canonical_user_id != mcp_user_id:
                    # Expected when token was issued with old user_id - resolved correctly
                    logger.debug(f"MCP user_id resolved: jwt={mcp_user_id} -> canonical={canonical_user_id}")
                user_id = canonical_user_id
            else:
                # No user found by github_id, fall back to JWT user_id
                user_id = mcp_user_id
        else:
            user_id = mcp_user_id

    # SECURITY: Always require user_id for database access
    if not user_id:
        logger.error("SECURITY: Attempted database access without user_id")
        raise ValueError("Database access requires authenticated user with user_id")

    g.user_legato_db = init_db(user_id=user_id)

    # CRITICAL: Ensure visibility of recent writes from other connections
    # In WAL mode, committed changes should be visible, but we need to ensure
    # no stale transaction is holding an old snapshot.
    # 1. Rollback any implicit transaction that might have started
    # 2. Use RESTART checkpoint which waits for readers and checkpoints all frames
    try:
        g.user_legato_db.rollback()  # Clear any implicit transaction
    except Exception:
        pass  # Ignore if no transaction active
    g.user_legato_db.execute("PRAGMA wal_checkpoint(RESTART)")

    return g.user_legato_db


def delete_user_data(user_id: str) -> dict:
    """Delete all data for a user.

    In multi-tenant mode, this deletes the user's database file.
    Also cleans up auth records.

    Args:
        user_id: The user's unique ID

    Returns:
        Dict with deletion status
    """
    results = {'user_id': user_id, 'deleted': []}

    # Delete user's database file if it exists
    user_db_path = get_user_db_path(user_id)
    if user_db_path.exists():
        # Also delete WAL and SHM files
        for suffix in ['', '-wal', '-shm']:
            p = Path(str(user_db_path) + suffix)
            if p.exists():
                p.unlink()
                results['deleted'].append(str(p.name))

    logger.info(f"Deleted user data for {user_id}: {results}")
    return results
