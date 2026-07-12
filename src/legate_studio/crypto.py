"""
Encryption utilities for multi-tenant data protection.

Implements a key hierarchy:
- Master Key (env var LEGATE_MASTER_KEY — preferred, or DB fallback with loud warning)
  └── Per-User DEK (PBKDF2-derived from master key + random per-user salt)
        └── Encrypts: API keys, sensitive preferences

All user-specific sensitive data is encrypted with a key derived from
a random per-user salt, so even database access doesn't expose other users' data.

## Key management

Set LEGATE_MASTER_KEY to a URL-safe base64-encoded 32-byte key in your environment.
Generate one with:

    python -m legate_studio.crypto

If LEGATE_MASTER_KEY is not set, the master key is loaded from (or generated into)
the legato.db system_config table. This is insecure for production because the key
and the data it protects live in the same database. Always set LEGATE_MASTER_KEY
in production deployments.

## PBKDF2 salt migration

Per-user PBKDF2 salts are stored in the user_crypto table. On first access after
this migration, existing users get a random salt generated, their API keys are
re-encrypted under the new key (600K iterations), and the salt is persisted.
New users get a random salt at account creation.
"""

import base64
import logging
import os

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# PBKDF2 iterations — NIST 2026 recommendation for SHA-256
PBKDF2_ITERATIONS = 600_000

# Legacy iteration count used before this migration — needed for re-encryption
_LEGACY_PBKDF2_ITERATIONS = 100_000

# Master key - cached after first load
_master_key: str | None = None

# Track whether master key came from env var (True) or DB (False)
_master_key_from_env: bool = False


def _get_master_key() -> str:
    """Get the master encryption key.

    Priority:
    1. Environment variable LEGATE_MASTER_KEY (strongly preferred for production)
    2. Stored in database system_config table (auto-generated on first use — loud warning)

    The key is auto-generated and persisted if not found, so development
    environments work without any configuration.
    """
    global _master_key, _master_key_from_env
    if _master_key is not None:
        return _master_key

    # Check environment variable first (preferred for production)
    env_key = os.environ.get("LEGATE_MASTER_KEY")
    if env_key:
        _master_key = env_key
        _master_key_from_env = True
        logger.info("Master encryption key loaded from LEGATE_MASTER_KEY environment variable")
        return _master_key

    # Fall back to DB — load or generate.
    # In multi-tenant (hosted) mode the master key must not share a file with the
    # data it protects. Refuse to GENERATE a new db-stored key for a fresh hosted
    # deployment — but if a previous version already stored one, keep using it
    # (refusing would lock the deployment out of its own encrypted data) while
    # logging a critical to migrate.
    if os.environ.get("LEGATO_MODE") == "multi-tenant":
        if not _db_master_key_exists():
            raise RuntimeError(
                "LEGATE_MASTER_KEY is not set. Refusing to generate a database-stored "
                "master key in multi-tenant mode — the key and the encrypted data it "
                "protects would live in the same file. Set LEGATE_MASTER_KEY in the "
                "environment. Generate one with: python -m legate_studio.crypto"
            )
        logger.critical(
            "LEGATE_MASTER_KEY not set — using the master key stored in legato.db. "
            "Migrate it to the LEGATE_MASTER_KEY environment variable: the key currently "
            "shares a database file with the encrypted data it protects."
        )
    else:
        logger.warning(
            "⚠️  SECURITY WARNING: LEGATE_MASTER_KEY environment variable is not set. "
            "The master encryption key is being loaded from (or stored in) the database. "
            "This means the key and the data it protects are in the SAME database — "
            "a database dump exposes all encrypted user data. "
            "Set LEGATE_MASTER_KEY in your environment for production deployments. "
            "Generate a key with: python -m legate_studio.crypto"
        )
    _master_key = _load_or_create_master_key()
    _master_key_from_env = False
    return _master_key


def is_master_key_from_env() -> bool:
    """Return True if the master key was loaded from the environment variable.

    Call this after _get_master_key() has been invoked (e.g. at startup).
    Used by create_app() to emit a startup warning in the logs.
    """
    _get_master_key()  # ensure loaded
    return _master_key_from_env


def _db_master_key_exists() -> bool:
    """Return True if a master key is already stored in legato.db (read-only check).

    Used to distinguish a fresh hosted deployment (no key yet — refuse to create
    an insecure db-stored one) from an existing one (already has a db-stored key
    that we must keep using to stay able to decrypt existing data).
    """
    import sqlite3
    from pathlib import Path

    db_dir = Path(os.environ.get("LEGATO_DB_DIR", "/data"))
    if not db_dir.exists():
        db_dir = Path("./data")
    db_path = db_dir / "legato.db"

    if not db_path.exists():
        return False

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT 1 FROM system_config WHERE key = 'master_encryption_key' LIMIT 1"
        ).fetchone()
        return row is not None
    except sqlite3.OperationalError:
        # system_config table doesn't exist yet → no key stored
        return False
    finally:
        conn.close()


def _load_or_create_master_key() -> str:
    """Load master key from database, or generate and store if not exists."""
    import sqlite3
    from pathlib import Path

    # Get database path (same location as other DBs)
    db_dir = Path(os.environ.get("LEGATO_DB_DIR", "/data"))
    if not db_dir.exists():
        db_dir = Path("./data")
    db_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / "legato.db"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        # Ensure system_config table exists with canonical schema (both created_at AND updated_at).
        # The canonical definition lives in rag/database.py init_db(). This CREATE IF NOT EXISTS
        # keeps crypto.py self-contained for startup ordering safety, and matches the schema there.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migration: add updated_at if table was created by an older version of this file
        try:
            conn.execute("ALTER TABLE system_config ADD COLUMN updated_at TEXT DEFAULT CURRENT_TIMESTAMP")
        except Exception:
            pass  # Column already exists
        conn.commit()

        # Try to load existing key
        row = conn.execute("SELECT value FROM system_config WHERE key = 'master_encryption_key'").fetchone()

        if row:
            logger.info("Loaded master encryption key from database (consider moving to LEGATE_MASTER_KEY env var)")
            return row["value"]

        # Generate new key
        new_key = generate_master_key()

        conn.execute(
            "INSERT INTO system_config (key, value) VALUES (?, ?)",
            ("master_encryption_key", new_key),
        )
        conn.commit()

        logger.info("Generated and stored new master encryption key in database")
        return new_key

    finally:
        conn.close()


def _get_user_salt(user_id: str) -> bytes | None:
    """Load the per-user PBKDF2 salt from the user_crypto table.

    Returns None if no salt has been stored yet (legacy user pre-migration).
    """
    import sqlite3
    from pathlib import Path

    db_dir = Path(os.environ.get("LEGATO_DB_DIR", "/data"))
    if not db_dir.exists():
        db_dir = Path("./data")
    db_path = db_dir / "legato.db"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_crypto (
                user_id TEXT PRIMARY KEY,
                pbkdf2_salt BLOB NOT NULL,
                pbkdf2_iterations INTEGER NOT NULL DEFAULT 600000,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        row = conn.execute(
            "SELECT pbkdf2_salt, pbkdf2_iterations FROM user_crypto WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row:
            return bytes(row["pbkdf2_salt"]), row["pbkdf2_iterations"]
        return None, None
    finally:
        conn.close()


def _store_user_salt(user_id: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> None:
    """Persist a per-user PBKDF2 salt into the user_crypto table."""
    import sqlite3
    from pathlib import Path

    db_dir = Path(os.environ.get("LEGATO_DB_DIR", "/data"))
    if not db_dir.exists():
        db_dir = Path("./data")
    db_path = db_dir / "legato.db"

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_crypto (
                user_id TEXT PRIMARY KEY,
                pbkdf2_salt BLOB NOT NULL,
                pbkdf2_iterations INTEGER NOT NULL DEFAULT 600000,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            """
            INSERT INTO user_crypto (user_id, pbkdf2_salt, pbkdf2_iterations)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                pbkdf2_salt = excluded.pbkdf2_salt,
                pbkdf2_iterations = excluded.pbkdf2_iterations,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, salt, iterations),
        )
        conn.commit()
    finally:
        conn.close()


def _derive_key_with_params(master: str, salt: bytes, iterations: int) -> bytes:
    """Derive a Fernet-compatible key using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    derived = kdf.derive(master.encode("utf-8"))
    return base64.urlsafe_b64encode(derived)


def derive_user_key(user_id: str) -> bytes:
    """Derive a per-user encryption key from the master key.

    Uses PBKDF2 with a random per-user salt (stored in user_crypto table).
    For existing users without a stored salt, the migration path in
    _migrate_user_salt() handles re-encryption transparently.

    Args:
        user_id: The user's unique identifier

    Returns:
        A 32-byte URL-safe base64-encoded key suitable for Fernet encryption
    """
    master = _get_master_key()
    salt, iterations = _get_user_salt(user_id)

    if salt is not None:
        # Happy path: use stored random salt with stored iteration count
        return _derive_key_with_params(master, salt, iterations)

    # No salt stored yet — this is a legacy user. We can't migrate here
    # (we'd need to re-encrypt their API keys too) so we return the legacy key
    # to allow reads, and rely on the migration path in encrypt_for_user() /
    # get_user_api_key() to upgrade on next write.
    logger.debug("User %s has no stored PBKDF2 salt — using legacy key for read", user_id)
    return _derive_key_with_params(master, user_id.encode("utf-8"), _LEGACY_PBKDF2_ITERATIONS)


def _migrate_user_to_random_salt(user_id: str) -> None:
    """Migrate a legacy user to random PBKDF2 salt + 600K iterations.

    This is called the first time we write encrypted data for a user who
    doesn't yet have a stored salt. The process:
      1. Load all encrypted API keys for this user
      2. Decrypt each with the legacy key (user_id salt, 100K iterations)
      3. Generate a fresh 32-byte random salt
      4. Re-encrypt each API key with the new key (random salt, 600K iterations)
      5. Store the new salt in user_crypto
      6. Write the re-encrypted keys back to user_api_keys

    If decryption fails for any key (key was already encrypted with a different
    scheme), that key is left in place — the user will need to re-enter it.
    """
    import sqlite3
    from pathlib import Path

    master = _get_master_key()

    # Build legacy key (user_id as salt, 100K iterations)
    legacy_key = _derive_key_with_params(master, user_id.encode("utf-8"), _LEGACY_PBKDF2_ITERATIONS)
    legacy_fernet = Fernet(legacy_key)

    # Generate new random salt
    new_salt = os.urandom(32)
    new_key = _derive_key_with_params(master, new_salt, PBKDF2_ITERATIONS)
    new_fernet = Fernet(new_key)

    db_dir = Path(os.environ.get("LEGATO_DB_DIR", "/data"))
    if not db_dir.exists():
        db_dir = Path("./data")
    db_path = db_dir / "legato.db"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        # Load all encrypted API keys for this user
        rows = conn.execute(
            "SELECT id, provider, key_encrypted FROM user_api_keys WHERE user_id = ?",
            (user_id,),
        ).fetchall()

        if not rows:
            # No API keys to migrate — just store the new salt
            _store_user_salt(user_id, new_salt, PBKDF2_ITERATIONS)
            logger.info("Migrated user %s to random PBKDF2 salt (no API keys to re-encrypt)", user_id)
            return

        # Try to decrypt and re-encrypt each key
        re_encrypted = []
        failed_providers = []
        for row in rows:
            try:
                plaintext = legacy_fernet.decrypt(bytes(row["key_encrypted"]))
                new_ciphertext = new_fernet.encrypt(plaintext)
                re_encrypted.append((new_ciphertext, row["id"]))
            except InvalidToken:
                # Key was not encrypted with the legacy scheme — leave it alone
                logger.warning(
                    "Could not re-encrypt API key for user %s provider %s during salt migration "
                    "(not encrypted with legacy key) — user will need to re-enter this key",
                    user_id, row["provider"],
                )
                failed_providers.append(row["provider"])

        # Write re-encrypted keys back
        for new_ciphertext, row_id in re_encrypted:
            conn.execute(
                "UPDATE user_api_keys SET key_encrypted = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (new_ciphertext, row_id),
            )
        conn.commit()

        # Store the new salt (after successful DB writes)
        _store_user_salt(user_id, new_salt, PBKDF2_ITERATIONS)

        if failed_providers:
            logger.warning(
                "User %s salt migration: %d/%d API keys re-encrypted successfully; "
                "%d could not be migrated (%s) — user must re-enter those keys",
                user_id, len(re_encrypted), len(rows), len(failed_providers),
                ", ".join(failed_providers),
            )
        else:
            logger.info(
                "User %s migrated to random PBKDF2 salt: %d API key(s) re-encrypted "
                "(%d iterations → %d iterations)",
                user_id, len(re_encrypted), _LEGACY_PBKDF2_ITERATIONS, PBKDF2_ITERATIONS,
            )

    finally:
        conn.close()


def provision_user_salt(user_id: str) -> None:
    """Provision a random PBKDF2 salt for a new user.

    Call this at account creation time so the user starts with a random salt
    rather than triggering the migration path later.

    Safe to call multiple times — no-ops if salt already exists.
    """
    salt, _ = _get_user_salt(user_id)
    if salt is not None:
        return  # Already provisioned
    new_salt = os.urandom(32)
    _store_user_salt(user_id, new_salt, PBKDF2_ITERATIONS)
    logger.debug("Provisioned random PBKDF2 salt for new user %s", user_id)


def encrypt_for_user(user_id: str, plaintext: str) -> bytes:
    """Encrypt data for a specific user.

    On first call for a legacy user (no stored salt), transparently migrates
    their existing API keys to the new random-salt scheme before encrypting.

    Args:
        user_id: The user's unique identifier
        plaintext: The data to encrypt

    Returns:
        Encrypted bytes (Fernet token)
    """
    salt, _ = _get_user_salt(user_id)
    if salt is None:
        # Legacy user — run migration before encrypting new data
        logger.info("Running PBKDF2 salt migration for user %s on first encrypted write", user_id)
        _migrate_user_to_random_salt(user_id)

    key = derive_user_key(user_id)
    f = Fernet(key)
    return f.encrypt(plaintext.encode("utf-8"))


def decrypt_for_user(user_id: str, ciphertext: bytes) -> str | None:
    """Decrypt user's data.

    Automatically uses the stored random salt if available, otherwise falls
    back to the legacy key (user_id as salt, 100K iterations) for pre-migration
    data that hasn't been re-encrypted yet.

    Args:
        user_id: The user's unique identifier
        ciphertext: The encrypted data (Fernet token)

    Returns:
        Decrypted string, or None if decryption fails
    """
    master = _get_master_key()
    salt, iterations = _get_user_salt(user_id)

    if salt is not None:
        # Use stored random salt
        try:
            key = _derive_key_with_params(master, salt, iterations)
            f = Fernet(key)
            return f.decrypt(ciphertext).decode("utf-8")
        except InvalidToken:
            logger.error(
                "Failed to decrypt data for user %s with stored salt — "
                "data may have been encrypted before migration",
                user_id,
            )
            return None
        except Exception as e:
            logger.error("Failed to decrypt data for user %s: %s", user_id, e)
            return None
    else:
        # Legacy path: no salt stored — try with user_id as salt
        try:
            key = _derive_key_with_params(master, user_id.encode("utf-8"), _LEGACY_PBKDF2_ITERATIONS)
            f = Fernet(key)
            return f.decrypt(ciphertext).decode("utf-8")
        except InvalidToken:
            logger.error("Failed to decrypt data for user %s: invalid token", user_id)
            return None
        except Exception as e:
            logger.error("Failed to decrypt data for user %s: %s", user_id, e)
            return None


def encrypt_api_key(user_id: str, api_key: str) -> tuple[bytes, str]:
    """Encrypt an API key and return the ciphertext and hint.

    Args:
        user_id: The user's unique identifier
        api_key: The API key to encrypt

    Returns:
        Tuple of (encrypted_key, key_hint)
        The hint is the last 4 characters for UI display
    """
    encrypted = encrypt_for_user(user_id, api_key)
    hint = api_key[-4:] if len(api_key) >= 4 else "****"
    return encrypted, hint


def decrypt_api_key(user_id: str, encrypted_key: bytes) -> str | None:
    """Decrypt an API key.

    Args:
        user_id: The user's unique identifier
        encrypted_key: The encrypted API key

    Returns:
        The decrypted API key, or None if decryption fails
    """
    return decrypt_for_user(user_id, encrypted_key)


def generate_master_key() -> str:
    """Generate a new master key for initial setup.

    Returns:
        A URL-safe base64-encoded 32-byte key
    """
    return base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8")


# Convenience function for generating keys during setup
if __name__ == "__main__":
    print("Generated master key (set as LEGATE_MASTER_KEY environment variable):")
    print(generate_master_key())
