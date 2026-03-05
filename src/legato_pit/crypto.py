"""
Encryption utilities for multi-tenant data protection.

Implements a key hierarchy:
- Master Key (env var, rotatable)
  └── Per-User DEK (derived from user_id + master)
        └── Encrypts: API keys, sensitive preferences

All user-specific sensitive data is encrypted with a key derived from
their user_id, so even database access doesn't expose other users' data.
"""

import os
import base64
import logging
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# Master key - cached after first load
_master_key: Optional[str] = None


def _get_master_key() -> str:
    """Get the master encryption key.

    Priority:
    1. Environment variable LEGATO_MASTER_KEY (for manual override)
    2. Stored in database system_config table (auto-generated on first use)

    The key is auto-generated and persisted if not found, so users
    never need to configure encryption manually.
    """
    global _master_key
    if _master_key is not None:
        return _master_key

    # Check environment variable first (allows manual override)
    env_key = os.environ.get('LEGATO_MASTER_KEY')
    if env_key:
        _master_key = env_key
        return _master_key

    # Load or generate from database
    _master_key = _load_or_create_master_key()
    return _master_key


def _load_or_create_master_key() -> str:
    """Load master key from database, or generate and store if not exists."""
    import sqlite3
    from pathlib import Path

    # Get database path (same location as other DBs)
    db_dir = Path(os.environ.get('LEGATO_DB_DIR', '/data'))
    if not db_dir.exists():
        db_dir = Path('./data')
    db_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / 'legato.db'

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
        row = conn.execute(
            "SELECT value FROM system_config WHERE key = 'master_encryption_key'"
        ).fetchone()

        if row:
            logger.info("Loaded master encryption key from database")
            return row['value']

        # Generate new key
        new_key = generate_master_key()

        conn.execute(
            "INSERT INTO system_config (key, value) VALUES (?, ?)",
            ('master_encryption_key', new_key)
        )
        conn.commit()

        logger.info("Generated and stored new master encryption key")
        return new_key

    finally:
        conn.close()


def derive_user_key(user_id: str) -> bytes:
    """Derive a per-user encryption key from the master key.

    Uses PBKDF2 with the user_id as salt to create a unique key
    for each user. This means:
    - Each user's data is encrypted with a different key
    - Compromising one user's data doesn't expose others
    - Master key rotation requires re-encrypting all data

    Args:
        user_id: The user's unique identifier

    Returns:
        A 32-byte key suitable for Fernet encryption
    """
    master = _get_master_key()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=user_id.encode('utf-8'),
        iterations=100_000,
    )
    derived = kdf.derive(master.encode('utf-8'))
    return base64.urlsafe_b64encode(derived)


def encrypt_for_user(user_id: str, plaintext: str) -> bytes:
    """Encrypt data for a specific user.

    Args:
        user_id: The user's unique identifier
        plaintext: The data to encrypt

    Returns:
        Encrypted bytes (Fernet token)
    """
    key = derive_user_key(user_id)
    f = Fernet(key)
    return f.encrypt(plaintext.encode('utf-8'))


def decrypt_for_user(user_id: str, ciphertext: bytes) -> Optional[str]:
    """Decrypt user's data.

    Args:
        user_id: The user's unique identifier
        ciphertext: The encrypted data (Fernet token)

    Returns:
        Decrypted string, or None if decryption fails
    """
    try:
        key = derive_user_key(user_id)
        f = Fernet(key)
        return f.decrypt(ciphertext).decode('utf-8')
    except InvalidToken:
        logger.error(f"Failed to decrypt data for user {user_id}: invalid token")
        return None
    except Exception as e:
        logger.error(f"Failed to decrypt data for user {user_id}: {e}")
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


def decrypt_api_key(user_id: str, encrypted_key: bytes) -> Optional[str]:
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
    return base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')


# Convenience function for generating keys during setup
if __name__ == '__main__':
    print("Generated master key (add to LEGATO_MASTER_KEY env var):")
    print(generate_master_key())
