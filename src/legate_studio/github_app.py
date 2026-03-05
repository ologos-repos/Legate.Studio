"""
GitHub App authentication and token management.

Handles:
- JWT generation for App authentication
- Installation access token retrieval
- Token caching and refresh
- User installation management

GitHub Apps provide per-installation tokens that are:
- Scoped to specific repositories
- Short-lived (1 hour)
- Revocable by the user
- Auditable in GitHub
"""

import os
import time
import logging
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path

import jwt
import requests

logger = logging.getLogger(__name__)

# GitHub App configuration from environment
_app_config: Optional[dict] = None


def _get_app_config() -> dict:
    """Load GitHub App configuration from environment."""
    global _app_config
    if _app_config is None:
        app_id = os.environ.get('GITHUB_APP_ID')
        client_id = os.environ.get('GITHUB_APP_CLIENT_ID')
        client_secret = os.environ.get('GITHUB_APP_CLIENT_SECRET')
        private_key_path = os.environ.get('GITHUB_APP_PRIVATE_KEY_PATH')
        private_key = os.environ.get('GITHUB_APP_PRIVATE_KEY')

        if not all([app_id, client_id, client_secret]):
            raise ValueError(
                "GitHub App not configured. Set GITHUB_APP_ID, "
                "GITHUB_APP_CLIENT_ID, and GITHUB_APP_CLIENT_SECRET"
            )

        # Load private key from file or environment
        if private_key_path and Path(private_key_path).exists():
            private_key = Path(private_key_path).read_text()
        elif private_key:
            # Handle escaped newlines in env var
            private_key = private_key.replace('\\n', '\n')
        else:
            raise ValueError(
                "GitHub App private key not found. Set GITHUB_APP_PRIVATE_KEY_PATH "
                "or GITHUB_APP_PRIVATE_KEY"
            )

        _app_config = {
            'app_id': app_id,
            'client_id': client_id,
            'client_secret': client_secret,
            'private_key': private_key,
        }

    return _app_config


def generate_app_jwt() -> str:
    """Generate a JWT for authenticating as the GitHub App.

    JWTs are used to:
    - List installations
    - Get installation access tokens
    - Manage app settings

    Returns:
        A signed JWT valid for 10 minutes
    """
    config = _get_app_config()

    now = int(time.time())
    payload = {
        'iat': now - 60,  # Issued 60 seconds ago (clock skew tolerance)
        'exp': now + 600,  # Expires in 10 minutes
        'iss': config['app_id'],
    }

    return jwt.encode(payload, config['private_key'], algorithm='RS256')


def get_installation_access_token(installation_id: int) -> dict:
    """Get an access token for a specific installation.

    Installation tokens are:
    - Valid for 1 hour
    - Scoped to the repositories the installation has access to
    - Rate limited at 15,000 requests/hour (vs 5,000 for PATs)

    Args:
        installation_id: The GitHub App installation ID

    Returns:
        Dict with 'token', 'expires_at', and 'permissions'
    """
    app_jwt = generate_app_jwt()

    response = requests.post(
        f'https://api.github.com/app/installations/{installation_id}/access_tokens',
        headers={
            'Authorization': f'Bearer {app_jwt}',
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    return {
        'token': data['token'],
        'expires_at': data['expires_at'],
        'permissions': data.get('permissions', {}),
        'repository_selection': data.get('repository_selection'),
    }


def get_app_installations() -> list:
    """Get all installations of the GitHub App.

    Returns:
        List of installation dicts with id, account, permissions, etc.
    """
    app_jwt = generate_app_jwt()

    response = requests.get(
        'https://api.github.com/app/installations',
        headers={
            'Authorization': f'Bearer {app_jwt}',
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
        },
        timeout=30,
    )
    response.raise_for_status()

    return response.json()


def get_installation_for_user(github_login: str) -> Optional[dict]:
    """Find the installation for a specific GitHub user.

    Args:
        github_login: The GitHub username

    Returns:
        Installation dict if found, None otherwise
    """
    installations = get_app_installations()

    for installation in installations:
        account = installation.get('account', {})
        if account.get('login', '').lower() == github_login.lower():
            return installation

    return None


def exchange_code_for_user_token(code: str) -> dict:
    """Exchange an OAuth code for a user access token.

    This is used during the "Request user authorization during installation"
    flow. The user authorizes the app, and we get a code that can be
    exchanged for an access token.

    Args:
        code: The OAuth authorization code

    Returns:
        Dict with 'access_token', 'refresh_token', 'expires_in', etc.
    """
    config = _get_app_config()

    response = requests.post(
        'https://github.com/login/oauth/access_token',
        headers={'Accept': 'application/json'},
        data={
            'client_id': config['client_id'],
            'client_secret': config['client_secret'],
            'code': code,
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    if 'error' in data:
        raise ValueError(f"OAuth error: {data['error_description']}")

    return data


def refresh_user_token(refresh_token: str) -> dict:
    """Refresh an expired user access token.

    Args:
        refresh_token: The refresh token from initial authorization

    Returns:
        Dict with new 'access_token', 'refresh_token', 'expires_in', etc.
    """
    config = _get_app_config()

    response = requests.post(
        'https://github.com/login/oauth/access_token',
        headers={'Accept': 'application/json'},
        data={
            'client_id': config['client_id'],
            'client_secret': config['client_secret'],
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    if 'error' in data:
        raise ValueError(f"Token refresh error: {data['error_description']}")

    return data


def get_user_info(access_token: str) -> dict:
    """Get GitHub user info using an access token.

    Args:
        access_token: A user access token

    Returns:
        Dict with user info (id, login, email, etc.)
    """
    response = requests.get(
        'https://api.github.com/user',
        headers={
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
        },
        timeout=30,
    )
    response.raise_for_status()

    return response.json()


def get_user_emails(access_token: str) -> list:
    """Get GitHub user's email addresses.

    Args:
        access_token: A user access token with email scope

    Returns:
        List of email dicts with 'email', 'primary', 'verified'
    """
    response = requests.get(
        'https://api.github.com/user/emails',
        headers={
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
        },
        timeout=30,
    )
    response.raise_for_status()

    return response.json()


class InstallationTokenManager:
    """Manages installation access tokens with caching.

    Tokens are cached until 5 minutes before expiry to avoid
    unnecessary API calls while ensuring fresh tokens.
    """

    def __init__(self, db_conn=None):
        """Initialize the token manager.

        Args:
            db_conn: Optional database connection for persistent caching
        """
        self.db = db_conn
        self._cache: dict[int, dict] = {}

    def get_token(self, installation_id: int) -> str:
        """Get a valid access token for an installation.

        Returns a cached token if still valid, otherwise fetches a new one.

        Args:
            installation_id: The GitHub App installation ID

        Returns:
            A valid access token string
        """
        # Check memory cache
        cached = self._cache.get(installation_id)
        if cached:
            expires_at = datetime.fromisoformat(cached['expires_at'].replace('Z', '+00:00'))
            if expires_at > datetime.now(expires_at.tzinfo) + timedelta(minutes=5):
                return cached['token']

        # Fetch new token
        token_data = get_installation_access_token(installation_id)
        self._cache[installation_id] = token_data

        # Persist to database if available
        if self.db:
            self._save_token_to_db(installation_id, token_data)

        logger.info(f"Fetched new installation token for {installation_id}")
        return token_data['token']

    def _save_token_to_db(self, installation_id: int, token_data: dict):
        """Save token to database (encrypted)."""
        from .crypto import encrypt_for_user

        # Find user_id for this installation
        row = self.db.execute(
            "SELECT user_id FROM github_app_installations WHERE installation_id = ?",
            (installation_id,)
        ).fetchone()

        if row:
            user_id = row['user_id']
            encrypted = encrypt_for_user(user_id, token_data['token'])

            self.db.execute(
                """
                UPDATE github_app_installations
                SET access_token_encrypted = ?, token_expires_at = ?, updated_at = CURRENT_TIMESTAMP
                WHERE installation_id = ?
                """,
                (encrypted, token_data['expires_at'], installation_id)
            )
            self.db.commit()

    def invalidate(self, installation_id: int):
        """Remove a token from the cache (e.g., on revocation)."""
        self._cache.pop(installation_id, None)


# Global token manager instance
_token_manager: Optional[InstallationTokenManager] = None


def get_token_manager(db_conn=None) -> InstallationTokenManager:
    """Get the global token manager instance."""
    global _token_manager
    if _token_manager is None:
        _token_manager = InstallationTokenManager(db_conn)
    elif db_conn and _token_manager.db is None:
        _token_manager.db = db_conn
    return _token_manager
