"""
Tests for API key resolution (core.get_api_key_with_source).

The resolver must prefer a user's own securely stored key (any tier) and
fall back to platform environment keys only for managed tiers. This is what
lets managed/beta users process motifs with their own key when the platform
key for a provider is not configured.
"""

import pytest

from legate_studio import core


@pytest.fixture
def _clean_env(monkeypatch):
    """Ensure provider env keys are unset unless a test sets them."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _set_tier(monkeypatch, tier):
    monkeypatch.setattr(core, "get_effective_tier", lambda user_id: tier)


def _set_user_key(monkeypatch, key):
    """Stub the stored-key lookup (auth.get_user_api_key)."""
    import legate_studio.auth as auth

    monkeypatch.setattr(auth, "get_user_api_key", lambda user_id, provider: key)


def test_managed_tier_uses_platform_key_when_no_user_key(monkeypatch, _clean_env):
    _set_tier(monkeypatch, "managed_lite")
    _set_user_key(monkeypatch, None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "platform-key")

    key, source = core.get_api_key_with_source("user-1", "anthropic")
    assert key == "platform-key"
    assert source == "platform"


def test_user_key_takes_precedence_over_platform_key(monkeypatch, _clean_env):
    _set_tier(monkeypatch, "managed_standard")
    _set_user_key(monkeypatch, "my-own-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "platform-key")

    key, source = core.get_api_key_with_source("user-1", "anthropic")
    assert key == "my-own-key"
    assert source == "user"


def test_managed_tier_falls_back_to_user_key_when_platform_key_missing(monkeypatch, _clean_env):
    """The motifs bug: managed/beta user, no platform ANTHROPIC_API_KEY set."""
    _set_tier(monkeypatch, "managed_lite")
    _set_user_key(monkeypatch, "my-own-key")

    key, source = core.get_api_key_with_source("user-1", "anthropic")
    assert key == "my-own-key"
    assert source == "user"


def test_managed_tier_no_keys_at_all(monkeypatch, _clean_env):
    _set_tier(monkeypatch, "managed_lite")
    _set_user_key(monkeypatch, None)

    key, source = core.get_api_key_with_source("user-1", "anthropic")
    assert key is None
    assert source is None


def test_byok_tier_uses_user_key(monkeypatch, _clean_env):
    _set_tier(monkeypatch, "byok")
    _set_user_key(monkeypatch, "my-own-key")

    key, source = core.get_api_key_with_source("user-1", "gemini")
    assert key == "my-own-key"
    assert source == "user"


def test_byok_tier_never_gets_platform_key(monkeypatch, _clean_env):
    _set_tier(monkeypatch, "byok")
    _set_user_key(monkeypatch, None)
    monkeypatch.setenv("GEMINI_API_KEY", "platform-key")

    key, source = core.get_api_key_with_source("user-1", "gemini")
    assert key is None
    assert source is None


def test_decrypt_failure_falls_back_to_platform_key(monkeypatch, _clean_env):
    """A corrupt stored key must not block the platform-key fallback."""
    import legate_studio.auth as auth

    def _boom(user_id, provider):
        raise ValueError("decryption failed")

    _set_tier(monkeypatch, "managed_lite")
    monkeypatch.setattr(auth, "get_user_api_key", _boom)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "platform-key")

    key, source = core.get_api_key_with_source("user-1", "anthropic")
    assert key == "platform-key"
    assert source == "platform"


def test_wrapper_returns_key_only(monkeypatch, _clean_env):
    _set_tier(monkeypatch, "managed_lite")
    _set_user_key(monkeypatch, "my-own-key")

    assert core.get_api_key_for_user("user-1", "anthropic") == "my-own-key"
