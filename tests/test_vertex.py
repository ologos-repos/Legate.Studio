"""
Tests for the Vertex AI platform backend (legate_studio.vertex) and the motif
processor's backend resolution. No real Vertex/GCP calls — SDK entry points are
monkeypatched.
"""

import json

import pytest

from legate_studio import vertex


@pytest.fixture(autouse=True)
def _clean_vertex_env(monkeypatch):
    for var in (
        "LEGATE_PLATFORM_BACKEND",
        "VERTEX_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "VERTEX_LOCATION",
        "VERTEX_MODEL",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.delenv(var, raising=False)
    # Reset the once-per-process credential guard.
    monkeypatch.setattr(vertex, "_credentials_ready", False)


# ── enablement / config ─────────────────────────────────────────────────────


def test_disabled_by_default():
    assert vertex.platform_backend() == "direct"
    assert vertex.vertex_enabled() is False


def test_enabled_requires_backend_and_project(monkeypatch):
    monkeypatch.setenv("LEGATE_PLATFORM_BACKEND", "vertex")
    assert vertex.vertex_enabled() is False  # no project yet
    monkeypatch.setenv("VERTEX_PROJECT", "my-proj")
    assert vertex.vertex_enabled() is True


def test_backend_vertex_without_project_stays_disabled(monkeypatch):
    monkeypatch.setenv("LEGATE_PLATFORM_BACKEND", "vertex")
    assert vertex.vertex_enabled() is False


def test_defaults(monkeypatch):
    assert vertex.vertex_model() == "gemini-2.5-flash"
    assert vertex.vertex_location() == "us-central1"
    monkeypatch.setenv("VERTEX_MODEL", "claude-sonnet-4-5@20250929")
    monkeypatch.setenv("VERTEX_LOCATION", "us-east5")
    assert vertex.vertex_model() == "claude-sonnet-4-5@20250929"
    assert vertex.vertex_location() == "us-east5"


# ── model family routing ────────────────────────────────────────────────────


def test_model_family():
    assert vertex.model_family("claude-sonnet-4-5@20250929") == "anthropic"
    assert vertex.model_family("gemini-2.5-flash") == "gemini"
    assert vertex.model_family("something-else") == "gemini"  # safe default
    assert vertex.model_family("") == "gemini"


def test_generate_dispatches_by_model(monkeypatch):
    calls = {}

    def fake_anthropic(s, u, m, mt, t):
        calls["anthropic"] = m
        return "claude-out"

    def fake_gemini(s, u, m, mt, t):
        calls["gemini"] = m
        return "gemini-out"

    monkeypatch.setattr(vertex, "_generate_anthropic_vertex", fake_anthropic)
    monkeypatch.setattr(vertex, "_generate_gemini_vertex", fake_gemini)
    monkeypatch.setattr(vertex, "ensure_credentials", lambda: None)

    assert vertex.generate("sys", "usr", model="claude-sonnet-4-5@20250929") == "claude-out"
    assert vertex.generate("sys", "usr", model="gemini-2.5-flash") == "gemini-out"
    assert calls == {"anthropic": "claude-sonnet-4-5@20250929", "gemini": "gemini-2.5-flash"}


def test_generate_uses_default_model(monkeypatch):
    seen = {}
    monkeypatch.setattr(vertex, "ensure_credentials", lambda: None)
    monkeypatch.setattr(
        vertex, "_generate_gemini_vertex",
        lambda s, u, m, mt, t: seen.setdefault("model", m) or "ok",
    )
    vertex.generate("sys", "usr")  # no model → VERTEX_MODEL default
    assert seen["model"] == "gemini-2.5-flash"


# ── credential materialization ──────────────────────────────────────────────


def test_ensure_credentials_writes_inline_json(monkeypatch, tmp_path):
    monkeypatch.setenv("LEGATO_DB_DIR", str(tmp_path))
    sa = json.dumps({"type": "service_account", "project_id": "p"})
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", sa)

    vertex.ensure_credentials()

    cred_path = tmp_path / "gcp-service-account.json"
    assert cred_path.exists()
    assert json.loads(cred_path.read_text())["project_id"] == "p"
    import os
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(cred_path)


def test_ensure_credentials_ignores_invalid_json(monkeypatch, tmp_path):
    monkeypatch.setenv("LEGATO_DB_DIR", str(tmp_path))
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "not json{")
    vertex.ensure_credentials()
    import os
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_ensure_credentials_noop_without_inline_json(monkeypatch, tmp_path):
    monkeypatch.setenv("LEGATO_DB_DIR", str(tmp_path))
    vertex.ensure_credentials()  # nothing set → ambient ADC, no file
    assert not (tmp_path / "gcp-service-account.json").exists()


# ── motif processor backend resolution ──────────────────────────────────────


def _make_processor():
    from legate_studio.motif_processor import MotifProcessor

    return MotifProcessor("job-x", "user-1")


def test_motif_uses_vertex_for_managed_when_no_byok(monkeypatch):
    from legate_studio import core
    from legate_studio.rag import usage

    monkeypatch.setattr(core, "get_provider_priority", lambda uid: ("anthropic", "gemini", "openai"))
    monkeypatch.setattr(core, "get_api_key_with_source", lambda uid, prov: (None, None))
    monkeypatch.setattr(core, "get_effective_tier", lambda uid: "managed_lite")
    monkeypatch.setattr(usage, "is_managed_tier", lambda tier: True)
    monkeypatch.setattr(vertex, "vertex_enabled", lambda: True)

    key, provider = _make_processor()._get_user_api_key()
    assert key is None and provider == "vertex"


def test_motif_byok_key_beats_vertex(monkeypatch):
    from legate_studio import core
    from legate_studio.rag import usage

    monkeypatch.setattr(core, "get_provider_priority", lambda uid: ("gemini", "anthropic", "openai"))
    monkeypatch.setattr(
        core, "get_api_key_with_source",
        lambda uid, prov: ("my-gemini-key", "user") if prov == "gemini" else (None, None),
    )
    monkeypatch.setattr(core, "get_effective_tier", lambda uid: "managed_lite")
    monkeypatch.setattr(usage, "is_managed_tier", lambda tier: True)
    monkeypatch.setattr(vertex, "vertex_enabled", lambda: True)

    key, provider = _make_processor()._get_user_api_key()
    assert (key, provider) == ("my-gemini-key", "gemini")


def test_motif_falls_back_to_platform_direct_key_when_vertex_off(monkeypatch):
    from legate_studio import core
    from legate_studio.rag import usage

    monkeypatch.setattr(core, "get_provider_priority", lambda uid: ("anthropic", "gemini", "openai"))
    monkeypatch.setattr(
        core, "get_api_key_with_source",
        lambda uid, prov: ("platform-anthropic", "platform") if prov == "anthropic" else (None, None),
    )
    monkeypatch.setattr(core, "get_effective_tier", lambda uid: "managed_lite")
    monkeypatch.setattr(usage, "is_managed_tier", lambda tier: True)
    monkeypatch.setattr(vertex, "vertex_enabled", lambda: False)

    key, provider = _make_processor()._get_user_api_key()
    assert (key, provider) == ("platform-anthropic", "anthropic")
