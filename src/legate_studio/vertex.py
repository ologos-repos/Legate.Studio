"""Google Vertex AI platform backend.

Lets the managed/platform tier serve inference through Vertex AI instead of
direct provider APIs, so we can serve any Vertex-hosted model (Claude via
``AnthropicVertex``, Gemini via ``google-genai``) behind one GCP billing +
auth surface. BYOK users are unaffected — they keep using their own direct
provider keys (see ``core.get_api_key_with_source``).

**Provider-agnostic by design.** Vertex is a *transport*, not a provider. The
model family is inferred from the model id prefix, so any common model slots in:

  - ``claude-...``  → Claude on Vertex (``AnthropicVertex``)
  - ``gemini-...``  → Gemini on Vertex (``google-genai``, ``vertexai=True``)

**Inert until configured.** Nothing here activates unless
``LEGATE_PLATFORM_BACKEND=vertex`` AND a GCP project is set. When unset the app
behaves exactly as before (direct provider keys).

Configuration (environment variables):

  LEGATE_PLATFORM_BACKEND      'direct' (default) | 'vertex'
  VERTEX_PROJECT               GCP project id  (or GOOGLE_CLOUD_PROJECT) — required to enable
  VERTEX_LOCATION              region, default 'us-central1'
                               (Claude also accepts 'global' / 'us' / 'eu')
  VERTEX_MODEL                 default model id; family inferred from the prefix
                               (default 'gemini-2.5-flash')
  GOOGLE_SERVICE_ACCOUNT_JSON  inline service-account JSON. Written to a file in the
                               data dir at startup and exposed via
                               GOOGLE_APPLICATION_CREDENTIALS. Mirrors the inline
                               GITHUB_APP_PRIVATE_KEY pattern. If unset, ambient
                               Application Default Credentials are used as-is.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_VERTEX_MODEL = "gemini-2.5-flash"
DEFAULT_VERTEX_LOCATION = "us-central1"

# Guard so the credential file is written at most once per process.
_credentials_ready = False


def platform_backend() -> str:
    """Configured platform inference backend: 'direct' (default) or 'vertex'."""
    return os.environ.get("LEGATE_PLATFORM_BACKEND", "direct").strip().lower()


def vertex_project() -> str | None:
    """GCP project id for Vertex, or None if not configured."""
    return os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")


def vertex_location() -> str:
    return os.environ.get("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION)


def vertex_model() -> str:
    return os.environ.get("VERTEX_MODEL", DEFAULT_VERTEX_MODEL)


def vertex_enabled() -> bool:
    """True only when the platform backend is Vertex and a project is configured."""
    return platform_backend() == "vertex" and bool(vertex_project())


def model_family(model: str) -> str:
    """Infer the SDK family from a model id ('anthropic' or 'gemini').

    Defaults to 'gemini' for unknown ids since that is the safe Vertex-native
    default; callers pass explicit Claude ids to use the Anthropic path.
    """
    m = (model or "").lower()
    if m.startswith("claude"):
        return "anthropic"
    return "gemini"


def ensure_credentials() -> None:
    """Materialize GCP credentials from an inline JSON env var, once.

    If GOOGLE_SERVICE_ACCOUNT_JSON is set and GOOGLE_APPLICATION_CREDENTIALS is
    not already pointing somewhere, write the JSON to a private file in the data
    dir and point ADC at it. No-op if the inline var is absent (ambient ADC is
    then used) or the file was already written this process.
    """
    global _credentials_ready
    if _credentials_ready:
        return

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            json.loads(raw)  # validate before writing
        except json.JSONDecodeError:
            logger.error("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON — ignoring it")
            _credentials_ready = True
            return

        data_dir = Path(os.environ.get("LEGATO_DB_DIR", "/data"))
        if not data_dir.exists():
            data_dir = Path("./data")
        data_dir.mkdir(parents=True, exist_ok=True)

        cred_path = data_dir / "gcp-service-account.json"
        cred_path.write_text(raw)
        try:
            cred_path.chmod(0o600)
        except OSError:
            pass
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
        logger.info("Wrote GCP service-account credentials to %s for Vertex", cred_path)

    _credentials_ready = True


def generate(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """Single-turn generation through Vertex AI; returns the response text.

    Dispatches to the Claude or Gemini Vertex path based on the model id.
    """
    ensure_credentials()
    model = model or vertex_model()
    if model_family(model) == "anthropic":
        return _generate_anthropic_vertex(system, user, model, max_tokens, temperature)
    return _generate_gemini_vertex(system, user, model, max_tokens, temperature)


def _generate_anthropic_vertex(
    system: str, user: str, model: str, max_tokens: int, temperature: float
) -> str:
    """Claude on Vertex via the anthropic[vertex] SDK (AnthropicVertex)."""
    from anthropic import AnthropicVertex

    client = AnthropicVertex(project_id=vertex_project(), region=vertex_location())
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _generate_gemini_vertex(
    system: str, user: str, model: str, max_tokens: int, temperature: float
) -> str:
    """Gemini on Vertex via the google-genai SDK in Vertex mode."""
    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=vertex_project(), location=vertex_location())
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.text
