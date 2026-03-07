"""
Chat Blueprint - RAG-Enabled Conversational Interface

Provides web UI and API for chatting with the knowledge base.
Uses ChatSessionManager for in-memory buffering with periodic flush.

Subscription awareness:
  - managed_lite ($5/mo):     platform keys, $4.50/mo token credits, cap enforced
  - managed_standard ($10/mo): platform keys, $9.00/mo token credits, cap enforced
  - managed_plus ($20/mo):    platform keys, $18.00/mo token credits, cap enforced
  - byok ($0.99/mo):          user's own keys, no cap (they pay provider directly)
  - trial:                    limited access, no managed keys
  - single-tenant mode:       always uses env vars, no cap enforcement
"""

import logging
import os
import secrets

from flask import Blueprint, current_app, g, jsonify, render_template, request, session, url_for

from .core import beta_gate, library_required, login_required, paid_required
from .rag.chat_session_manager import get_chat_manager

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__, url_prefix="/chat")


def _is_multi_tenant() -> bool:
    """Return True if running in multi-tenant SaaS mode."""
    return current_app.config.get("LEGATO_MODE") == "multi-tenant"


def _get_user_id() -> str | None:
    """Get the current user's ID from session."""
    return session.get("user", {}).get("user_id")


def _resolve_api_key(chat_provider_value: str) -> tuple[str | None, str]:
    """Resolve the API key and tier for the current user + provider.

    In single-tenant mode, always returns (None, 'single-tenant') — ChatService
    will fall back to environment variables as before.

    In multi-tenant mode, uses get_api_key_for_user() which handles BYOK vs Managed.

    Args:
        chat_provider_value: ChatProvider.value string ('claude', 'openai', 'gemini')

    Returns:
        Tuple of (api_key: str | None, tier: str)
        api_key is None in single-tenant mode (env var fallback)
    """
    if not _is_multi_tenant():
        return None, "single-tenant"

    from .core import get_api_key_for_user, get_effective_tier

    user_id = _get_user_id()
    tier = get_effective_tier(user_id) if user_id else "trial"

    # Map chat provider value → API key provider name
    provider_map = {"claude": "anthropic", "openai": "openai", "gemini": "gemini"}
    key_provider = provider_map.get(chat_provider_value, "anthropic")

    api_key = get_api_key_for_user(user_id, key_provider)
    return api_key, tier


def get_services():
    """Get or create RAG services (uses env-var key for default service).

    The default ChatService in g.chat_services uses the environment-variable key
    (or None in multi-tenant, resolved per-request in send_message). The per-request
    key resolution happens in send_message() when constructing the actual service
    used for the LLM call.
    """
    if "chat_services" not in g:
        from .rag.chat_service import ChatProvider, ChatService
        from .rag.context_builder import ContextBuilder
        from .rag.database import get_user_legato_db, init_chat_db
        from .rag.embedding_service import EmbeddingService
        from .rag.openai_provider import OpenAIEmbeddingProvider

        # Initialize databases
        # User-scoped legato db for embeddings/knowledge
        legato_db = get_user_legato_db()

        # chat.db for sessions/messages (shared for now)
        if "chat_db_conn" not in g:
            g.chat_db_conn = init_chat_db()

        # Create embedding provider
        try:
            provider = OpenAIEmbeddingProvider()
        except ValueError:
            from .rag.ollama_provider import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()

        # Embedding service uses user's legato db
        embedding_service = EmbeddingService(provider, legato_db)
        context_builder = ContextBuilder(embedding_service)

        # Create default chat service (env-var key, for stats/config display)
        provider_name = os.environ.get("CHAT_PROVIDER", "claude").lower()
        if provider_name == "gemini":
            chat_provider = ChatProvider.GEMINI
        elif provider_name == "openai":
            chat_provider = ChatProvider.OPENAI
        else:
            chat_provider = ChatProvider.CLAUDE

        chat_service = ChatService(
            provider=chat_provider,
            model=os.environ.get("CHAT_MODEL"),
        )

        g.chat_services = {
            "embedding": embedding_service,
            "context": context_builder,
            "chat": chat_service,
            "legato_db": legato_db,  # For RAG/embeddings (from get_user_legato_db)
            "chat_db": g.chat_db_conn,  # For sessions/messages
        }

    return g.chat_services


def get_or_create_session(db_conn) -> str:
    """Get current chat session or create a new one.

    Uses ChatSessionManager - session header is written to DB immediately,
    messages are buffered in memory.
    """
    session_id = session.get("chat_session_id")
    user = session.get("user", {})
    manager = get_chat_manager()

    if not session_id:
        session_id = secrets.token_urlsafe(16)
        session["chat_session_id"] = session_id

    # Ensure session exists in manager (creates in DB if new)
    manager.get_or_create_session(session_id, user.get("username"), db_conn)

    return session_id


def get_chat_history(db_conn, session_id: str, limit: int = 20):
    """Get recent chat history for a session.

    Combines messages from DB and in-memory buffer.
    """
    manager = get_chat_manager()
    messages = manager.get_messages(session_id, limit, db_conn)

    return [
        {
            "role": m["role"],
            "content": m["content"],
            "context": m.get("context_used"),
            "timestamp": None,  # Not tracked in buffer
        }
        for m in messages
    ]


def save_message(db_conn, session_id: str, role: str, content: str, context=None, model=None):
    """Save a chat message (buffered in memory, flushed periodically)."""
    manager = get_chat_manager()
    manager.add_message(session_id, role, content, context, model, db_conn)


@chat_bp.route("/")
@library_required
@paid_required
@beta_gate("chat")
def index():
    """Chat interface page."""
    services = get_services()
    stats = services["context"].get_stats()

    # Pass tier to template so it can show/hide the usage indicator
    tier = "single-tenant"
    if _is_multi_tenant():
        from .core import get_effective_tier
        user_id = _get_user_id()
        tier = get_effective_tier(user_id) if user_id else "trial"

    return render_template(
        "chat.html",
        stats=stats,
        provider=services["chat"].provider.value,
        model=services["chat"].model,
        tier=tier,
    )


@chat_bp.route("/api/send", methods=["POST"])
@login_required
@paid_required
@beta_gate("chat")
def send_message():
    """Send a message and get a response.

    Request body:
    {
        "message": "User's question",
        "include_context": true,  # Optional, default true
        "provider": "claude",     # Optional: claude, openai, or gemini
        "model": "claude-sonnet-4-20250514"  # Optional: specific model
    }

    Response:
    {
        "response": "Assistant's response",
        "context": [{"entry_id": "...", "title": "...", "similarity": 0.85}],
        "model": "claude-sonnet-4-20250514"
    }

    Error (BYOK, no key configured):
    {"error": "No API key configured for this provider. Add your key in Settings."}, 400

    Error (Managed cap reached):
    {"error": "...", "credit_cap_reached": true, "upgrade_url": "..."}, 429
    """
    data = request.get_json()

    if not data or not data.get("message"):
        return jsonify({"error": "message required"}), 400

    message = data["message"]
    include_context = data.get("include_context", True)
    requested_provider = data.get("provider")
    requested_model = data.get("model")

    try:
        from .rag.chat_service import ChatProvider, ChatService

        services = get_services()

        # ── Determine effective provider ──────────────────────────────────────
        chat_service = services["chat"]

        # Build the provider enum for what was requested (or default)
        if requested_provider == "gemini":
            effective_provider = ChatProvider.GEMINI
        elif requested_provider == "openai":
            effective_provider = ChatProvider.OPENAI
        elif requested_provider == "claude":
            effective_provider = ChatProvider.CLAUDE
        else:
            effective_provider = chat_service.provider

        effective_model = requested_model or chat_service.model

        # ── Resolve API key & tier ────────────────────────────────────────────
        api_key, tier = _resolve_api_key(effective_provider.value)

        # BYOK check: if multi-tenant and NOT a managed tier and NOT single-tenant, user must
        # have their own key stored. Managed tiers use platform keys (already resolved above).
        from .rag.usage import is_managed_tier as _is_managed_tier
        if _is_multi_tenant() and tier != "single-tenant" and not _is_managed_tier(tier) and not api_key:
            return jsonify({
                "error": (
                    f"No API key configured for {effective_provider.value}. "
                    "Add your key in Settings."
                )
            }), 400

        user_id = _get_user_id()

        # ── Credit cap check (Managed tier only) ─────────────────────────────
        if _is_multi_tenant() and user_id:
            from .rag.usage import check_credit_cap, get_cap_for_tier, is_managed_tier
            if is_managed_tier(tier):
                allowed, remaining = check_credit_cap(user_id, tier=tier)
                cap_dollars = get_cap_for_tier(tier) / 1_000_000
                if not allowed:
                    return jsonify({
                        "error": (
                            f"Monthly credit limit reached (${cap_dollars:.2f}). "
                            "Purchase more credits or upgrade to BYOK for unlimited chat "
                            "with your own keys."
                        ),
                        "credit_cap_reached": True,
                        "upgrade_url": url_for("auth.setup"),
                        "topup_url": url_for("chat.buy_credits"),
                    }), 429

        # ── Instantiate ChatService with resolved key ────────────────────────
        # In single-tenant mode api_key is None and ChatService falls back to env vars.
        chat_service = ChatService(
            provider=effective_provider,
            model=effective_model,
            api_key=api_key,
        )

        session_id = get_or_create_session(services["chat_db"])

        # Get conversation history
        history = get_chat_history(services["chat_db"], session_id, limit=10)
        history_messages = [{"role": h["role"], "content": h["content"]} for h in history]

        # Build prompt with RAG context
        messages = services["context"].build_messages(
            query=message,
            history=history_messages,
        )

        # Get context entries for response
        prompt_data = services["context"].build_prompt(message)
        context_entries = prompt_data.get("context_entries", [])

        # Save user message (buffered)
        save_message(services["chat_db"], session_id, "user", message)

        # ── LLM call ──────────────────────────────────────────────────────────
        result = chat_service.chat(messages)
        response_text = result["text"]
        usage = result["usage"]

        # ── Token usage tracking (Managed tier only) ─────────────────────────
        from .rag.usage import is_managed_tier as _is_managed
        if _is_multi_tenant() and _is_managed(tier) and user_id:
            from .rag.usage import estimate_cost, record_usage_event, update_usage_meter
            cost = estimate_cost(
                effective_provider.value,
                effective_model,
                usage["input_tokens"],
                usage["output_tokens"],
            )
            record_usage_event(
                user_id,
                effective_provider.value,
                usage["input_tokens"],
                usage["output_tokens"],
                cost,
            )
            update_usage_meter(
                user_id,
                usage["input_tokens"],
                usage["output_tokens"],
                cost,
            )

        # Save assistant message with context (buffered)
        save_message(
            services["chat_db"],
            session_id,
            "assistant",
            response_text,
            context=context_entries,
            model=chat_service.model,
        )

        # Flush turn to disk - ensures complete turns are never lost
        manager = get_chat_manager()
        manager.flush_session(session_id, services["chat_db"])

        # Auto-title the session if no title yet (use first 50 chars of first message)
        session_row = (
            services["chat_db"]
            .execute("SELECT title FROM chat_sessions WHERE session_id = ?", (session_id,))
            .fetchone()
        )
        if session_row and not session_row["title"]:
            title = message[:50] + ("..." if len(message) > 50 else "")
            services["chat_db"].execute(
                ("UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?"),
                (title, session_id),
            )
            services["chat_db"].commit()

        return jsonify(
            {
                "response": response_text,
                "context": context_entries if include_context else [],
                "model": chat_service.model,
                "provider": chat_service.provider.value,
            }
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/api/usage", methods=["GET"])
@login_required
@paid_required
@beta_gate("chat")
def get_usage():
    """Get current month's token usage and credit cap status.

    Only meaningful for Managed-tier users in multi-tenant mode.
    For other tiers/modes, returns a no-tracking response.

    Response (managed tier — example for managed_standard):
    {
        "tracked": true,
        "tier": "managed_standard",
        "tokens_in": 12345,
        "tokens_out": 5678,
        "cost_microdollars": 123456,
        "cost_dollars": 0.1235,
        "base_cap_microdollars": 9000000,
        "topup_credits_microdollars": 0,
        "effective_cap_microdollars": 9000000,
        "remaining_microdollars": 8876544,
        "remaining_dollars": 8.8765,
        "cap_dollars": 9.0,
        "period": "2026-03",
        "percent_used": 1.4,
        "topup_price_dollars": 5.0,
        "topup_credits_dollars": 4.5
    }

    Response (non-managed / single-tenant):
    {
        "tracked": false,
        "tier": "byok"
    }
    """
    if not _is_multi_tenant():
        return jsonify({"tracked": False, "tier": "single-tenant"})

    from .core import get_effective_tier
    from .rag.usage import get_usage_summary, is_managed_tier
    user_id = _get_user_id()
    tier = get_effective_tier(user_id) if user_id else "trial"

    if not is_managed_tier(tier):
        return jsonify({"tracked": False, "tier": tier})

    summary = get_usage_summary(user_id, tier=tier)
    summary["tracked"] = True
    summary["tier"] = tier
    summary["topup_price_dollars"] = 5.0
    summary["topup_credits_dollars"] = 4.5
    return jsonify(summary)


@chat_bp.route("/api/credits/buy", methods=["POST"])
@login_required
@paid_required
@beta_gate("chat")
def buy_credits():
    """Initiate a credit top-up purchase.

    This is a stub for the Stripe payment integration. In production, this
    would create a Stripe PaymentIntent and return a client secret for the
    frontend to complete payment with Stripe.js.

    For now, it returns the information needed to show the purchase UI and
    records a stub top-up for testing (only in non-production environments).

    Request body: {} (empty — fixed $5 top-up)

    Response:
    {
        "topup_price_dollars": 5.0,
        "topup_credits_dollars": 4.5,
        "topup_credits_microdollars": 4500000,
        "stub": true,  # Present in non-production environments
        "message": "Stripe integration coming soon. Contact support to purchase credits."
    }
    """
    if not _is_multi_tenant():
        return jsonify({"error": "Credit top-ups only available in multi-tenant mode"}), 400

    from .core import get_effective_tier
    from .rag.usage import is_managed_tier
    user_id = _get_user_id()
    tier = get_effective_tier(user_id) if user_id else "trial"

    if not is_managed_tier(tier):
        return jsonify({
            "error": "Credit top-ups are only available for Managed tier users.",
            "upgrade_url": url_for("auth.setup"),
        }), 400

    # TODO: When Stripe is integrated, create a PaymentIntent here and return
    # the client_secret for Stripe.js to complete the payment flow.
    # On payment success, call record_credit_topup() from a Stripe webhook handler.

    return jsonify({
        "topup_price_dollars": 5.0,
        "topup_credits_dollars": 4.5,
        "topup_credits_microdollars": 4_500_000,
        "stub": True,
        "message": (
            "Credit top-ups are coming soon. "
            "Contact support to manually add credits, or switch to BYOK "
            "for unlimited usage with your own API keys."
        ),
        "byok_url": url_for("auth.setup"),
    })


@chat_bp.route("/api/history", methods=["GET"])
@login_required
@paid_required
@beta_gate("chat")
def get_history():
    """Get chat history for current session.

    Response:
    {
        "messages": [
            {
                "role": "user" | "assistant",
                "content": "...",
                "context": [...],
                "timestamp": "..."
            }
        ]
    }
    """
    try:
        services = get_services()
        session_id = session.get("chat_session_id")

        if not session_id:
            return jsonify({"messages": []})

        history = get_chat_history(services["chat_db"], session_id, limit=50)
        return jsonify({"messages": history})

    except Exception as e:
        logger.error(f"Get history failed: {e}")
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/api/sessions", methods=["GET"])
@login_required
@paid_required
@beta_gate("chat")
def list_sessions():
    """List all chat sessions for the current user."""
    try:
        services = get_services()
        user = session.get("user", {})

        rows = (
            services["chat_db"]
            .execute(
                """
            SELECT session_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT 20
            """,
                (user.get("username"),),
            )
            .fetchall()
        )

        return jsonify(
            {
                "sessions": [
                    {
                        "session_id": r["session_id"],
                        "title": r["title"],
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                    }
                    for r in rows
                ]
            }
        )

    except Exception as e:
        logger.error(f"List sessions failed: {e}")
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/api/sessions/new", methods=["POST"])
@login_required
@paid_required
@beta_gate("chat")
def new_session():
    """Start a new chat session."""
    try:
        services = get_services()
        manager = get_chat_manager()

        # Flush current session before clearing
        old_session_id = session.get("chat_session_id")
        if old_session_id:
            manager.end_session(old_session_id, services["chat_db"])

        # Clear from flask session
        session.pop("chat_session_id", None)

        # Create new one
        session_id = get_or_create_session(services["chat_db"])

        return jsonify(
            {
                "session_id": session_id,
                "message": "New session created",
            }
        )

    except Exception as e:
        logger.error(f"New session failed: {e}")
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/api/sessions/<session_id>/load", methods=["POST"])
@login_required
@paid_required
@beta_gate("chat")
def load_session(session_id):
    """Switch to/load a specific chat session.

    Sets the current session in the Flask session and returns the session info.
    """
    try:
        services = get_services()
        user = session.get("user", {})

        # Verify ownership
        row = (
            services["chat_db"]
            .execute(
                "SELECT user_id, title FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            )
            .fetchone()
        )

        if not row or row["user_id"] != user.get("username"):
            return jsonify({"error": "Session not found"}), 404

        # Set as current session
        session["chat_session_id"] = session_id

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "title": row["title"],
            }
        )

    except Exception as e:
        logger.error(f"Load session failed: {e}")
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/api/sessions/<session_id>", methods=["DELETE"])
@login_required
@paid_required
@beta_gate("chat")
def delete_session(session_id):
    """Delete a chat session."""
    try:
        services = get_services()
        manager = get_chat_manager()
        user = session.get("user", {})

        # Verify ownership
        row = (
            services["chat_db"]
            .execute(
                "SELECT user_id FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            )
            .fetchone()
        )

        if not row or row["user_id"] != user.get("username"):
            return jsonify({"error": "Session not found"}), 404

        # Remove from memory cache (don't flush - we're deleting anyway)
        manager._sessions.pop(session_id, None)

        # Delete messages and session from DB
        services["chat_db"].execute(
            "DELETE FROM chat_messages WHERE session_id = ?",
            (session_id,),
        )
        services["chat_db"].execute(
            "DELETE FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        )
        services["chat_db"].commit()

        # Clear from flask session if current
        if session.get("chat_session_id") == session_id:
            session.pop("chat_session_id", None)

        return jsonify({"success": True})

    except Exception as e:
        logger.error(f"Delete session failed: {e}")
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/api/config", methods=["GET"])
@login_required
@paid_required
@beta_gate("chat")
def get_config():
    """Get current chat configuration."""
    from .rag.chat_service import ChatProvider, ChatService

    services = get_services()

    return jsonify(
        {
            "provider": services["chat"].provider.value,
            "model": services["chat"].model,
            "available_models": {
                "claude": ChatService.get_available_models(ChatProvider.CLAUDE),
                "openai": ChatService.get_available_models(ChatProvider.OPENAI),
            },
        }
    )


@chat_bp.route("/api/stats", methods=["GET"])
@login_required
@paid_required
@beta_gate("chat")
def get_stats():
    """Get RAG system statistics for debugging.

    Response:
    {
        "knowledge_entries": 42,
        "embeddings": 42,
        "provider": "text-embedding-3-small",
        "has_openai_key": true,
        "has_anthropic_key": true
    }
    """
    import os

    try:
        services = get_services()
        stats = services["context"].get_stats()

        # Add key availability info
        stats["has_openai_key"] = bool(os.environ.get("OPENAI_API_KEY"))
        stats["has_anthropic_key"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
        stats["has_system_pat"] = bool(os.environ.get("SYSTEM_PAT"))

        return jsonify(stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/api/models", methods=["GET"])
@login_required
@paid_required
@beta_gate("chat")
def get_models():
    """Get available models for a provider.

    Query params:
    - provider: 'claude' or 'openai' (default: both)

    Response:
    {
        "claude": [{"id": "...", "name": "..."}],
        "openai": [{"id": "...", "name": "..."}]
    }
    """
    from .rag.chat_service import ChatProvider, ChatService

    provider = request.args.get("provider")

    try:
        if provider == "claude":
            return jsonify(
                {
                    "claude": ChatService.get_available_models(ChatProvider.CLAUDE),
                }
            )
        elif provider == "openai":
            return jsonify(
                {
                    "openai": ChatService.get_available_models(ChatProvider.OPENAI),
                }
            )
        elif provider == "gemini":
            return jsonify(
                {
                    "gemini": ChatService.get_available_models(ChatProvider.GEMINI),
                }
            )
        else:
            return jsonify(
                {
                    "claude": ChatService.get_available_models(ChatProvider.CLAUDE),
                    "openai": ChatService.get_available_models(ChatProvider.OPENAI),
                    "gemini": ChatService.get_available_models(ChatProvider.GEMINI),
                }
            )
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return jsonify({"error": str(e)}), 500
