"""
MCP Protocol Handler for Claude.ai

Implements the Model Context Protocol (JSON-RPC 2.0) to expose
Legate Studio library tools and resources to Claude via the MCP connector.

Protocol version: 2025-06-18
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime

from flask import Blueprint, current_app, g, jsonify, request

from .core import limiter
from .oauth_server import require_mcp_auth, verify_access_token

logger = logging.getLogger(__name__)

mcp_bp = Blueprint("mcp", __name__, url_prefix="/mcp")

# Disable strict slashes so /mcp and /mcp/ both work
mcp_bp.strict_slashes = False

# MCP Protocol version (as of June 2025 spec)
MCP_PROTOCOL_VERSION = "2025-06-18"


def get_db():
    """Get legato database connection for current user."""
    from .rag.database import get_user_legato_db

    return get_user_legato_db()


def get_library_db(args: dict) -> tuple:
    """Get the appropriate database connection based on args.

    If 'library_id' is present in args, returns the shared library DB with the
    caller's role. Otherwise returns the personal user DB with role=None.

    Args:
        args: Tool arguments dict (may contain 'library_id')

    Returns:
        (db_connection, role) where role is None (personal), 'owner', or 'collaborator'

    Raises:
        ValueError: If library_id provided but user has no active membership
    """
    library_id = args.get("library_id", "").strip() if args.get("library_id") else None

    if not library_id:
        return get_db(), None

    from .rag.database import get_shared_library_db, init_db as init_shared_db

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None
    if not user_id:
        raise ValueError("Authentication required")

    # Require managed tier to access shared libraries
    require_managed_tier(user_id)

    # Verify caller is an active member
    shared_meta = init_shared_db()
    row = shared_meta.execute(
        """
        SELECT role FROM shared_library_members
        WHERE shared_library_id = ? AND user_id = ? AND status = 'active'
        """,
        (library_id, user_id),
    ).fetchone()

    if not row:
        raise ValueError(f"Access denied: you are not an active member of library '{library_id}'")

    role = row["role"]
    db = get_shared_library_db(library_id)
    return db, role


def check_write_permission(library_id: str | None, role: str | None) -> None:
    """Verify caller has write permission for the given library context.

    Args:
        library_id: None for personal library, or the shared library UUID
        role: None for personal, 'owner', or 'collaborator'

    Raises:
        ValueError: If caller is a collaborator (must use Draft & Merge workflow)
    """
    if library_id is None:
        return  # Personal library: always allow
    if role == "owner":
        return  # Owner: always allow
    if role == "collaborator":
        raise ValueError(
            "Collaborators cannot write directly to shared libraries. "
            "Use Draft & Merge workflow: create_draft → submit_draft → merge_draft."
        )
    # Unknown role — deny
    raise ValueError(f"Insufficient permissions for library '{library_id}'")


def require_managed_tier(user_id: str) -> None:
    """Verify user has a managed subscription tier.

    Args:
        user_id: The user ID to check

    Raises:
        ValueError: If user is on trial or byok tier
    """
    from .core import get_effective_tier

    tier = get_effective_tier(user_id)
    allowed_tiers = {"managed_lite", "managed_standard", "managed_plus", "beta"}
    if tier not in allowed_tiers:
        raise ValueError(
            "Shared libraries require a managed subscription. "
            "Upgrade at legate.studio/billing to access shared libraries."
        )


def commit_and_checkpoint(db):
    """Commit transaction and force WAL checkpoint for cross-worker visibility.

    In multi-worker deployments (gunicorn with workers > 1), SQLite WAL mode
    can have visibility issues where commits in one worker aren't immediately
    visible to connections in other workers. This helper ensures data is
    durably committed and visible by:
    1. Committing the transaction
    2. Forcing a WAL RESTART checkpoint to merge WAL into main database

    Args:
        db: SQLite database connection
    """
    db.commit()
    try:
        db.execute("PRAGMA wal_checkpoint(RESTART)")
    except Exception as e:
        logger.warning(f"WAL checkpoint failed: {e}")


def get_embedding_service():
    """Get embedding service for semantic search."""
    if "mcp_embedding_service" not in g:
        from .rag.embedding_provider import get_embedding_provider
        from .rag.embedding_service import EmbeddingService

        try:
            provider = get_embedding_provider()
            g.mcp_embedding_service = EmbeddingService(provider, get_db())
        except Exception as e:
            logger.warning(f"Could not create embedding service: {e}")
            return None

    return g.mcp_embedding_service


# ============ Per-User Rate Limiting ============
# Flask-Limiter evaluates key_func during its before_request hook, which fires
# before the view function (and before @require_mcp_auth) runs. To ensure
# g.mcp_user is populated at that point, we pre-populate it here in a
# before_request handler that runs first.


@mcp_bp.before_request
def _pre_populate_mcp_user():
    """Pre-populate g.mcp_user from the JWT so the rate-limit key_func can use it.

    Flask-Limiter's before_request fires before the view function runs, so
    g.mcp_user (set by @require_mcp_auth) wouldn't be available during the
    key_func call. This handler performs a lightweight JWT decode to make the
    user_id available for rate-limit keying, without duplicating auth logic —
    the full @require_mcp_auth decorator still enforces auth and handles errors.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return  # No token — key_func will fall back to IP; auth will 401 later

    token = auth_header[7:]
    try:
        claims = verify_access_token(token)
        if claims:
            g.mcp_user = claims
    except Exception:
        pass  # Auth errors handled properly by @require_mcp_auth


def get_mcp_user_id():
    """Rate-limit key function: authenticated user ID from JWT.

    Returns a per-user key for authenticated requests so rate limits
    apply per-user (not per-IP). Falls back to IP for unauthenticated
    requests — those will be rejected by @require_mcp_auth anyway.
    """
    from flask_limiter.util import get_remote_address

    user = getattr(g, "mcp_user", None)
    if user and user.get("user_id"):
        return f"mcp:{user['user_id']}"
    return get_remote_address()


# ============ Protocol Version Discovery ============


@mcp_bp.route("", methods=["HEAD", "OPTIONS"])
@mcp_bp.route("/", methods=["HEAD", "OPTIONS"])
def mcp_head():
    """Protocol version discovery and CORS preflight.

    Claude/ChatGPT checks this to verify server compatibility.
    """
    if request.method == "OPTIONS":
        # CORS preflight
        response = current_app.make_default_options_response()
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, HEAD, OPTIONS"
        response.headers["MCP-Protocol-Version"] = MCP_PROTOCOL_VERSION
        return response

    return (
        "",
        200,
        {"MCP-Protocol-Version": MCP_PROTOCOL_VERSION, "Content-Type": "application/json"},
    )


# ============ Main JSON-RPC Handler ============


@mcp_bp.route("", methods=["POST"])
@mcp_bp.route("/", methods=["POST"])
@limiter.limit("1000 per hour", key_func=get_mcp_user_id)
@require_mcp_auth
def mcp_post():
    """Handle MCP JSON-RPC 2.0 requests.

    All MCP communication goes through this endpoint.
    Requests are routed to specific handlers based on the method.

    Rate limit: 1000 requests/hour per authenticated user (keyed by user_id).
    Generous limit designed to catch runaway agents without affecting normal
    Claude usage (~50-100 calls/hour for a human-driven session).
    Flask-Limiter fires before_request (rate check) after _pre_populate_mcp_user,
    which ensures g.mcp_user is set from the JWT before the key_func runs.
    """
    try:
        msg = request.get_json()
    except Exception:
        return jsonify({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}), 400

    if not msg:
        return jsonify({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}), 400

    method = msg.get("method")
    params = msg.get("params", {})
    msg_id = msg.get("id")

    logger.debug(f"MCP request: method={method}")

    try:
        result = dispatch_mcp_method(method, params)
        return jsonify({"jsonrpc": "2.0", "id": msg_id, "result": result})
    except MCPError as e:
        return jsonify({"jsonrpc": "2.0", "id": msg_id, "error": {"code": e.code, "message": e.message}})
    except Exception as e:
        logger.error(f"MCP handler error: {e}", exc_info=True)
        return jsonify({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32603, "message": "Internal error"}}), 500


class MCPError(Exception):
    """MCP protocol error."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


# ============ Method Dispatcher ============


def dispatch_mcp_method(method: str, params: dict) -> dict:
    """Dispatch JSON-RPC method to handler."""

    handlers = {
        "initialize": handle_initialize,
        "initialized": handle_initialized,
        "ping": handle_ping,
        "tools/list": handle_tools_list,
        "tools/call": handle_tool_call,
        "resources/list": handle_resources_list,
        "resources/read": handle_resource_read,
        "prompts/list": handle_prompts_list,
        "prompts/get": handle_prompt_get,
    }

    handler = handlers.get(method)
    if not handler:
        raise MCPError(-32601, f"Method not found: {method}")

    return handler(params)


# ============ Lifecycle Handlers ============


def handle_initialize(params: dict) -> dict:
    """Handle initialize request - negotiate capabilities."""
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {
            "tools": {"listChanged": False},
            "resources": {"listChanged": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "legate-studio", "version": "1.0.0"},
    }


def handle_initialized(params: dict) -> dict:
    """Handle initialized notification - client is ready."""
    logger.info(f"MCP client initialized: {g.mcp_user.get('sub', 'unknown')}")
    return {}


def handle_ping(params: dict) -> dict:
    """Handle ping request."""
    return {"pong": True}


# ============ Tool Definitions ============

TOOLS = [
    {
        "name": "search_library",
        "description": (
            "Hybrid search across Legate Studio library notes using AI embeddings AND keyword "
            "matching. Returns results in two buckets: high-confidence matches and 'maybe related' "
            "lower-confidence matches."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query - describe what you're looking for",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results per bucket (default: 10)",
                    "default": 10,
                },
                "category": {
                    "type": "string",
                    "description": ("Optional: filter to a specific category (e.g., 'concept', 'epiphany')"),
                },
                "expand_query": {
                    "type": "boolean",
                    "description": (
                        "Whether to expand query with synonyms/related terms for better recall (default: true)"
                    ),
                    "default": True,
                },
                "include_low_confidence": {
                    "type": "boolean",
                    "description": ("Whether to include 'maybe related' lower-confidence results (default: true)"),
                    "default": True,
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_note",
        "description": (
            "Create a new note in the Legate Studio library. The note will be saved to GitHub and "
            "indexed for search. To create a task, include task_status "
            "(pending/in_progress/blocked/done) and optionally due_date."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the note"},
                "content": {
                    "type": "string",
                    "description": "The content of the note in markdown format",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "The category for the note (e.g., 'concept', 'epiphany', 'reflection', "
                        "'glimmer', 'reminder', 'worklog')"
                    ),
                },
                "task_status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "blocked", "done"],
                    "description": (
                        "Optional: Mark this note as a task with the given status. Tasks appear in the tasks view."
                    ),
                },
                "due_date": {
                    "type": "string",
                    "description": "Optional: Due date for tasks in ISO format (YYYY-MM-DD)",
                },
                "subfolder": {
                    "type": "string",
                    "description": (
                        "Optional: Subfolder within the category to place the note (e.g., "
                        "'projects', 'backlog', 'research')"
                    ),
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["title", "content", "category"],
        },
    },
    {
        "name": "list_categories",
        "description": "List all available note categories in the Legate Studio library.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_note",
        "description": (
            "Get the full content of a specific note. Supports lookup by entry_id (most reliable), "
            "file_path (stable), or title (fuzzy match). At least one lookup param required. If "
            "multiple provided, uses fallback chain: entry_id → file_path → title."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "The entry ID (e.g., 'kb-abc12345') - most reliable lookup",
                },
                "file_path": {
                    "type": "string",
                    "description": (
                        "The file path in the library (e.g., 'concepts/2026-01-10-my-note.md') - stable identifier"
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Note title for fuzzy matching - least reliable but convenient",
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_notes",
        "description": (
            "Get the full content of multiple notes. Fetch by specific entry_ids, or by "
            "category/subfolder with optional pattern filtering. Returns note content directly - "
            "does NOT write to filesystem. Use this instead of download_notes when you need "
            "content in memory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Array of entry IDs to fetch (e.g., ['kb-abc123', 'kb-def456']). If "
                        "provided, category/subfolder are ignored."
                    ),
                },
                "category": {
                    "type": "string",
                    "description": "Category to fetch notes from (ignored if entry_ids provided)",
                },
                "subfolder": {
                    "type": "string",
                    "description": "Subfolder within category (requires category)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter filenames (e.g., '*.md', 'project-*')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum notes to return (default: 50, max: 100)",
                    "default": 50,
                },
                "include_content": {
                    "type": "boolean",
                    "description": ("Include full note content (default: true). Set false to get metadata only."),
                    "default": True,
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "append_to_note",
        "description": (
            "Append content to an existing note. Useful for journals, logs, or incrementally "
            "building up notes without fetching and replacing the entire content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "The entry ID of the note to append to",
                },
                "content": {"type": "string", "description": "Content to append to the note"},
                "separator": {
                    "type": "string",
                    "description": ("Separator between existing content and new content (default: '\\n\\n')"),
                    "default": "\n\n",
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["entry_id", "content"],
        },
    },
    {
        "name": "get_related_notes",
        "description": (
            "Find notes semantically similar to a given note using embeddings. Great for "
            "discovering related content, research, and exploration."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "The entry ID of the note to find related notes for",
                },
                "limit": {
                    "type": "integer",
                    "description": ("Maximum number of related notes to return (default: 10, max: 25)"),
                    "default": 10,
                },
                "category": {
                    "type": "string",
                    "description": "Optional: filter results to a specific category",
                },
                "include_content": {
                    "type": "boolean",
                    "description": ("Include full content of related notes (default: false, returns snippets)"),
                    "default": False,
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["entry_id"],
        },
    },
    {
        "name": "get_library_stats",
        "description": (
            "Get statistics about the library: note counts by category, total notes, recent "
            "activity, and more. Useful for understanding what's available before diving in."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_recent_notes",
        "description": "List the most recently created notes in the library.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of notes to return (default: 20)",
                    "default": 20,
                },
                "category": {
                    "type": "string",
                    "description": "Optional: filter to a specific category",
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "spawn_agent",
        "description": (
            "Queue a chord project for human approval. Links 1-5 existing library notes to create "
            "a project that will be implemented by GitHub Copilot after approval. The project "
            "appears in the Legate Studio agent queue for review."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                    "description": ("Array of 1-5 entry_ids (e.g., ['kb-abc123']) to link to this project"),
                },
                "project_name": {
                    "type": "string",
                    "description": (
                        "Slug-style name for the project (e.g., 'mcp-bedrock-adapter'). Auto-generated if not provided."
                    ),
                },
                "project_type": {
                    "type": "string",
                    "enum": ["note", "chord"],
                    "description": (
                        "Project scope: 'note' for single-PR simple projects, 'chord' for complex "
                        "multi-phase projects (default: 'note')"
                    ),
                    "default": "note",
                },
                "additional_comments": {
                    "type": "string",
                    "description": ("Additional context, instructions, or requirements for the implementation"),
                },
                "target_chord_repo": {
                    "type": "string",
                    "description": (
                        "Optional: Target an existing chord repository (e.g., "
                        "'org/repo-name.Chord') instead of creating a new one. When provided, "
                        "creates an incident issue on that repo for Copilot to work."
                    ),
                },
            },
            "required": ["note_ids"],
        },
    },
    {
        "name": "update_note",
        "description": (
            "Update an existing note in the Legate Studio library. Supports two content update "
            "modes: (1) full replacement via 'content' parameter, or (2) precise diff-based edits "
            "via 'edits' parameter. The 'edits' mode is preferred for targeted changes as it "
            "requires less context and is more precise. Cannot use both 'content' and 'edits' "
            "together."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": ("The entry ID of the note to update (e.g., 'library.concept.my-note')"),
                },
                "title": {"type": "string", "description": "New title for the note (optional)"},
                "content": {
                    "type": "string",
                    "description": (
                        "Full replacement content in markdown. Use this only for complete "
                        "rewrites. For targeted changes, prefer 'edits' instead."
                    ),
                },
                "edits": {
                    "type": "array",
                    "description": (
                        "Array of precise string replacement operations. Each edit finds and "
                        "replaces exact text. Edits are applied sequentially. Preferred over "
                        "'content' for targeted changes."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {
                                "type": "string",
                                "description": (
                                    "The exact text to find in the note content. Must be unique "
                                    "unless replace_all is true."
                                ),
                            },
                            "new_string": {
                                "type": "string",
                                "description": ("The replacement text. Can be empty to delete the old_string."),
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": (
                                    "If true, replace all occurrences. If false (default), "
                                    "old_string must appear exactly once."
                                ),
                                "default": False,
                            },
                        },
                        "required": ["old_string", "new_string"],
                    },
                },
                "category": {
                    "type": "string",
                    "description": "New category for the note (optional)",
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["entry_id"],
        },
    },
    {
        "name": "move_category",
        "description": (
            "Move a note to a different category. Updates the note's category, moves the file to "
            "the new folder in GitHub, and updates the entry ID to reflect the new category."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": ("The entry ID of the note to move (e.g., 'library.concept.my-note')"),
                },
                "new_category": {
                    "type": "string",
                    "description": "The target category to move the note to",
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["entry_id", "new_category"],
        },
    },
    {
        "name": "create_subfolder",
        "description": (
            "Create a subfolder under a category. The subfolder will be created in GitHub with a .gitkeep file."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The category under which to create the subfolder",
                },
                "subfolder_name": {
                    "type": "string",
                    "description": ("Name of the subfolder to create (e.g., 'projects', 'backlog', 'research')"),
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["category", "subfolder_name"],
        },
    },
    {
        "name": "list_subfolders",
        "description": "List all subfolders under a category.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "The category to list subfolders for"},
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "list_subfolder_contents",
        "description": "List all notes within a specific subfolder of a category.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The category containing the subfolder",
                },
                "subfolder": {
                    "type": ["string", "null"],
                    "description": ("The subfolder name, or null/empty to list notes at the category root"),
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "move_to_subfolder",
        "description": (
            "Move a note to a different subfolder within its current category. Use null or empty "
            "string for subfolder to move to the category root."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_id": {"type": "string", "description": "The entry ID of the note to move"},
                "subfolder": {
                    "type": ["string", "null"],
                    "description": ("The target subfolder name, or null/empty to move to category root"),
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["entry_id"],
        },
    },
    {
        "name": "rename_note",
        "description": (
            "Rename a note's title. This updates the note's title, regenerates the entry_id and "
            "file path based on the new title, moves the file in GitHub, and updates all "
            "references. Git-native operation with atomic commit."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_id": {"type": "string", "description": "The entry ID of the note to rename"},
                "new_title": {"type": "string", "description": "The new title for the note"},
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["entry_id", "new_title"],
        },
    },
    {
        "name": "rename_subfolder",
        "description": (
            "Rename a subfolder within a category. Moves all notes in the subfolder to the new "
            "path and updates their file paths. Git-native operation with atomic commits."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The category containing the subfolder",
                },
                "old_name": {
                    "type": "string",
                    "description": "Current name of the subfolder to rename",
                },
                "new_name": {"type": "string", "description": "New name for the subfolder"},
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["category", "old_name", "new_name"],
        },
    },
    {
        "name": "delete_note",
        "description": (
            "Delete a note from the Legate Studio library. Removes from both GitHub and local "
            "database. Requires confirmation flag."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_id": {"type": "string", "description": "The entry ID of the note to delete"},
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to confirm deletion. This is a safety check.",
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["entry_id", "confirm"],
        },
    },
    {
        "name": "list_tasks",
        "description": (
            "List notes that have been marked as tasks with their status. Filter by status, due date, or category."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "done", "blocked"],
                    "description": "Filter by task status",
                },
                "due_before": {
                    "type": "string",
                    "description": "Filter tasks due before this date (ISO format: YYYY-MM-DD)",
                },
                "due_after": {
                    "type": "string",
                    "description": "Filter tasks due after this date (ISO format: YYYY-MM-DD)",
                },
                "category": {"type": "string", "description": "Filter by note category"},
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of tasks to return (default: 50)",
                    "default": 50,
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "update_task_status",
        "description": (
            "Update or set the task status of a note. Use this to mark any existing note as a "
            "task, or to change the status of an existing task. Tasks appear in the dedicated "
            "tasks view."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "The entry ID of the note to update (e.g., 'kb-abc12345')",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "done", "blocked"],
                    "description": (
                        "Task status: pending (not started), in_progress (active), blocked (waiting), done (completed)"
                    ),
                },
                "due_date": {
                    "type": "string",
                    "description": "Optional due date in ISO format (YYYY-MM-DD)",
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["entry_id", "status"],
        },
    },
    {
        "name": "link_notes",
        "description": ("Create an explicit relationship between two notes. Links are bidirectional for discovery."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Entry ID of the source note"},
                "target_id": {"type": "string", "description": "Entry ID of the target note"},
                "link_type": {
                    "type": "string",
                    "enum": [
                        "related",
                        "depends_on",
                        "blocks",
                        "implements",
                        "references",
                        "contradicts",
                        "supports",
                    ],
                    "description": "Type of relationship (default: 'related')",
                    "default": "related",
                },
                "description": {
                    "type": "string",
                    "description": "Optional description of the relationship",
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["source_id", "target_id"],
        },
    },
    {
        "name": "get_note_context",
        "description": ("Get a note with its full context: linked notes, semantic neighbors, and related projects."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_id": {"type": "string", "description": "The entry ID of the note"},
                "include_semantic": {
                    "type": "boolean",
                    "description": "Include semantically similar notes (default: true)",
                    "default": True,
                },
                "semantic_limit": {
                    "type": "integer",
                    "description": "Max semantic neighbors to include (default: 5)",
                    "default": 5,
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["entry_id"],
        },
    },
    {
        "name": "process_motif",
        "description": (
            "Push text or markdown content into the transcript processing pipeline. Returns a job ID to check status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The text or markdown content to process",
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "text", "transcript"],
                    "description": "Content format (default: 'markdown')",
                    "default": "markdown",
                },
                "source_label": {
                    "type": "string",
                    "description": (
                        "Label for the source of this content (e.g., 'claude-conversation', 'external-doc')"
                    ),
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "get_processing_status",
        "description": "Check the status of an async processing job.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The job ID returned from process_motif",
                }
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "check_connection",
        "description": (
            "Diagnostic tool to check MCP connection status, user authentication, and GitHub App "
            "setup. Use this to troubleshoot connectivity issues."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "verify_sync_state",
        "description": (
            "Check consistency between database entries and GitHub files. Identifies notes that "
            "exist in the database but are missing from GitHub (orphaned DB entries) or exist in "
            "GitHub but missing from database. Use this to diagnose sync mismatches before running "
            "repair_sync_state. Pass library_id to check a shared library instead of your personal library."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional: check only notes in a specific category",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entries to check (default: 100, max: 500)",
                    "default": 100,
                },
                "library_id": {
                    "type": "string",
                    "description": (
                        "Optional: UUID of a shared library to check. "
                        "Caller must be the library owner. Omit for personal library."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "repair_sync_state",
        "description": (
            "Repair sync mismatches between database and GitHub. For entries that exist in the "
            "database but are missing from GitHub, recreates the GitHub file from database "
            "content. This heals orphaned database entries caused by failed creates/updates."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Specific entry IDs to repair. If not provided, repairs all orphaned entries (up to limit)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entries to repair (default: 10, max: 50)",
                    "default": 10,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": ("If true, reports what would be repaired without making changes (default: false)"),
                    "default": False,
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_assets",
        "description": (
            "List assets (images, files) in a category's assets folder. Assets can be referenced "
            "in notes using markdown: ![alt text](assets/filename.png)"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": ("Filter assets by category (optional - lists all if not specified)"),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of assets to return (default: 50)",
                    "default": 50,
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_asset",
        "description": "Get metadata for a specific asset including its markdown reference.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string", "description": "The asset ID (e.g., 'asset-abc123')"},
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["asset_id"],
        },
    },
    {
        "name": "delete_asset",
        "description": "Delete an asset from the library. Removes from both GitHub and database.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string", "description": "The asset ID to delete"},
                "confirm": {"type": "boolean", "description": "Must be true to confirm deletion"},
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["asset_id", "confirm"],
        },
    },
    {
        "name": "get_asset_reference",
        "description": (
            "Get the markdown reference for an asset to use in notes. Returns a properly formatted "
            "markdown image/link reference."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string", "description": "The asset ID"},
                "alt_text": {
                    "type": "string",
                    "description": "Optional alt text to use (overrides stored alt_text)",
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["asset_id"],
        },
    },
    {
        "name": "upload_asset",
        "description": (
            "Upload an image or file to a category's assets folder. The file content must be "
            "base64-encoded. Returns a markdown reference you can use in notes. Supported types: "
            "PNG, JPEG, GIF, WebP, SVG, PDF."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Target category for the asset (e.g., 'concept', 'epiphany')",
                },
                "filename": {
                    "type": "string",
                    "description": ("Original filename with extension (e.g., 'diagram.png', 'chart.jpg')"),
                },
                "content_base64": {"type": "string", "description": "Base64-encoded file content"},
                "alt_text": {
                    "type": "string",
                    "description": "Alt text for images (used in markdown reference)",
                },
                "description": {
                    "type": "string",
                    "description": "Optional description of the asset",
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["category", "filename", "content_base64"],
        },
    },
    {
        "name": "upload_markdown_as_note",
        "description": (
            "Upload a markdown file directly as a note. Parses any existing frontmatter and "
            "augments it with required fields. Useful for importing existing markdown files or "
            "when you already have formatted markdown content with metadata."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "markdown_content": {
                    "type": "string",
                    "description": (
                        "The full markdown content, optionally including YAML frontmatter delimited by ---"
                    ),
                },
                "category": {
                    "type": "string",
                    "description": (
                        "The category for the note (e.g., 'concept', 'epiphany'). Required if not "
                        "specified in frontmatter."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": (
                        "The title for the note. If not provided, extracted from frontmatter or first heading."
                    ),
                },
                "subfolder": {
                    "type": "string",
                    "description": "Optional: Subfolder within the category to place the note",
                },
                "preserve_frontmatter": {
                    "type": "boolean",
                    "description": (
                        "If true, preserve existing frontmatter fields like tags, dates, etc. (default: true)"
                    ),
                    "default": True,
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["markdown_content"],
        },
    },
    {
        "name": "create_category",
        "description": (
            "Create a new category in the Legate Studio library. Categories organize notes by "
            "type/purpose. Creates the category folder in GitHub."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Category name (lowercase letters, numbers, hyphens only - e.g., "
                        "'research', 'daily-journal', 'project')"
                    ),
                },
                "display_name": {
                    "type": "string",
                    "description": ("Human-readable display name (e.g., 'Research Notes', 'Daily Journal')"),
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what this category is for",
                },
                "color": {
                    "type": "string",
                    "description": (
                        "Optional hex color code for UI (e.g., '#10b981'). Defaults to indigo if not specified."
                    ),
                },
                "library_id": {
                    "type": "string",
                    "description": "Shared library ID. Omit for personal library.",
                },
            },
            "required": ["name", "display_name"],
        },
    },
    {
        "name": "sync_shared_library",
        "description": (
            "Sync a shared library's database from its GitHub repository. "
            "Only the library owner can trigger this. Pulls the latest content "
            "from the 'main' branch and updates the library's local database. "
            "Returns sync statistics (entries created, updated, errors)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "UUID of the shared library to sync",
                },
            },
            "required": ["library_id"],
        },
    },
    # ============ Draft & Merge Workflow Tools ============
    {
        "name": "create_draft",
        "description": (
            "Create a draft for a shared library. Collaborators use this to propose changes "
            "without writing directly to the library. Three modes: 'new_note' (propose a new "
            "note with title+content+category), 'edit' (propose changes to an existing note via "
            "target_entry_id+content), 'delete' (propose deletion of an existing note via "
            "target_entry_id). Only one active draft per author per target note is allowed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "UUID of the shared library to draft for",
                },
                "draft_type": {
                    "type": "string",
                    "enum": ["new_note", "edit", "delete"],
                    "description": (
                        "Type of draft: 'new_note' to propose a new note, 'edit' to propose "
                        "changes to an existing note, 'delete' to propose deletion of a note"
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Title for the new note (required for new_note drafts)",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Content for the note in markdown format "
                        "(required for new_note and edit drafts)"
                    ),
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Category for the new note (required for new_note drafts, "
                        "e.g., 'concept', 'epiphany')"
                    ),
                },
                "subfolder": {
                    "type": "string",
                    "description": "Optional subfolder within the category (for new_note drafts)",
                },
                "target_entry_id": {
                    "type": "string",
                    "description": (
                        "Entry ID of the note to edit or delete "
                        "(required for edit and delete drafts)"
                    ),
                },
            },
            "required": ["library_id", "draft_type"],
        },
    },
    {
        "name": "submit_draft",
        "description": (
            "Submit a draft for owner review. Changes the draft status from 'draft' to "
            "'submitted'. Only the draft's author can submit it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "UUID of the shared library",
                },
                "draft_id": {
                    "type": "string",
                    "description": "UUID of the draft to submit",
                },
            },
            "required": ["library_id", "draft_id"],
        },
    },
    {
        "name": "list_drafts",
        "description": (
            "List drafts for a shared library. Library owners see all submitted drafts by "
            "default. Collaborators see only their own drafts. Optional filters: status "
            "('draft', 'submitted', 'merged', 'rejected') and author (GitHub login)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "UUID of the shared library",
                },
                "status": {
                    "type": "string",
                    "enum": ["draft", "submitted", "merged", "rejected"],
                    "description": "Filter by draft status (omit to see all accessible drafts)",
                },
                "author": {
                    "type": "string",
                    "description": "Filter by author GitHub login",
                },
            },
            "required": ["library_id"],
        },
    },
    {
        "name": "review_draft",
        "description": (
            "Review a specific draft. Shows the draft content, and for edit drafts shows "
            "the original note content alongside for comparison. Accessible by the library "
            "owner and the draft's author."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "UUID of the shared library",
                },
                "draft_id": {
                    "type": "string",
                    "description": "UUID of the draft to review",
                },
            },
            "required": ["library_id", "draft_id"],
        },
    },
    {
        "name": "merge_draft",
        "description": (
            "Merge a submitted draft into the shared library. OWNER ONLY. The draft must "
            "have status 'submitted'. For 'new_note' drafts, creates the note on GitHub and "
            "in the database. For 'edit' drafts, updates the existing note. For 'delete' "
            "drafts, removes the note. Includes conflict detection: if the target note was "
            "modified after the draft was created, returns a warning. Use force=true to "
            "override the conflict check and merge anyway."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "UUID of the shared library",
                },
                "draft_id": {
                    "type": "string",
                    "description": "UUID of the draft to merge",
                },
                "force": {
                    "type": "boolean",
                    "description": (
                        "Override conflict check and merge even if the target note was "
                        "modified after the draft was created (default: false)"
                    ),
                    "default": False,
                },
            },
            "required": ["library_id", "draft_id"],
        },
    },
    {
        "name": "reject_draft",
        "description": (
            "Reject a draft with optional feedback. OWNER ONLY. Sets the draft status to "
            "'rejected'. Provide feedback to explain why the draft was rejected so the "
            "author can revise and resubmit."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "UUID of the shared library",
                },
                "draft_id": {
                    "type": "string",
                    "description": "UUID of the draft to reject",
                },
                "feedback": {
                    "type": "string",
                    "description": "Optional feedback explaining why the draft was rejected",
                },
            },
            "required": ["library_id", "draft_id"],
        },
    },
    # ============ Library Management Tools ============
    {
        "name": "create_shared_library",
        "description": (
            "Create a new shared library that other collaborators can contribute to. "
            "Requires a managed subscription tier. Provisions a private GitHub repo "
            "at Legate.Library.{slug} and initializes the library database. "
            "Returns the library_id needed to invite collaborators."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Human-readable library name (e.g., 'Team Research Notes')",
                },
                "slug": {
                    "type": "string",
                    "description": (
                        "URL-safe slug for the library (e.g., 'team-research'). "
                        "Auto-generated from name if not provided."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Optional description of the library's purpose",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_libraries",
        "description": (
            "List all libraries you have access to: your personal library plus any shared "
            "libraries where you are an active member. Returns library_id, name, slug, your "
            "role (owner/collaborator), and member count for each shared library."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "invite_collaborator",
        "description": (
            "Invite a GitHub user to collaborate on a shared library. "
            "You must be the library owner. The invitee will appear as 'invited' until "
            "they call accept_invitation. Also adds them as a GitHub repo collaborator."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "UUID of the shared library",
                },
                "github_login": {
                    "type": "string",
                    "description": "GitHub username of the person to invite",
                },
            },
            "required": ["library_id", "github_login"],
        },
    },
    {
        "name": "accept_invitation",
        "description": (
            "Accept a pending invitation to collaborate on a shared library. "
            "After accepting, you can use library_id with all note tools to read and "
            "submit drafts in the shared library."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "UUID of the shared library you were invited to",
                },
            },
            "required": ["library_id"],
        },
    },
    {
        "name": "remove_collaborator",
        "description": (
            "Remove a collaborator from a shared library. "
            "You must be the library owner. Revokes access and removes them from the "
            "GitHub repo. Their unsubmitted drafts are deleted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "UUID of the shared library",
                },
                "github_login": {
                    "type": "string",
                    "description": "GitHub username of the collaborator to remove",
                },
            },
            "required": ["library_id", "github_login"],
        },
    },
]


def handle_tools_list(params: dict) -> dict:
    """Return list of available tools."""
    return {"tools": TOOLS}


def handle_tool_call(params: dict) -> dict:
    """Handle tool invocation."""
    name = params.get("name")
    args = params.get("arguments", {})

    tool_handlers = {
        "search_library": tool_search_library,
        "create_note": tool_create_note,
        "list_categories": tool_list_categories,
        "get_note": tool_get_note,
        "get_notes": tool_get_notes,
        "append_to_note": tool_append_to_note,
        "get_related_notes": tool_get_related_notes,
        "get_library_stats": tool_get_library_stats,
        "list_recent_notes": tool_list_recent_notes,
        "spawn_agent": tool_spawn_agent,
        "update_note": tool_update_note,
        "move_category": tool_move_category,
        "create_subfolder": tool_create_subfolder,
        "list_subfolders": tool_list_subfolders,
        "list_subfolder_contents": tool_list_subfolder_contents,
        "move_to_subfolder": tool_move_to_subfolder,
        "rename_note": tool_rename_note,
        "rename_subfolder": tool_rename_subfolder,
        "delete_note": tool_delete_note,
        "list_tasks": tool_list_tasks,
        "update_task_status": tool_update_task_status,
        "link_notes": tool_link_notes,
        "get_note_context": tool_get_note_context,
        "process_motif": tool_process_motif,
        "get_processing_status": tool_get_processing_status,
        "check_connection": tool_check_connection,
        "verify_sync_state": tool_verify_sync_state,
        "repair_sync_state": tool_repair_sync_state,
        "list_assets": tool_list_assets,
        "get_asset": tool_get_asset,
        "delete_asset": tool_delete_asset,
        "get_asset_reference": tool_get_asset_reference,
        "upload_asset": tool_upload_asset,
        "upload_markdown_as_note": tool_upload_markdown_as_note,
        "create_category": tool_create_category,
        "sync_shared_library": tool_sync_shared_library,
        # Draft & Merge workflow tools
        "create_draft": tool_create_draft,
        "submit_draft": tool_submit_draft,
        "list_drafts": tool_list_drafts,
        "review_draft": tool_review_draft,
        "merge_draft": tool_merge_draft,
        "reject_draft": tool_reject_draft,
        # Library management tools
        "create_shared_library": tool_create_shared_library,
        "list_libraries": tool_list_libraries,
        "invite_collaborator": tool_invite_collaborator,
        "accept_invitation": tool_accept_invitation,
        "remove_collaborator": tool_remove_collaborator,
    }

    handler = tool_handlers.get(name)
    if not handler:
        raise MCPError(-32602, f"Unknown tool: {name}")

    try:
        result = handler(args)
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]}
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}", exc_info=True)
        return {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True}


# ============ Tool Implementations ============


def tool_search_library(args: dict) -> dict:
    """Hybrid search across library notes with optional query expansion."""
    query = args.get("query", "")
    limit = args.get("limit", 10)
    category = args.get("category")
    expand_query = args.get("expand_query", True)
    include_low_confidence = args.get("include_low_confidence", True)

    if not query:
        return {"error": "Query is required", "results": []}

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    service = get_embedding_service()

    def format_result(r: dict) -> dict:
        """Format a result for output."""
        return {
            "entry_id": r["entry_id"],
            "title": r["title"],
            "category": r.get("category"),
            "similarity": round(r.get("similarity", 0), 3),
            "match_types": r.get("match_types", []),
            "snippet": (r.get("content", "")[:300] + "...") if r.get("content") else None,
        }

    if service:
        # Use hybrid search with query expansion
        if expand_query:
            search_result = service.search_with_expansion(
                query=query,
                entry_type="knowledge",
                limit=limit,
                expand=True,
            )
        else:
            search_result = service.hybrid_search(
                query=query,
                entry_type="knowledge",
                limit=limit,
                include_low_confidence=include_low_confidence,
            )

        results = search_result.get("results", [])
        maybe_related = search_result.get("maybe_related", [])

        # Filter by category if specified
        if category:
            results = [r for r in results if r.get("category") == category]
            maybe_related = [r for r in maybe_related if r.get("category") == category]

        response = {
            "query": query,
            "results": [format_result(r) for r in results[:limit]],
            "search_type": "hybrid",
            "total_found": search_result.get("total_found", len(results)),
        }

        # Add query expansion info if used
        if expand_query and "queries_used" in search_result:
            response["queries_used"] = search_result["queries_used"]

        # Add maybe_related bucket if requested and has results
        if include_low_confidence and maybe_related:
            response["maybe_related"] = [format_result(r) for r in maybe_related[:limit]]

        return response

    else:
        # Fallback to text search
        sql = """
            SELECT entry_id, title, category, content
            FROM knowledge_entries
            WHERE title LIKE ? OR content LIKE ?
        """
        params = [f"%{query}%", f"%{query}%"]

        if category:
            sql += " AND category = ?"
            params.append(category)

        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        results = db.execute(sql, params).fetchall()

        return {
            "query": query,
            "results": [
                {
                    "entry_id": r["entry_id"],
                    "title": r["title"],
                    "category": r["category"],
                    "snippet": (r["content"][:300] + "...") if r["content"] else None,
                }
                for r in results
            ],
            "search_type": "text",
        }


def compute_content_hash(content: str) -> str:
    """Compute a stable hash of content for deduplication and integrity."""
    normalized = content.strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def generate_slug(title: str) -> str:
    """Generate a URL-safe slug from a title."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:50].strip("-")
    return slug or "untitled"


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
    base_id = f"library.{category}.{slug}"
    if content_hash:
        # Append first 6 chars of hash to disambiguate
        return f"{base_id}-{content_hash[:6]}"
    return base_id


def _generate_embedding_for_entry(entry_db_id: int, entry_id: str, content: str):
    """Generate embedding for a newly created or updated entry.

    Args:
        entry_db_id: The entry's database primary key (integer)
        entry_id: The entry's semantic ID (for logging)
        content: The entry's content text
    """

    from .core import get_api_key_for_user

    try:
        # Get user context from session if available
        user_id = None
        try:
            from flask import session

            user_id = session.get("user", {}).get("user_id")
        except RuntimeError:
            pass  # No request context

        gemini_key = None
        if user_id:
            gemini_key = get_api_key_for_user(user_id, "gemini")

        from .rag.embedding_provider import get_embedding_provider
        from .rag.embedding_service import EmbeddingService

        db = get_db()
        try:
            provider = get_embedding_provider(api_key=gemini_key)
        except RuntimeError:
            logger.debug("No embedding provider available - skipping embedding generation")
            return
        embedding_service = EmbeddingService(provider, db)

        # Generate embedding synchronously using integer database ID
        if embedding_service.generate_and_store(entry_db_id, "knowledge", content):
            logger.info(f"Generated embedding for {entry_id} (db_id={entry_db_id})")

    except Exception as e:
        logger.error(f"Failed to generate embedding for {entry_id}: {e}")


def tool_create_note(args: dict) -> dict:
    """Create a new note in the library."""
    from .rag.database import get_user_categories
    from .rag.github_service import create_file

    title = args.get("title", "").strip()
    content = args.get("content", "").strip()
    category = args.get("category", "").lower().strip()
    task_status = args.get("task_status", "").strip() if args.get("task_status") else None
    due_date = args.get("due_date", "").strip() if args.get("due_date") else None
    subfolder = args.get("subfolder", "").strip() if args.get("subfolder") else None

    if not title:
        return {"error": "Title is required"}
    if not category:
        return {"error": "Category is required"}

    # Validate subfolder name if provided (no slashes, reasonable characters)
    if subfolder and ("/" in subfolder or "\\" in subfolder):
        return {"error": ("Subfolder name cannot contain slashes. Use a simple name like 'projects' or 'backlog'.")}

    # Validate task_status if provided
    valid_statuses = {"pending", "in_progress", "blocked", "done"}
    if task_status and task_status not in valid_statuses:
        return {"error": f"Invalid task_status. Must be one of: {', '.join(sorted(valid_statuses))}"}

    # Get library db + check write permission
    try:
        db, role = get_library_db(args)
        check_write_permission(args.get("library_id"), role)
    except ValueError as e:
        return {"error": str(e)}

    # Validate category - use user's custom categories, not just defaults
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None
    categories = get_user_categories(db, user_id or "default")
    valid_categories = {c["name"] for c in categories}
    category_folders = {c["name"]: c["folder_name"] for c in categories}

    if category not in valid_categories:
        return {"error": f"Invalid category. Must be one of: {', '.join(sorted(valid_categories))}"}

    # Compute content hash for integrity/deduplication
    content_hash = compute_content_hash(content)

    # Generate slug and canonical entry_id
    slug = generate_slug(title)

    # Canonical ID format: library.{category}.{slug}
    # This matches what library_sync expects from frontmatter
    entry_id = generate_entry_id(category, title)

    # Check for entry_id collision with existing entry
    # This handles long titles that truncate to the same slug
    collision = db.execute("SELECT entry_id FROM knowledge_entries WHERE entry_id = ?", (entry_id,)).fetchone()
    if collision:
        logger.info(f"Entry ID collision detected for '{title}', disambiguating with content hash")
        entry_id = generate_entry_id(category, title, content_hash)

    # Build file path (including subfolder if provided)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    # Fallback folder name - use category name directly (singular, matching DB defaults)
    folder = category_folders.get(category, category)
    if subfolder:
        file_path = f"{folder}/{subfolder}/{date_str}-{slug}.md"
    else:
        file_path = f"{folder}/{date_str}-{slug}.md"

    # Build frontmatter - include task fields if provided
    timestamp = datetime.utcnow().isoformat() + "Z"
    frontmatter_lines = [
        "---",
        f"id: {entry_id}",
        f'title: "{title}"',
        f"category: {category}",
        f"created: {timestamp}",
        f"content_hash: {content_hash}",
        "source: mcp-claude",
        "domain_tags: []",
        "key_phrases: []",
    ]

    # Add optional fields to frontmatter
    if subfolder:
        frontmatter_lines.append(f"subfolder: {subfolder}")
    if task_status:
        frontmatter_lines.append(f"task_status: {task_status}")
    if due_date:
        frontmatter_lines.append(f"due_date: {due_date}")

    frontmatter_lines.append("---")
    frontmatter_lines.append("")
    frontmatter = "\n".join(frontmatter_lines)

    full_content = frontmatter + content

    # Create file in GitHub using user's installation token
    from .auth import get_user_installation_token
    from .core import get_user_library_repo

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None
    github_login = g.mcp_user.get("sub") if hasattr(g, "mcp_user") else None
    logger.info(f"MCP create_note: user_id={user_id}, github_login={github_login}")

    token = get_user_installation_token(user_id, "library") if user_id else None
    if not token:
        logger.warning(
            f"MCP create_note: No token for user_id={user_id} - user may need to complete GitHub App setup via web"
        )
        return {"error": "GitHub authorization required. Please re-authenticate."}

    repo = get_user_library_repo(user_id)

    # Atomic operation: Insert to DB first (in transaction), then push to GitHub.
    # If GitHub fails, rollback DB. This prevents orphaned DB entries without GitHub files.
    try:
        # Insert into local database with task fields, subfolder, and content_hash
        # This is in an implicit transaction - not committed yet
        if task_status:
            cursor = db.execute(
                """
                INSERT INTO knowledge_entries
                (entry_id, title, category, content, file_path, source_transcript,
                task_status, due_date, content_hash, subfolder, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'mcp-claude',
                ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                (
                    entry_id,
                    title,
                    category,
                    content,
                    file_path,
                    task_status,
                    due_date,
                    content_hash,
                    subfolder,
                ),
            )
        else:
            cursor = db.execute(
                """
                INSERT INTO knowledge_entries
                (entry_id, title, category, content, file_path, source_transcript,
                content_hash, subfolder, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'mcp-claude', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                (entry_id, title, category, content, file_path, content_hash, subfolder),
            )
        row = cursor.fetchone()
        entry_db_id = row[0]

        # Now push to GitHub - if this fails, we'll rollback the DB insert
        try:
            create_file(
                repo=repo,
                path=file_path,
                content=full_content,
                message=f"Create note via MCP: {title}",
                token=token,
            )
        except Exception as github_err:
            # GitHub failed - rollback the database insert
            logger.error(f"GitHub create failed for {entry_id}, rolling back DB: {github_err}")
            db.rollback()
            return {"error": f"Failed to create file in GitHub: {str(github_err)}"}

        # Both succeeded - commit the database
        commit_and_checkpoint(db)

    except Exception as db_err:
        logger.error(f"Database insert failed for {entry_id}: {db_err}")
        try:
            db.rollback()
        except Exception:
            pass
        return {"error": f"Failed to create note in database: {str(db_err)}"}

    # Generate embedding for the new entry using integer database ID
    _generate_embedding_for_entry(entry_db_id, entry_id, content)

    logger.info(f"MCP created note: {entry_id} - {title}" + (f" [task:{task_status}]" if task_status else ""))

    result = {
        "success": True,
        "entry_id": entry_id,
        "title": title,
        "category": category,
        "file_path": file_path,
        "available_categories": sorted(valid_categories),
    }

    # Include optional fields in response if set
    if subfolder:
        result["subfolder"] = subfolder
    if task_status:
        result["task_status"] = task_status
    if due_date:
        result["due_date"] = due_date

    return result


def tool_list_categories(args: dict) -> dict:
    """List all available categories."""
    from .rag.database import get_user_categories

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None
    categories = get_user_categories(db, user_id or "default")

    # Get counts per category
    counts = db.execute("""
        SELECT category, COUNT(*) as count
        FROM knowledge_entries
        GROUP BY category
    """).fetchall()
    count_map = {r["category"]: r["count"] for r in counts}

    return {
        "categories": [
            {
                "name": c["name"],
                "display_name": c["display_name"],
                "description": c.get("description"),
                "note_count": count_map.get(c["name"], 0),
            }
            for c in categories
        ]
    }


def tool_get_note(args: dict) -> dict:
    """Get full content of a specific note with multi-method lookup."""
    entry_id = args.get("entry_id", "").strip() if args.get("entry_id") else None
    file_path = args.get("file_path", "").strip() if args.get("file_path") else None
    title = args.get("title", "").strip() if args.get("title") else None

    if not entry_id and not file_path and not title:
        return {"error": "At least one lookup parameter required: entry_id, file_path, or title"}

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    entry = None
    lookup_method = None

    # Fallback chain: entry_id → file_path → title
    if entry_id:
        entry = db.execute(
            """
            SELECT entry_id, title, category, content, file_path,
                   created_at, updated_at, chord_status, chord_repo, task_status, due_date
            FROM knowledge_entries
            WHERE entry_id = ?
            """,
            (entry_id,),
        ).fetchone()
        lookup_method = "entry_id"

    if not entry and file_path:
        entry = db.execute(
            """
            SELECT entry_id, title, category, content, file_path,
                   created_at, updated_at, chord_status, chord_repo, task_status, due_date
            FROM knowledge_entries
            WHERE file_path = ?
            """,
            (file_path,),
        ).fetchone()
        lookup_method = "file_path"

    if not entry and title:
        # Fuzzy match: case-insensitive LIKE search
        entry = db.execute(
            """
            SELECT entry_id, title, category, content, file_path,
                   created_at, updated_at, chord_status, chord_repo, task_status, due_date
            FROM knowledge_entries
            WHERE LOWER(title) LIKE LOWER(?)
            ORDER BY
                CASE WHEN LOWER(title) = LOWER(?) THEN 0 ELSE 1 END,
                updated_at DESC
            LIMIT 1
            """,
            (f"%{title}%", title),
        ).fetchone()
        lookup_method = "title"

    if not entry:
        search_term = entry_id or file_path or title
        return {"error": f"Note not found: {search_term}"}

    return {
        "entry_id": entry["entry_id"],
        "title": entry["title"],
        "category": entry["category"],
        "content": entry["content"],
        "file_path": entry["file_path"],
        "created_at": entry["created_at"],
        "updated_at": entry["updated_at"],
        "chord_status": entry["chord_status"],
        "chord_repo": entry["chord_repo"],
        "task_status": entry["task_status"],
        "due_date": entry["due_date"],
        "lookup_method": lookup_method,
    }


def tool_get_notes(args: dict) -> dict:
    """Get full content of multiple notes by entry_ids or category/subfolder."""
    import fnmatch

    entry_ids = args.get("entry_ids", [])
    category = args.get("category", "").strip() if args.get("category") else None
    subfolder = args.get("subfolder", "").strip() if args.get("subfolder") else None
    pattern = args.get("pattern", "").strip() if args.get("pattern") else None
    limit = min(args.get("limit", 50), 100)  # Cap at 100
    include_content = args.get("include_content", True)

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    notes = []

    if entry_ids:
        # Fetch specific notes by entry_id
        if not isinstance(entry_ids, list):
            return {"error": "entry_ids must be an array"}

        if len(entry_ids) > 100:
            return {"error": "Maximum 100 entry_ids per request"}

        # Batch query for efficiency
        placeholders = ",".join(["?" for _ in entry_ids])
        rows = db.execute(
            f"""
            SELECT entry_id, title, category, content, file_path, subfolder,
                   created_at, updated_at, task_status, due_date
            FROM knowledge_entries
            WHERE entry_id IN ({placeholders})
            """,
            entry_ids,
        ).fetchall()

        # Create lookup for ordering
        results_map = {r["entry_id"]: r for r in rows}

        # Return in requested order, noting any not found
        not_found = []
        for eid in entry_ids:
            if eid in results_map:
                row = results_map[eid]
                note = {
                    "entry_id": row["entry_id"],
                    "title": row["title"],
                    "category": row["category"],
                    "file_path": row["file_path"],
                    "subfolder": row["subfolder"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "task_status": row["task_status"],
                    "due_date": row["due_date"],
                }
                if include_content:
                    note["content"] = row["content"]
                notes.append(note)
            else:
                not_found.append(eid)

        return {
            "notes": notes,
            "count": len(notes),
            "not_found": not_found if not_found else None,
            "lookup_method": "entry_ids",
        }

    elif category:
        # Fetch by category/subfolder
        if subfolder:
            rows = db.execute(
                """
                SELECT entry_id, title, category, content, file_path, subfolder,
                       created_at, updated_at, task_status, due_date
                FROM knowledge_entries
                WHERE category = ? AND subfolder = ?
                ORDER BY file_path ASC
                """,
                (category, subfolder),
            ).fetchall()
        else:
            rows = db.execute(
                """
                SELECT entry_id, title, category, content, file_path, subfolder,
                       created_at, updated_at, task_status, due_date
                FROM knowledge_entries
                WHERE category = ?
                ORDER BY file_path ASC
                """,
                (category,),
            ).fetchall()

        # Apply pattern filter if provided
        for row in rows:
            if pattern:
                filename = row["file_path"].split("/")[-1] if row["file_path"] else ""
                if not fnmatch.fnmatch(filename, pattern):
                    continue

            note = {
                "entry_id": row["entry_id"],
                "title": row["title"],
                "category": row["category"],
                "file_path": row["file_path"],
                "subfolder": row["subfolder"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "task_status": row["task_status"],
                "due_date": row["due_date"],
            }
            if include_content:
                note["content"] = row["content"]
            notes.append(note)

            if len(notes) >= limit:
                break

        return {
            "notes": notes,
            "count": len(notes),
            "category": category,
            "subfolder": subfolder,
            "pattern": pattern,
            "lookup_method": "category",
        }

    else:
        return {"error": "Either entry_ids or category is required"}


def tool_append_to_note(args: dict) -> dict:
    """Append content to an existing note."""
    from .rag.github_service import commit_file, get_file_content

    entry_id = args.get("entry_id", "").strip()
    append_content = args.get("content", "")
    separator = args.get("separator", "\n\n")

    if not entry_id:
        return {"error": "entry_id is required"}
    if not append_content:
        return {"error": "content is required"}

    try:
        db, role = get_library_db(args)
        check_write_permission(args.get("library_id"), role)
    except ValueError as e:
        return {"error": str(e)}

    # Get existing note
    entry = db.execute(
        """
        SELECT id, entry_id, title, category, content, file_path
        FROM knowledge_entries
        WHERE entry_id = ?
        """,
        (entry_id,),
    ).fetchone()

    if not entry:
        return {"error": f"Note not found: {entry_id}"}

    # Build new content
    existing_content = entry["content"] or ""
    new_content = existing_content + separator + append_content

    # Get current file from GitHub to preserve frontmatter
    try:
        github_content = get_file_content(entry["file_path"])
        if github_content:
            # Find where body starts (after frontmatter)
            if github_content.startswith("---"):
                end_fm = github_content.find("---", 3)
                if end_fm != -1:
                    frontmatter = github_content[: end_fm + 3]
                    full_content = frontmatter + "\n\n" + new_content
                else:
                    full_content = new_content
            else:
                full_content = new_content
        else:
            full_content = new_content
    except Exception:
        # Fallback: just use the content without frontmatter preservation
        full_content = new_content

    # Commit to GitHub
    try:
        commit_file(entry["file_path"], full_content, f"Append to {entry['title']}")
    except Exception as e:
        return {"error": f"Failed to update GitHub: {str(e)}"}

    # Update database
    db.execute(
        """
        UPDATE knowledge_entries
        SET content = ?, updated_at = CURRENT_TIMESTAMP
        WHERE entry_id = ?
        """,
        (new_content, entry_id),
    )
    db.commit()

    # Regenerate embedding
    try:
        _generate_embedding_for_entry(entry["id"], entry_id, new_content)
    except Exception as emb_err:
        logger.warning(f"Failed to regenerate embedding for {entry_id}: {emb_err}")

    return {
        "success": True,
        "entry_id": entry_id,
        "title": entry["title"],
        "appended_length": len(append_content),
        "new_total_length": len(new_content),
    }


def tool_get_related_notes(args: dict) -> dict:
    """Find notes semantically similar to a given note."""
    entry_id = args.get("entry_id", "").strip()
    limit = min(args.get("limit", 10), 25)  # Cap at 25
    category = args.get("category", "").strip() if args.get("category") else None
    include_content = args.get("include_content", False)

    if not entry_id:
        return {"error": "entry_id is required"}

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    # Get the source note
    entry = db.execute(
        """
        SELECT entry_id, title, category, content
        FROM knowledge_entries
        WHERE entry_id = ?
        """,
        (entry_id,),
    ).fetchone()

    if not entry:
        return {"error": f"Note not found: {entry_id}"}

    # Use embedding service to find similar notes
    service = get_embedding_service()
    if not service:
        return {"error": "Embedding service not available. OPENAI_API_KEY may not be configured."}

    try:
        # Use the note's title + content snippet as the query
        query_text = entry["title"] + " " + (entry["content"][:1000] if entry["content"] else "")

        search_result = service.hybrid_search(
            query=query_text,
            entry_type="knowledge",
            limit=limit + 1,  # +1 to exclude self
            include_low_confidence=False,
        )

        related = []
        for r in search_result.get("results", []):
            # Skip the source note itself
            if r["entry_id"] == entry_id:
                continue

            # Filter by category if specified
            if category and r.get("category") != category:
                continue

            note = {
                "entry_id": r["entry_id"],
                "title": r["title"],
                "category": r.get("category"),
                "similarity": round(r.get("similarity", 0), 3),
            }

            if include_content:
                note["content"] = r.get("content")
            else:
                # Include a snippet
                content = r.get("content", "")
                note["snippet"] = (content[:300] + "...") if len(content) > 300 else content

            related.append(note)

            if len(related) >= limit:
                break

        return {
            "source_note": {
                "entry_id": entry["entry_id"],
                "title": entry["title"],
                "category": entry["category"],
            },
            "related_notes": related,
            "count": len(related),
        }

    except Exception as e:
        logger.error(f"Failed to find related notes: {e}", exc_info=True)
        return {"error": f"Search failed: {str(e)}"}


def tool_get_library_stats(args: dict) -> dict:
    """Get statistics about the library."""
    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    # Total notes
    total = db.execute("SELECT COUNT(*) as count FROM knowledge_entries").fetchone()["count"]

    # Notes by category
    by_category = db.execute(
        """
        SELECT category, COUNT(*) as count
        FROM knowledge_entries
        GROUP BY category
        ORDER BY count DESC
        """
    ).fetchall()

    # Notes by task status
    by_task_status = db.execute(
        """
        SELECT task_status, COUNT(*) as count
        FROM knowledge_entries
        WHERE task_status IS NOT NULL
        GROUP BY task_status
        ORDER BY count DESC
        """
    ).fetchall()

    # Recent activity (notes created/updated in last 7 days)
    recent_created = db.execute(
        """
        SELECT COUNT(*) as count
        FROM knowledge_entries
        WHERE created_at >= datetime('now', '-7 days')
        """
    ).fetchone()["count"]

    recent_updated = db.execute(
        """
        SELECT COUNT(*) as count
        FROM knowledge_entries
        WHERE updated_at >= datetime('now', '-7 days')
          AND updated_at != created_at
        """
    ).fetchone()["count"]

    # Most recent note
    most_recent = db.execute(
        """
        SELECT entry_id, title, category, created_at
        FROM knowledge_entries
        ORDER BY created_at DESC
        LIMIT 1
        """
    ).fetchone()

    # Subfolder count
    subfolder_count = db.execute(
        """
        SELECT COUNT(DISTINCT subfolder) as count
        FROM knowledge_entries
        WHERE subfolder IS NOT NULL AND subfolder != ''
        """
    ).fetchone()["count"]

    return {
        "total_notes": total,
        "by_category": [{"category": r["category"], "count": r["count"]} for r in by_category],
        "by_task_status": [{"status": r["task_status"], "count": r["count"]} for r in by_task_status]
        if by_task_status
        else None,
        "subfolder_count": subfolder_count,
        "recent_activity": {
            "created_last_7_days": recent_created,
            "updated_last_7_days": recent_updated,
        },
        "most_recent_note": {
            "entry_id": most_recent["entry_id"],
            "title": most_recent["title"],
            "category": most_recent["category"],
            "created_at": most_recent["created_at"],
        }
        if most_recent
        else None,
    }


def tool_list_recent_notes(args: dict) -> dict:
    """List recently created notes."""
    limit = min(args.get("limit", 20), 100)  # Cap at 100
    category = args.get("category")

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    if category:
        notes = db.execute(
            """
            SELECT entry_id, title, category, created_at
            FROM knowledge_entries
            WHERE category = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (category, limit),
        ).fetchall()
    else:
        notes = db.execute(
            """
            SELECT entry_id, title, category, created_at
            FROM knowledge_entries
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return {
        "notes": [
            {
                "entry_id": n["entry_id"],
                "title": n["title"],
                "category": n["category"],
                "created_at": n["created_at"],
            }
            for n in notes
        ],
        "count": len(notes),
    }


def _create_incident_on_chord(
    chord_repo: str,
    notes: list,
    additional_comments: str,
    user_id: str = None,
    github_login: str = None,
) -> dict:
    """Dispatch an incident to Conduct for an existing chord repository.

    Args:
        chord_repo: Full repo name (org/repo-name.Chord)
        notes: List of note dicts with entry_id, title, content
        additional_comments: Additional context for the incident
        user_id: User ID for multi-tenant mode
        github_login: GitHub username (org) for dispatch

    Returns:
        dict with success/error and dispatch details
    """
    import secrets

    import requests as http_requests

    from .auth import get_user_installation_token

    # Get user token in multi-tenant mode
    token = get_user_installation_token(user_id, "library") if user_id else None
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    # Use user's GitHub login as org, fallback to env var
    org = github_login or os.environ.get("LEGATO_ORG", "bobbyhiddn")
    conduct_repo = os.environ.get("CONDUCT_REPO", "Legato.Conduct")

    primary = notes[0]

    # Build issue title (Conduct will use this)
    issue_title = primary["title"]

    # Build tasker body for the incident
    notes_section = "\n".join([f"- **{n['title']}** (`{n['entry_id']}`)" for n in notes])
    tasker_body = f"""## Incident: {primary["title"]}

### Linked Notes
{notes_section}

### Context
{primary["content"][:1500] if primary["content"] else "No content"}

### Additional Comments
{additional_comments if additional_comments else "None provided"}

---
*Incident dispatched via MCP by Claude | {len(notes)} note(s) linked*
"""

    # Generate a queue_id for tracking
    queue_id = f"incident-{secrets.token_hex(6)}"

    # Dispatch to Conduct with target_repo to create incident on existing chord
    payload = {
        "event_type": "spawn-agent",
        "client_payload": {
            "queue_id": queue_id,
            "target_repo": chord_repo,
            "issue_title": issue_title,
            "tasker_body": tasker_body,
        },
    }

    try:
        response = http_requests.post(
            f"https://api.github.com/repos/{org}/{conduct_repo}/dispatches",
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=15,
        )

        # 204 No Content = success for repository_dispatch
        if response.status_code == 204:
            logger.info(f"Dispatched incident to Conduct for {chord_repo}: {queue_id}")

            return {
                "success": True,
                "incident_dispatched": True,
                "queue_id": queue_id,
                "chord_repo": chord_repo,
                "notes_linked": len(notes),
                "note_ids": [n["entry_id"] for n in notes],
                "message": (f"Incident dispatched to Conduct for {chord_repo}. Copilot will work this issue."),
            }
        else:
            logger.error(f"Dispatch failed: {response.status_code} - {response.text}")
            return {
                "error": f"Failed to dispatch incident: HTTP {response.status_code}",
                "detail": response.text,
            }

    except Exception as e:
        logger.error(f"Failed to dispatch incident for {chord_repo}: {e}")
        return {"error": f"Failed to dispatch incident: {str(e)}"}


def tool_spawn_agent(args: dict) -> dict:
    """Queue a chord project from library notes for human approval,
    or create an incident on an existing chord."""
    import re
    import secrets

    from .rag.database import get_connection, get_db_path

    note_ids = args.get("note_ids", [])
    project_name = args.get("project_name", "").strip()
    project_type = args.get("project_type", "note").lower()
    additional_comments = args.get("additional_comments", "").strip()
    target_chord_repo = args.get("target_chord_repo", "").strip()

    # Validate note_ids
    if not note_ids:
        return {"error": "At least one note_id is required"}
    if len(note_ids) > 5:
        return {"error": "Maximum 5 notes can be linked to a project"}
    if not isinstance(note_ids, list):
        note_ids = [note_ids]

    # Validate project_type
    if project_type not in ("note", "chord"):
        project_type = "note"

    # Look up all the notes
    db = get_db()
    notes = []
    for nid in note_ids:
        entry = db.execute(
            (
                "SELECT entry_id, title, category, content, domain_tags, key_phrases FROM"
                "knowledge_entries WHERE entry_id = ?"
            ),
            (nid.strip(),),
        ).fetchone()
        if entry:
            notes.append(dict(entry))
        else:
            return {"error": f"Note not found: {nid}"}

    if not notes:
        return {"error": "No valid notes found"}

    # Use first note as primary
    primary = notes[0]

    # Get user context from MCP auth (needed for both incident and queue flows)
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None
    github_login = g.mcp_user.get("sub") if hasattr(g, "mcp_user") else None

    # If targeting an existing chord, create an incident issue instead of queuing
    if target_chord_repo:
        return _create_incident_on_chord(target_chord_repo, notes, additional_comments, user_id, github_login)

    # Generate project name if not provided
    if not project_name:
        # Create slug from first note's title
        slug = re.sub(r"[^a-z0-9]+", "-", primary["title"].lower()).strip("-")
        project_name = slug[:50]  # Limit length

    # Generate queue_id
    queue_id = f"aq-{secrets.token_hex(6)}"

    # Build signal JSON
    repo_suffix = "Chord" if project_type == "chord" else "Note"
    signal_json = {
        "title": primary["title"],
        "intent": primary["content"][:500] if primary["content"] else "",
        "domain_tags": primary.get("domain_tags", "").split(",") if primary.get("domain_tags") else [],
        "source_notes": [n["entry_id"] for n in notes],
        "additional_comments": additional_comments,
        "path": f"{project_name}.{repo_suffix}",
    }

    # Build tasker body
    notes_section = "\n".join([f"- **{n['title']}** (`{n['entry_id']}`)" for n in notes])
    tasker_body = f"""## Tasker: {primary["title"]}

### Linked Notes
{notes_section}

### Context
{primary["content"][:1000] if primary["content"] else "No content"}

### Additional Comments
{additional_comments if additional_comments else "None provided"}

---
*Queued via MCP by Claude | {len(notes)} note(s) linked*
"""

    # Build description
    if len(notes) > 1:
        titles = ", ".join(n["title"][:30] for n in notes)
        description = f"Multi-note chord linking {len(notes)} notes: {titles}"
    else:
        description = primary["content"][:200] if primary["content"] else primary["title"]

    # Build initial comments array with Claude's comment if provided
    initial_comments = []
    if additional_comments:
        initial_comments.append(
            {
                "text": additional_comments,
                "author": "claude",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

    # Insert into agent_queue
    try:
        agents_db = get_connection(get_db_path("agents.db"))

        agents_db.execute(
            """
            INSERT INTO agent_queue
            (queue_id, project_name, project_type, title, description,
             signal_json, tasker_body, source_transcript,
             related_entry_id, comments, status, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                queue_id,
                project_name,
                project_type,
                primary["title"],
                description,
                json.dumps(signal_json),
                tasker_body,
                "mcp-claude",
                ",".join(n["entry_id"] for n in notes),
                json.dumps(initial_comments),
                user_id,  # Multi-tenant: isolate by user
            ),
        )
        commit_and_checkpoint(agents_db)

        logger.info(f"MCP queued agent: {queue_id} - {project_name} ({len(notes)} notes)")

        return {
            "success": True,
            "queue_id": queue_id,
            "project_name": project_name,
            "project_type": project_type,
            "notes_linked": len(notes),
            "note_ids": [n["entry_id"] for n in notes],
            "message": (f"Project '{project_name}' queued for approval. Visit /agents in Legate Studio to approve."),
        }

    except Exception as e:
        logger.error(f"Failed to queue agent: {e}")
        return {"error": f"Failed to queue project: {str(e)}"}


def apply_edits(content: str, edits: list) -> tuple[str, list, str | None]:
    """Apply a sequence of string replacement edits to content.

    Args:
        content: The original content string
        edits: List of edit operations, each with:
            - old_string: Text to find (required)
            - new_string: Replacement text (required, can be empty)
            - replace_all: If True, replace all occurrences (default False)

    Returns:
        Tuple of (modified_content, applied_edits, error_message)
        - If successful: (new_content, list_of_applied, None)
        - If error: (original_content, [], error_message)
    """
    if not edits:
        return content, [], "No edits provided"

    modified = content
    applied = []

    for i, edit in enumerate(edits):
        old_string = edit.get("old_string", "")
        new_string = edit.get("new_string", "")
        replace_all = edit.get("replace_all", False)

        # Validate old_string is present
        if not old_string:
            return content, [], f"Edit {i + 1}: 'old_string' is required and cannot be empty"

        # new_string can be empty (for deletions), but must be present
        if "new_string" not in edit:
            return (
                content,
                [],
                f"Edit {i + 1}: 'new_string' is required (can be empty for deletions)",
            )

        # Count occurrences
        count = modified.count(old_string)

        if count == 0:
            # Provide helpful context for debugging
            preview = old_string[:100] + "..." if len(old_string) > 100 else old_string
            return (
                content,
                [],
                f"Edit {i + 1}: 'old_string' not found in content. Searched for: {repr(preview)}",
            )

        if count > 1 and not replace_all:
            return (
                content,
                [],
                (
                    f"Edit {i + 1}: 'old_string' appears {count} times in content. "
                    "Set 'replace_all': true to replace all occurrences,"
                    " or provide more context to make it unique."
                ),
            )

        # Apply the edit
        if replace_all:
            modified = modified.replace(old_string, new_string)
            applied.append(
                {
                    "edit_index": i + 1,
                    "occurrences_replaced": count,
                    "old_preview": old_string[:50] + "..." if len(old_string) > 50 else old_string,
                    "new_preview": new_string[:50] + "..." if len(new_string) > 50 else new_string,
                }
            )
        else:
            modified = modified.replace(old_string, new_string, 1)
            applied.append(
                {
                    "edit_index": i + 1,
                    "occurrences_replaced": 1,
                    "old_preview": old_string[:50] + "..." if len(old_string) > 50 else old_string,
                    "new_preview": new_string[:50] + "..." if len(new_string) > 50 else new_string,
                }
            )

    return modified, applied, None


def tool_update_note(args: dict) -> dict:
    """Update an existing note in both GitHub and local database.

    Supports two content update modes:
    1. Full replacement: Provide 'content' with the complete new content
    2. Diff-based edits: Provide 'edits' array with precise string replacements

    The 'edits' mode is preferred for targeted changes as it:
    - Requires less context (only the strings to find and replace)
    - Is more precise and auditable
    - Reduces risk of accidentally changing other content

    If the file doesn't exist in GitHub (sync mismatch), this will auto-repair
    by creating the file from the database content.
    """
    from .rag.database import get_user_categories
    from .rag.github_service import commit_file, create_file, get_file_content

    entry_id = args.get("entry_id", "").strip()
    new_title = args.get("title", "").strip() if args.get("title") else None
    new_content = args.get("content", "").strip() if args.get("content") else None
    new_category = args.get("category", "").lower().strip() if args.get("category") else None
    edits = args.get("edits")  # List of {old_string, new_string, replace_all?}

    if not entry_id:
        return {"error": "entry_id is required"}

    # Validate mutual exclusivity of content and edits
    if new_content and edits:
        return {
            "error": (
                "Cannot use both 'content' and 'edits'. Use 'content' for full replacement or "
                "'edits' for targeted changes."
            )
        }

    # Validate edits structure if provided
    if edits is not None:
        if not isinstance(edits, list):
            return {"error": "'edits' must be an array of edit operations"}
        if len(edits) == 0:
            return {"error": "'edits' array cannot be empty"}

    # At least one update must be provided
    has_content_update = new_content is not None or (edits is not None and len(edits) > 0)
    if not new_title and not has_content_update and not new_category:
        return {"error": "At least one of title, content, edits, or category must be provided"}

    db = get_db()

    # Get existing note (include id for embedding generation)
    entry = db.execute(
        """
        SELECT id, entry_id, title, category, content, file_path
        FROM knowledge_entries
        WHERE entry_id = ?
        """,
        (entry_id,),
    ).fetchone()

    if not entry:
        return {"error": f"Note not found: {entry_id}"}

    # Get user context from MCP auth
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None

    # Validate new category if provided
    if new_category:
        categories = get_user_categories(db, user_id or "default")
        valid_categories = {c["name"] for c in categories}
        if new_category not in valid_categories:
            return {"error": f"Invalid category. Must be one of: {', '.join(sorted(valid_categories))}"}

    # Use existing values as defaults
    title = new_title or entry["title"]
    category = new_category or entry["category"]
    file_path = entry["file_path"]

    # Determine final content - supports both full replacement and diff-based edits
    content_changed = False
    applied_edits_info = []

    if edits:
        # Apply diff-based edits to existing content
        existing_content = entry["content"]
        content, applied_edits_info, edit_error = apply_edits(existing_content, edits)
        if edit_error:
            return {"error": edit_error}
        content_changed = True
    elif new_content is not None:
        # Full content replacement
        content = new_content
        content_changed = True
    else:
        # No content change
        content = entry["content"]

    # Recompute content_hash if content changed (via edits or full replacement)
    new_content_hash = compute_content_hash(content) if content_changed else None

    # Get user's installation token
    from .auth import get_user_installation_token
    from .core import get_user_library_repo

    token = get_user_installation_token(user_id, "library") if user_id else None
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    repo = get_user_library_repo(user_id)

    try:
        file_missing_in_github = False
        current_content = get_file_content(repo, file_path, token)
        if current_content:
            # Parse existing frontmatter
            if current_content.startswith("---"):
                parts = current_content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter_lines = parts[1].strip().split("\n")
                    # Update frontmatter fields
                    new_frontmatter_lines = []
                    has_content_hash = False
                    for line in frontmatter_lines:
                        if line.startswith("title:") and new_title:
                            new_frontmatter_lines.append(f'title: "{title}"')
                        elif line.startswith("category:") and new_category:
                            new_frontmatter_lines.append(f"category: {category}")
                        elif line.startswith("content_hash:") and new_content_hash:
                            new_frontmatter_lines.append(f"content_hash: {new_content_hash}")
                            has_content_hash = True
                        else:
                            new_frontmatter_lines.append(line)
                    # Add content_hash if it wasn't in frontmatter but content changed
                    if new_content_hash and not has_content_hash:
                        new_frontmatter_lines.append(f"content_hash: {new_content_hash}")
                    full_content = f"---\n{chr(10).join(new_frontmatter_lines)}\n---\n\n{content}"
                else:
                    full_content = content
            else:
                full_content = content
        else:
            # File doesn't exist in GitHub - this is a sync mismatch
            # Auto-repair by creating the file from database content
            file_missing_in_github = True
            logger.warning(
                f"Sync mismatch detected: {entry_id} exists in DB but not in GitHub at {file_path}. Auto-repairing."
            )
            timestamp = datetime.utcnow().isoformat() + "Z"
            content_hash = compute_content_hash(content)
            full_content = f"""---
id: {entry_id}
title: "{title}"
category: {category}
created: {timestamp}
content_hash: {content_hash}
source: mcp-claude
domain_tags: []
key_phrases: []
---

{content}"""

        # Commit to GitHub (or create if file was missing)
        if file_missing_in_github:
            # Auto-repair: create the missing file
            create_file(
                repo=repo,
                path=file_path,
                content=full_content,
                message=f"Repair sync: recreate note via MCP: {title}",
                token=token,
            )
        else:
            commit_file(
                repo=repo,
                path=file_path,
                content=full_content,
                message=f"Update note via MCP: {title}",
                token=token,
            )

        # Update local database (include content_hash if content changed)
        if new_content_hash:
            db.execute(
                """
                UPDATE knowledge_entries
                SET title = ?, category = ?, content = ?,
                content_hash = ?, updated_at = CURRENT_TIMESTAMP
                WHERE entry_id = ?
                """,
                (title, category, content, new_content_hash, entry_id),
            )
        else:
            db.execute(
                """
                UPDATE knowledge_entries
                SET title = ?, category = ?, content = ?, updated_at = CURRENT_TIMESTAMP
                WHERE entry_id = ?
                """,
                (title, category, content, entry_id),
            )
        commit_and_checkpoint(db)

        # Regenerate embedding if content changed (use integer database ID)
        if content_changed:
            try:
                _generate_embedding_for_entry(entry["id"], entry_id, content)
            except Exception as emb_err:
                logger.warning(f"Failed to regenerate embedding for {entry_id}: {emb_err}")

        logger.info(f"MCP updated note: {entry_id} - {title}")

        # Build response
        response = {
            "success": True,
            "entry_id": entry_id,
            "title": title,
            "category": category,
            "file_path": file_path,
            "changes": {
                "title": new_title is not None,
                "content": content_changed,
                "category": new_category is not None,
            },
        }

        # Include edit details when using diff-based mode
        if applied_edits_info:
            response["edits_applied"] = applied_edits_info
            response["edit_mode"] = "diff"
        elif content_changed:
            response["edit_mode"] = "full_replacement"

        # Indicate if sync was repaired
        if file_missing_in_github:
            response["sync_repaired"] = True
            response["message"] = "File was missing in GitHub and has been recreated from database content."

        return response

    except Exception as e:
        logger.error(f"Failed to update note: {e}")
        return {"error": f"Failed to update note: {str(e)}"}


def tool_move_category(args: dict) -> dict:
    """Move a note to a different category.

    This operation:
    1. Validates the new category exists
    2. Creates the file in the new category folder
    3. Deletes the file from the old location
    4. Updates the database with new category and entry_id
    5. Regenerates the embedding
    """
    from .rag.database import get_user_categories
    from .rag.github_service import create_file, delete_file, get_file_content

    entry_id = args.get("entry_id", "").strip()
    new_category = args.get("new_category", "").lower().strip()

    if not entry_id:
        return {"error": "entry_id is required"}
    if not new_category:
        return {"error": "new_category is required"}

    db = get_db()
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None

    # Validate new category
    categories = get_user_categories(db, user_id or "default")
    valid_categories = {c["name"] for c in categories}
    category_folders = {c["name"]: c["folder_name"] for c in categories}

    if new_category not in valid_categories:
        return {"error": f"Invalid category. Must be one of: {', '.join(sorted(valid_categories))}"}

    # Get existing note
    entry = db.execute(
        """
        SELECT id, entry_id, title, category, content, file_path, content_hash
        FROM knowledge_entries
        WHERE entry_id = ?
        """,
        (entry_id,),
    ).fetchone()

    if not entry:
        return {"error": f"Note not found: {entry_id}"}

    old_category = entry["category"]
    if old_category == new_category:
        return {"error": f"Note is already in category '{new_category}'"}

    title = entry["title"]
    content = entry["content"]
    old_file_path = entry["file_path"]
    content_hash = entry["content_hash"] or compute_content_hash(content)
    entry_db_id = entry["id"]

    # Generate new entry_id for the new category
    new_entry_id = generate_entry_id(new_category, title)

    # Check for collision with existing entry
    collision = db.execute(
        "SELECT entry_id FROM knowledge_entries WHERE entry_id = ? AND entry_id != ?",
        (new_entry_id, entry_id),
    ).fetchone()
    if collision:
        new_entry_id = generate_entry_id(new_category, title, content_hash)

    # Get user's installation token
    from .auth import get_user_installation_token
    from .core import get_user_library_repo

    token = get_user_installation_token(user_id, "library") if user_id else None
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    repo = get_user_library_repo(user_id)

    try:
        # Get current file content from GitHub to preserve frontmatter structure
        current_content = get_file_content(repo, old_file_path, token)

        if current_content:
            # Update frontmatter with new category and entry_id
            if current_content.startswith("---"):
                parts = current_content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter_lines = parts[1].strip().split("\n")
                    new_frontmatter_lines = []
                    for line in frontmatter_lines:
                        if line.startswith("id:"):
                            new_frontmatter_lines.append(f"id: {new_entry_id}")
                        elif line.startswith("category:"):
                            new_frontmatter_lines.append(f"category: {new_category}")
                        else:
                            new_frontmatter_lines.append(line)
                    full_content = f"---\n{chr(10).join(new_frontmatter_lines)}\n---{parts[2]}"
                else:
                    full_content = current_content
            else:
                full_content = current_content
        else:
            # If file doesn't exist in GitHub, build from database content
            timestamp = datetime.utcnow().isoformat() + "Z"
            full_content = f"""---
id: {new_entry_id}
title: "{title}"
category: {new_category}
created: {timestamp}
content_hash: {content_hash}
source: mcp-claude
domain_tags: []
key_phrases: []
---

{content}"""

        # Build new file path
        filename = old_file_path.split("/")[-1]  # Preserve the date-slug filename
        new_folder = category_folders.get(new_category, new_category)
        new_file_path = f"{new_folder}/{filename}"

        # Create new file in GitHub
        create_file(
            repo=repo,
            path=new_file_path,
            content=full_content,
            message=f"Move note to {new_category}: {title}",
            token=token,
        )

        # Delete old file from GitHub
        try:
            delete_file(
                repo=repo,
                path=old_file_path,
                message=f"Move note from {old_category} to {new_category}: {title}",
                token=token,
            )
        except Exception as del_err:
            logger.warning(f"Failed to delete old file {old_file_path}: {del_err}")
            # Continue anyway - the new file was created

        # Update database
        db.execute(
            """
            UPDATE knowledge_entries
            SET entry_id = ?, category = ?, file_path = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (new_entry_id, new_category, new_file_path, entry_db_id),
        )
        commit_and_checkpoint(db)

        # Regenerate embedding (category change might affect semantic meaning)
        try:
            _generate_embedding_for_entry(entry_db_id, new_entry_id, content)
        except Exception as emb_err:
            logger.warning(f"Failed to regenerate embedding for {new_entry_id}: {emb_err}")

        logger.info(f"MCP moved note: {entry_id} -> {new_entry_id} ({old_category} -> {new_category})")

        return {
            "success": True,
            "old_entry_id": entry_id,
            "new_entry_id": new_entry_id,
            "old_category": old_category,
            "new_category": new_category,
            "old_file_path": old_file_path,
            "new_file_path": new_file_path,
            "title": title,
        }

    except Exception as e:
        logger.error(f"Failed to move note: {e}")
        return {"error": f"Failed to move note: {str(e)}"}


def tool_create_subfolder(args: dict) -> dict:
    """Create a subfolder under a category."""
    from .rag.database import get_user_categories
    from .rag.github_service import create_file

    category = args.get("category", "").lower().strip()
    subfolder_name = args.get("subfolder_name", "").strip()

    if not category:
        return {"error": "category is required"}
    if not subfolder_name:
        return {"error": "subfolder_name is required"}

    # Validate subfolder name (no slashes, reasonable characters)
    if "/" in subfolder_name or "\\" in subfolder_name:
        return {"error": "Subfolder name cannot contain slashes"}
    if not re.match(r"^[a-zA-Z0-9_-]+$", subfolder_name):
        return {"error": "Subfolder name can only contain letters, numbers, underscores, and hyphens"}

    db = get_db()
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None

    # Validate category
    categories = get_user_categories(db, user_id or "default")
    valid_categories = {c["name"] for c in categories}
    category_folders = {c["name"]: c["folder_name"] for c in categories}

    if category not in valid_categories:
        return {"error": f"Invalid category. Must be one of: {', '.join(sorted(valid_categories))}"}

    folder = category_folders.get(category, category)
    subfolder_path = f"{folder}/{subfolder_name}/.gitkeep"

    # Get user's installation token
    from .auth import get_user_installation_token
    from .core import get_user_library_repo

    token = get_user_installation_token(user_id, "library") if user_id else None
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    repo = get_user_library_repo(user_id)

    try:
        # Create .gitkeep file to establish the subfolder
        create_file(
            repo=repo,
            path=subfolder_path,
            content="",
            message=f"Create subfolder: {folder}/{subfolder_name}",
            token=token,
        )

        logger.info(f"MCP created subfolder: {folder}/{subfolder_name}")

        return {
            "success": True,
            "category": category,
            "subfolder": subfolder_name,
            "path": f"{folder}/{subfolder_name}",
        }

    except Exception as e:
        logger.error(f"Failed to create subfolder: {e}")
        return {"error": f"Failed to create subfolder: {str(e)}"}


def tool_list_subfolders(args: dict) -> dict:
    """List all subfolders under a category."""
    from .rag.database import get_user_categories
    from .rag.github_service import list_folder

    category = args.get("category", "").lower().strip()

    if not category:
        return {"error": "category is required"}

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None

    # Validate category
    categories = get_user_categories(db, user_id or "default")
    valid_categories = {c["name"] for c in categories}
    category_folders = {c["name"]: c["folder_name"] for c in categories}

    if category not in valid_categories:
        return {"error": f"Invalid category. Must be one of: {', '.join(sorted(valid_categories))}"}

    folder = category_folders.get(category, category)

    # Get user's installation token
    from .auth import get_user_installation_token
    from .core import get_user_library_repo

    token = get_user_installation_token(user_id, "library") if user_id else None
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    repo = get_user_library_repo(user_id)

    try:
        # List contents of the category folder
        items = list_folder(repo, folder, token)

        # Filter to only directories (subfolders)
        subfolders = [item["name"] for item in items if item.get("type") == "dir"]

        # Also get count of notes per subfolder from database
        subfolder_counts = {}
        rows = db.execute(
            """
            SELECT subfolder, COUNT(*) as count
            FROM knowledge_entries
            WHERE category = ? AND subfolder IS NOT NULL
            GROUP BY subfolder
            """,
            (category,),
        ).fetchall()
        for row in rows:
            subfolder_counts[row["subfolder"]] = row["count"]

        # Get root count (notes without subfolder)
        root_count = db.execute(
            """
            SELECT COUNT(*) as count
            FROM knowledge_entries
            WHERE category = ? AND (subfolder IS NULL OR subfolder = '')
            """,
            (category,),
        ).fetchone()["count"]

        return {
            "category": category,
            "folder": folder,
            "subfolders": [
                {"name": sf, "path": f"{folder}/{sf}", "note_count": subfolder_counts.get(sf, 0)} for sf in subfolders
            ],
            "root_note_count": root_count,
        }

    except Exception as e:
        logger.error(f"Failed to list subfolders: {e}")
        return {"error": f"Failed to list subfolders: {str(e)}"}


def tool_list_subfolder_contents(args: dict) -> dict:
    """List all notes within a specific subfolder of a category."""
    from .rag.database import get_user_categories

    category = args.get("category", "").lower().strip()
    subfolder = args.get("subfolder", "").strip() if args.get("subfolder") else None

    if not category:
        return {"error": "category is required"}

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None

    # Validate category
    categories = get_user_categories(db, user_id or "default")
    valid_categories = {c["name"] for c in categories}
    category_folders = {c["name"]: c["folder_name"] for c in categories}

    if category not in valid_categories:
        return {"error": f"Invalid category. Must be one of: {', '.join(sorted(valid_categories))}"}

    folder = category_folders.get(category, category)

    try:
        # Query notes in this category/subfolder combination
        if subfolder:
            rows = db.execute(
                """
                SELECT entry_id, title, created_at, updated_at, file_path
                FROM knowledge_entries
                WHERE category = ? AND subfolder = ?
                ORDER BY updated_at DESC
                """,
                (category, subfolder),
            ).fetchall()
            path_display = f"{folder}/{subfolder}"
        else:
            rows = db.execute(
                """
                SELECT entry_id, title, created_at, updated_at, file_path
                FROM knowledge_entries
                WHERE category = ? AND (subfolder IS NULL OR subfolder = '')
                ORDER BY updated_at DESC
                """,
                (category,),
            ).fetchall()
            path_display = f"{folder} (root)"

        notes = [
            {
                "entry_id": row["entry_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "file_path": row["file_path"],
            }
            for row in rows
        ]

        return {
            "category": category,
            "subfolder": subfolder,
            "path": path_display,
            "note_count": len(notes),
            "notes": notes,
        }

    except Exception as e:
        logger.error(f"Failed to list subfolder contents: {e}")
        return {"error": f"Failed to list subfolder contents: {str(e)}"}


def tool_move_to_subfolder(args: dict) -> dict:
    """Move a note to a different subfolder within its current category."""
    from .rag.github_service import (
        commit_file,
        create_file,
        delete_file,
        file_exists,
        get_file_content,
    )

    entry_id = args.get("entry_id", "").strip()
    new_subfolder = args.get("subfolder", "").strip() if args.get("subfolder") else None

    if not entry_id:
        return {"error": "entry_id is required"}

    # Validate subfolder name if provided
    if new_subfolder and ("/" in new_subfolder or "\\" in new_subfolder):
        return {"error": "Subfolder name cannot contain slashes"}

    db = get_db()
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None

    # Get existing note
    entry = db.execute(
        """
        SELECT id, entry_id, title, category, content, file_path, subfolder
        FROM knowledge_entries
        WHERE entry_id = ?
        """,
        (entry_id,),
    ).fetchone()

    if not entry:
        return {"error": f"Note not found: {entry_id}"}

    old_subfolder = entry["subfolder"]
    if old_subfolder == new_subfolder:
        return {"error": f"Note is already in subfolder '{new_subfolder or '(root)'}"}

    title = entry["title"]
    category = entry["category"]
    old_file_path = entry["file_path"]
    entry_db_id = entry["id"]

    # Get category folder
    from .rag.database import get_user_categories

    categories = get_user_categories(db, user_id or "default")
    category_folders = {c["name"]: c["folder_name"] for c in categories}
    folder = category_folders.get(category, category)

    # Build new file path
    filename = old_file_path.split("/")[-1]  # Preserve the filename
    if new_subfolder:
        new_file_path = f"{folder}/{new_subfolder}/{filename}"
    else:
        new_file_path = f"{folder}/{filename}"

    # Get user's installation token
    from .auth import get_user_installation_token
    from .core import get_user_library_repo

    token = get_user_installation_token(user_id, "library") if user_id else None
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    repo = get_user_library_repo(user_id)

    try:
        # Get current file content from GitHub
        current_content = get_file_content(repo, old_file_path, token)

        if current_content:
            # Update subfolder in frontmatter
            if current_content.startswith("---"):
                parts = current_content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter_lines = parts[1].strip().split("\n")
                    new_frontmatter_lines = []
                    has_subfolder = False
                    for line in frontmatter_lines:
                        if line.startswith("subfolder:"):
                            has_subfolder = True
                            if new_subfolder:
                                new_frontmatter_lines.append(f"subfolder: {new_subfolder}")
                            # Else skip the line (remove subfolder field)
                        else:
                            new_frontmatter_lines.append(line)
                    # Add subfolder if it wasn't in frontmatter
                    if new_subfolder and not has_subfolder:
                        new_frontmatter_lines.append(f"subfolder: {new_subfolder}")
                    full_content = f"---\n{chr(10).join(new_frontmatter_lines)}\n---{parts[2]}"
                else:
                    full_content = current_content
            else:
                full_content = current_content
        else:
            return {"error": f"File not found in GitHub: {old_file_path}"}

        # Create new file (or update if destination already exists - collision case)
        if file_exists(repo, new_file_path, token):
            logger.info(f"Destination {new_file_path} exists, updating instead of creating")
            commit_file(
                repo=repo,
                path=new_file_path,
                content=full_content,
                message=f"Move note to subfolder: {title}",
                token=token,
            )
        else:
            create_file(
                repo=repo,
                path=new_file_path,
                content=full_content,
                message=f"Move note to subfolder: {title}",
                token=token,
            )

        # Delete old file
        try:
            delete_file(
                repo=repo,
                path=old_file_path,
                message=(f"Move note from {old_subfolder or '(root)'} to {new_subfolder or '(root)'}"),
                token=token,
            )
        except Exception as del_err:
            logger.warning(f"Failed to delete old file {old_file_path}: {del_err}")

        # Update database
        db.execute(
            """
            UPDATE knowledge_entries
            SET file_path = ?, subfolder = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (new_file_path, new_subfolder, entry_db_id),
        )
        commit_and_checkpoint(db)

        logger.info(
            f"MCP moved note to subfolder: {entry_id} ({old_subfolder or '(root)'} -> {new_subfolder or '(root)'})"
        )

        return {
            "success": True,
            "entry_id": entry_id,
            "title": title,
            "category": category,
            "old_subfolder": old_subfolder,
            "new_subfolder": new_subfolder,
            "old_file_path": old_file_path,
            "new_file_path": new_file_path,
        }

    except Exception as e:
        logger.error(f"Failed to move note to subfolder: {e}")
        return {"error": f"Failed to move note to subfolder: {str(e)}"}


def tool_rename_note(args: dict) -> dict:
    """Rename a note's title and update all associated paths and references.

    This is a git-native operation that:
    1. Updates the title in frontmatter
    2. Generates a new entry_id based on the new title
    3. Moves the file to a new path with the new slug
    4. Updates all database references
    5. Regenerates embeddings for the new entry_id
    """
    from .rag.github_service import (
        commit_file,
        create_file,
        delete_file,
        file_exists,
        get_file_content,
    )

    entry_id = args.get("entry_id", "").strip()
    new_title = args.get("new_title", "").strip()

    if not entry_id:
        return {"error": "entry_id is required"}
    if not new_title:
        return {"error": "new_title is required"}

    db = get_db()
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None

    # Get existing note
    entry = db.execute(
        """
        SELECT id, entry_id, title, category, content, file_path, subfolder, content_hash
        FROM knowledge_entries
        WHERE entry_id = ?
        """,
        (entry_id,),
    ).fetchone()

    if not entry:
        return {"error": f"Note not found: {entry_id}"}

    old_title = entry["title"]
    if old_title == new_title:
        return {"error": f"Note already has title '{new_title}'"}

    category = entry["category"]
    old_file_path = entry["file_path"]
    subfolder = entry["subfolder"]
    content = entry["content"]
    content_hash = entry["content_hash"]
    entry_db_id = entry["id"]

    # Generate new slug and entry_id
    def generate_slug(title: str) -> str:
        """Generate a URL-friendly slug from title."""
        slug = title.lower().strip()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        slug = slug.strip("-")
        return slug[:80] if len(slug) > 80 else slug

    new_slug = generate_slug(new_title)
    if not new_slug:
        return {"error": "New title must contain at least some alphanumeric characters"}

    # Generate new entry_id
    new_entry_id = f"library.{category}.{new_slug}"

    # Check for collision
    existing = db.execute(
        "SELECT entry_id FROM knowledge_entries WHERE entry_id = ? AND id != ?",
        (new_entry_id, entry_db_id),
    ).fetchone()
    if existing:
        # Add hash suffix to disambiguate
        if content_hash:
            new_entry_id = f"library.{category}.{new_slug}.{content_hash[:12]}"
        else:
            import hashlib

            hash_suffix = hashlib.sha256(new_title.encode()).hexdigest()[:12]
            new_entry_id = f"library.{category}.{new_slug}.{hash_suffix}"

    # Get category folder
    from .rag.database import get_user_categories

    categories = get_user_categories(db, user_id or "default")
    category_folders = {c["name"]: c["folder_name"] for c in categories}
    folder = category_folders.get(category, category)

    # Build new file path - preserve the date prefix from old filename
    old_filename = old_file_path.split("/")[-1]
    # Extract date prefix (YYYY-MM-DD) if present
    date_match = re.match(r"^(\d{4}-\d{2}-\d{2})-", old_filename)
    if date_match:
        date_prefix = date_match.group(1)
        new_filename = f"{date_prefix}-{new_slug}.md"
    else:
        # No date prefix, just use new slug
        new_filename = f"{new_slug}.md"

    if subfolder:
        new_file_path = f"{folder}/{subfolder}/{new_filename}"
    else:
        new_file_path = f"{folder}/{new_filename}"

    # Get user's installation token
    from .auth import get_user_installation_token
    from .core import get_user_library_repo

    token = get_user_installation_token(user_id, "library") if user_id else None
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    repo = get_user_library_repo(user_id)

    try:
        # Get current file content from GitHub
        current_content = get_file_content(repo, old_file_path, token)
        sync_repaired = False

        if not current_content:
            # File doesn't exist in GitHub - sync mismatch detected
            # Auto-repair using database content
            logger.warning(
                f"Sync mismatch detected: {entry_id} exists in DB but not in"
                f" GitHub at {old_file_path}. Auto-repairing during rename."
            )
            sync_repaired = True
            # Build content from database
            timestamp = datetime.utcnow().isoformat() + "Z"
            content_hash_value = content_hash or compute_content_hash(content)
            current_content = f"""---
id: {entry_id}
title: "{old_title}"
category: {category}
created: {timestamp}
content_hash: {content_hash_value}
source: mcp-claude
domain_tags: []
key_phrases: []
---

{content}"""

        # Update frontmatter with new title and entry_id
        if current_content.startswith("---"):
            parts = current_content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_lines = parts[1].strip().split("\n")
                new_frontmatter_lines = []
                has_title = False
                has_id = False
                for line in frontmatter_lines:
                    if line.startswith("title:"):
                        has_title = True
                        # Handle quoted titles
                        new_frontmatter_lines.append(f'title: "{new_title}"')
                    elif line.startswith("id:"):
                        has_id = True
                        new_frontmatter_lines.append(f"id: {new_entry_id}")
                    else:
                        new_frontmatter_lines.append(line)
                # Add if not present
                if not has_title:
                    new_frontmatter_lines.insert(0, f'title: "{new_title}"')
                if not has_id:
                    new_frontmatter_lines.insert(0, f"id: {new_entry_id}")
                full_content = f"---\n{chr(10).join(new_frontmatter_lines)}\n---{parts[2]}"
            else:
                full_content = current_content
        else:
            full_content = current_content

        # Create new file (or update if destination already exists)
        commit_message = f"Rename note: {old_title} → {new_title}"
        if file_exists(repo, new_file_path, token):
            logger.info(f"Destination {new_file_path} exists, updating instead of creating")
            commit_file(
                repo=repo,
                path=new_file_path,
                content=full_content,
                message=commit_message,
                token=token,
            )
        else:
            create_file(
                repo=repo,
                path=new_file_path,
                content=full_content,
                message=commit_message,
                token=token,
            )

        # Delete old file (only if path changed and file existed in GitHub)
        if old_file_path != new_file_path and not sync_repaired:
            try:
                delete_file(
                    repo=repo,
                    path=old_file_path,
                    message=f"Remove old file after rename: {old_title}",
                    token=token,
                )
            except Exception as del_err:
                logger.warning(f"Failed to delete old file {old_file_path}: {del_err}")

        # Update database
        db.execute(
            """
            UPDATE knowledge_entries
            SET entry_id = ?, title = ?, file_path = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (new_entry_id, new_title, new_file_path, entry_db_id),
        )

        # Update note_links table to update references
        db.execute(
            """
            UPDATE note_links
            SET source_entry_id = ?
            WHERE source_entry_id = ?
            """,
            (new_entry_id, entry_id),
        )
        db.execute(
            """
            UPDATE note_links
            SET target_entry_id = ?
            WHERE target_entry_id = ?
            """,
            (new_entry_id, entry_id),
        )

        commit_and_checkpoint(db)

        # Regenerate embedding for new entry_id
        try:
            service = get_embedding_service()
            if service:
                # Delete old embedding
                db.execute("DELETE FROM embeddings WHERE entry_id = ?", (entry_db_id,))
                # Generate new embedding
                service.generate_and_store(new_entry_id, "knowledge", content, db_id=entry_db_id)
                commit_and_checkpoint(db)
        except Exception as emb_err:
            logger.warning(f"Failed to regenerate embedding: {emb_err}")

        logger.info(
            f"MCP renamed note: {entry_id} → {new_entry_id} ('{old_title}' → '{new_title}')"
            + (" [sync repaired]" if sync_repaired else "")
        )

        response = {
            "success": True,
            "old_entry_id": entry_id,
            "new_entry_id": new_entry_id,
            "old_title": old_title,
            "new_title": new_title,
            "old_file_path": old_file_path,
            "new_file_path": new_file_path,
        }

        if sync_repaired:
            response["sync_repaired"] = True
            response["message"] = (
                "File was missing in GitHub and has been recreated from database contentduring rename."
            )

        return response

    except Exception as e:
        logger.error(f"Failed to rename note: {e}")
        return {"error": f"Failed to rename note: {str(e)}"}


def tool_rename_subfolder(args: dict) -> dict:
    """Rename a subfolder within a category.

    This is a git-native operation that:
    1. Moves all notes in the old subfolder to the new subfolder path
    2. Updates the subfolder field in frontmatter for each note
    3. Updates database records for all affected notes
    4. Removes the old .gitkeep and creates a new one
    """
    from .rag.database import get_user_categories
    from .rag.github_service import (
        create_file,
        delete_file,
        file_exists,
        get_file_content,
        list_folder,
    )

    category = args.get("category", "").lower().strip()
    old_name = args.get("old_name", "").strip()
    new_name = args.get("new_name", "").strip()

    if not category:
        return {"error": "category is required"}
    if not old_name:
        return {"error": "old_name is required"}
    if not new_name:
        return {"error": "new_name is required"}
    if old_name == new_name:
        return {"error": "old_name and new_name are the same"}

    # Validate subfolder names
    if "/" in old_name or "\\" in old_name or "/" in new_name or "\\" in new_name:
        return {"error": "Subfolder names cannot contain slashes"}
    if not re.match(r"^[a-zA-Z0-9_-]+$", new_name):
        return {"error": ("New subfolder name can only contain letters, numbers, underscores, and hyphens")}

    db = get_db()
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None

    # Validate category
    categories = get_user_categories(db, user_id or "default")
    valid_categories = {c["name"] for c in categories}
    category_folders = {c["name"]: c["folder_name"] for c in categories}

    if category not in valid_categories:
        return {"error": f"Invalid category. Must be one of: {', '.join(sorted(valid_categories))}"}

    folder = category_folders.get(category, category)
    old_subfolder_path = f"{folder}/{old_name}"
    new_subfolder_path = f"{folder}/{new_name}"

    # Get user's installation token
    from .auth import get_user_installation_token
    from .core import get_user_library_repo

    token = get_user_installation_token(user_id, "library") if user_id else None
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    repo = get_user_library_repo(user_id)

    try:
        # Check if old subfolder exists
        old_items = list_folder(repo, old_subfolder_path, token)
        if not old_items:
            return {"error": f"Subfolder not found: {old_subfolder_path}"}

        # Check if new subfolder already exists
        new_items = list_folder(repo, new_subfolder_path, token)
        if new_items:
            return {"error": f"Destination subfolder already exists: {new_subfolder_path}"}

        # Get all notes in the old subfolder from database
        notes = db.execute(
            """
            SELECT id, entry_id, title, file_path
            FROM knowledge_entries
            WHERE category = ? AND subfolder = ?
            """,
            (category, old_name),
        ).fetchall()

        moved_count = 0
        errors = []

        # Move each note to the new subfolder
        for note in notes:
            note_id = note["id"]
            note_entry_id = note["entry_id"]
            note_title = note["title"]
            old_file_path = note["file_path"]

            # Build new file path
            filename = old_file_path.split("/")[-1]
            new_file_path = f"{new_subfolder_path}/{filename}"

            try:
                # Get current file content
                current_content = get_file_content(repo, old_file_path, token)
                if not current_content:
                    errors.append(f"File not found: {old_file_path}")
                    continue

                # Update subfolder in frontmatter
                if current_content.startswith("---"):
                    parts = current_content.split("---", 2)
                    if len(parts) >= 3:
                        frontmatter_lines = parts[1].strip().split("\n")
                        new_frontmatter_lines = []
                        has_subfolder = False
                        for line in frontmatter_lines:
                            if line.startswith("subfolder:"):
                                has_subfolder = True
                                new_frontmatter_lines.append(f"subfolder: {new_name}")
                            else:
                                new_frontmatter_lines.append(line)
                        if not has_subfolder:
                            new_frontmatter_lines.append(f"subfolder: {new_name}")
                        full_content = f"---\n{chr(10).join(new_frontmatter_lines)}\n---{parts[2]}"
                    else:
                        full_content = current_content
                else:
                    full_content = current_content

                # Create new file
                create_file(
                    repo=repo,
                    path=new_file_path,
                    content=full_content,
                    message=f"Rename subfolder: move {note_title}",
                    token=token,
                )

                # Delete old file
                try:
                    delete_file(
                        repo=repo,
                        path=old_file_path,
                        message=f"Rename subfolder: remove old {note_title}",
                        token=token,
                    )
                except Exception as del_err:
                    logger.warning(f"Failed to delete old file {old_file_path}: {del_err}")

                # Update database
                db.execute(
                    """
                    UPDATE knowledge_entries
                    SET file_path = ?, subfolder = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (new_file_path, new_name, note_id),
                )
                moved_count += 1

            except Exception as note_err:
                errors.append(f"Failed to move {note_title}: {str(note_err)}")
                logger.error(f"Failed to move note {note_entry_id}: {note_err}")

        # Create .gitkeep in new subfolder (if not already there from moves)
        new_gitkeep_path = f"{new_subfolder_path}/.gitkeep"
        if not file_exists(repo, new_gitkeep_path, token):
            try:
                create_file(
                    repo=repo,
                    path=new_gitkeep_path,
                    content="",
                    message=f"Create subfolder: {new_subfolder_path}",
                    token=token,
                )
            except Exception as gitkeep_err:
                logger.warning(f"Failed to create .gitkeep in new subfolder: {gitkeep_err}")

        # Delete old .gitkeep if exists and folder is now empty
        old_gitkeep_path = f"{old_subfolder_path}/.gitkeep"
        try:
            # Check if old folder is empty (only .gitkeep remains)
            remaining_items = list_folder(repo, old_subfolder_path, token)
            remaining_files = [item for item in remaining_items if item["name"] != ".gitkeep"]
            if not remaining_files and file_exists(repo, old_gitkeep_path, token):
                delete_file(
                    repo=repo,
                    path=old_gitkeep_path,
                    message=f"Remove empty subfolder: {old_subfolder_path}",
                    token=token,
                )
        except Exception as cleanup_err:
            logger.warning(f"Failed to cleanup old subfolder: {cleanup_err}")

        commit_and_checkpoint(db)

        logger.info(f"MCP renamed subfolder: {old_subfolder_path} → {new_subfolder_path} ({moved_count} notes)")

        result = {
            "success": True,
            "category": category,
            "old_name": old_name,
            "new_name": new_name,
            "old_path": old_subfolder_path,
            "new_path": new_subfolder_path,
            "notes_moved": moved_count,
        }
        if errors:
            result["warnings"] = errors

        return result

    except Exception as e:
        logger.error(f"Failed to rename subfolder: {e}")
        return {"error": f"Failed to rename subfolder: {str(e)}"}


def tool_delete_note(args: dict) -> dict:
    """Delete a note from both GitHub and local database."""
    from .rag.github_service import delete_file

    entry_id = args.get("entry_id", "").strip()
    confirm = args.get("confirm", False)

    if not entry_id:
        return {"error": "entry_id is required"}

    if not confirm:
        return {
            "error": "Deletion requires confirmation. Set confirm=true to proceed.",
            "warning": ("This will permanently delete the note from both GitHub and the local database."),
        }

    db = get_db()

    # Get existing note
    entry = db.execute(
        """
        SELECT entry_id, title, file_path
        FROM knowledge_entries
        WHERE entry_id = ?
        """,
        (entry_id,),
    ).fetchone()

    if not entry:
        return {"error": f"Note not found: {entry_id}"}

    # Get user's installation token
    from .auth import get_user_installation_token
    from .core import get_user_library_repo

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None
    token = get_user_installation_token(user_id, "library") if user_id else None
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    repo = get_user_library_repo(user_id)
    file_path = entry["file_path"]
    title = entry["title"]

    try:
        # Delete from GitHub
        if file_path:
            try:
                delete_file(repo=repo, path=file_path, message=f"Delete note via MCP: {title}", token=token)
            except Exception as e:
                # File might not exist in GitHub, continue with local deletion
                logger.warning(f"Could not delete from GitHub (may not exist): {e}")

        # Delete from local database
        db.execute("DELETE FROM knowledge_entries WHERE entry_id = ?", (entry_id,))

        # Also delete any links involving this note
        db.execute(
            "DELETE FROM note_links WHERE source_entry_id = ? OR target_entry_id = ?",
            (entry_id, entry_id),
        )

        # Delete embeddings (using the integer id we fetched earlier)
        db.execute("DELETE FROM embeddings WHERE entry_id = ?", (entry["id"],))

        commit_and_checkpoint(db)

        logger.info(f"MCP deleted note: {entry_id} - {title}")

        return {
            "success": True,
            "deleted": {"entry_id": entry_id, "title": title, "file_path": file_path},
        }

    except Exception as e:
        logger.error(f"Failed to delete note: {e}")
        return {"error": f"Failed to delete note: {str(e)}"}


def tool_list_tasks(args: dict) -> dict:
    """List notes marked as tasks with optional filtering."""
    status = args.get("status")
    due_before = args.get("due_before")
    due_after = args.get("due_after")
    category = args.get("category")
    limit = min(args.get("limit", 50), 100)

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    # Build query dynamically
    sql = """
        SELECT entry_id, title, category, task_status, due_date, created_at, updated_at
        FROM knowledge_entries
        WHERE task_status IS NOT NULL
    """
    params = []

    if status:
        sql += " AND task_status = ?"
        params.append(status)

    if due_before:
        sql += " AND due_date <= ?"
        params.append(due_before)

    if due_after:
        sql += " AND due_date >= ?"
        params.append(due_after)

    if category:
        sql += " AND category = ?"
        params.append(category)

    sql += (
        " ORDER BY CASE task_status"
        " WHEN 'blocked' THEN 0 WHEN 'in_progress' THEN 1"
        " WHEN 'pending' THEN 2 ELSE 3 END,"
        " due_date ASC NULLS LAST, updated_at DESC LIMIT ?"
    )
    params.append(limit)

    tasks = db.execute(sql, params).fetchall()

    # Get counts by status
    status_counts = db.execute("""
        SELECT task_status, COUNT(*) as count
        FROM knowledge_entries
        WHERE task_status IS NOT NULL
        GROUP BY task_status
    """).fetchall()

    return {
        "tasks": [
            {
                "entry_id": t["entry_id"],
                "title": t["title"],
                "category": t["category"],
                "status": t["task_status"],
                "due_date": t["due_date"],
                "created_at": t["created_at"],
                "updated_at": t["updated_at"],
            }
            for t in tasks
        ],
        "count": len(tasks),
        "status_counts": {r["task_status"]: r["count"] for r in status_counts},
    }


def tool_update_task_status(args: dict) -> dict:
    """Update task status for a note."""
    entry_id = args.get("entry_id", "").strip()
    status = args.get("status", "").strip()
    due_date = args.get("due_date", "").strip() if args.get("due_date") else None

    if not entry_id:
        return {"error": "entry_id is required"}

    if not status:
        return {"error": "status is required"}

    valid_statuses = {"pending", "in_progress", "done", "blocked"}
    if status not in valid_statuses:
        return {"error": f"Invalid status. Must be one of: {', '.join(sorted(valid_statuses))}"}

    db = get_db()

    # Check note exists
    entry = db.execute(
        "SELECT entry_id, title, task_status FROM knowledge_entries WHERE entry_id = ?", (entry_id,)
    ).fetchone()

    if not entry:
        return {"error": f"Note not found: {entry_id}"}

    old_status = entry["task_status"]

    # Update task status and optionally due_date
    if due_date:
        db.execute(
            """
            UPDATE knowledge_entries
            SET task_status = ?, due_date = ?, updated_at = CURRENT_TIMESTAMP
            WHERE entry_id = ?
            """,
            (status, due_date, entry_id),
        )
    else:
        db.execute(
            """
            UPDATE knowledge_entries
            SET task_status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE entry_id = ?
            """,
            (status, entry_id),
        )
    commit_and_checkpoint(db)

    logger.info(f"MCP updated task status: {entry_id} {old_status} -> {status}")

    return {
        "success": True,
        "entry_id": entry_id,
        "title": entry["title"],
        "old_status": old_status,
        "new_status": status,
        "due_date": due_date,
    }


def tool_link_notes(args: dict) -> dict:
    """Create an explicit relationship between two notes."""
    source_id = args.get("source_id", "").strip()
    target_id = args.get("target_id", "").strip()
    link_type = args.get("link_type", "related").strip()
    description = args.get("description", "").strip() if args.get("description") else None

    if not source_id or not target_id:
        return {"error": "Both source_id and target_id are required"}

    if source_id == target_id:
        return {"error": "Cannot link a note to itself"}

    valid_link_types = {
        "related",
        "depends_on",
        "blocks",
        "implements",
        "references",
        "contradicts",
        "supports",
    }
    if link_type not in valid_link_types:
        return {"error": f"Invalid link_type. Must be one of: {', '.join(sorted(valid_link_types))}"}

    db = get_db()

    # Verify both notes exist
    source = db.execute("SELECT entry_id, title FROM knowledge_entries WHERE entry_id = ?", (source_id,)).fetchone()
    target = db.execute("SELECT entry_id, title FROM knowledge_entries WHERE entry_id = ?", (target_id,)).fetchone()

    if not source:
        return {"error": f"Source note not found: {source_id}"}
    if not target:
        return {"error": f"Target note not found: {target_id}"}

    try:
        # Insert the link (ignore if already exists)
        db.execute(
            """
            INSERT OR IGNORE INTO note_links
            (source_entry_id, target_entry_id, link_type, description, created_by)
            VALUES (?, ?, ?, ?, 'mcp-claude')
            """,
            (source_id, target_id, link_type, description),
        )

        # For bidirectional discovery, also create reverse link for symmetric types
        symmetric_types = {"related", "contradicts"}
        if link_type in symmetric_types:
            db.execute(
                """
                INSERT OR IGNORE INTO note_links
                (source_entry_id, target_entry_id, link_type, description, created_by)
                VALUES (?, ?, ?, ?, 'mcp-claude')
                """,
                (target_id, source_id, link_type, description),
            )

        commit_and_checkpoint(db)

        logger.info(f"MCP linked notes: {source_id} --[{link_type}]--> {target_id}")

        return {
            "success": True,
            "link": {
                "source": {"entry_id": source_id, "title": source["title"]},
                "target": {"entry_id": target_id, "title": target["title"]},
                "type": link_type,
                "description": description,
            },
        }

    except Exception as e:
        logger.error(f"Failed to link notes: {e}")
        return {"error": f"Failed to create link: {str(e)}"}


def tool_get_note_context(args: dict) -> dict:
    """Get a note with its full context: linked notes and semantic neighbors."""
    entry_id = args.get("entry_id", "").strip()
    include_semantic = args.get("include_semantic", True)
    semantic_limit = min(args.get("semantic_limit", 5), 20)

    if not entry_id:
        return {"error": "entry_id is required"}

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    # Get the main note
    entry = db.execute(
        """
        SELECT entry_id, title, category, content, file_path, task_status, due_date,
               created_at, updated_at, chord_status, chord_repo
        FROM knowledge_entries
        WHERE entry_id = ?
        """,
        (entry_id,),
    ).fetchone()

    if not entry:
        return {"error": f"Note not found: {entry_id}"}

    # Get outgoing links (this note links to others)
    outgoing = db.execute(
        """
        SELECT nl.target_entry_id, nl.link_type, nl.description,
               ke.title, ke.category
        FROM note_links nl
        JOIN knowledge_entries ke ON ke.entry_id = nl.target_entry_id
        WHERE nl.source_entry_id = ?
        """,
        (entry_id,),
    ).fetchall()

    # Get incoming links (others link to this note)
    incoming = db.execute(
        """
        SELECT nl.source_entry_id, nl.link_type, nl.description,
               ke.title, ke.category
        FROM note_links nl
        JOIN knowledge_entries ke ON ke.entry_id = nl.source_entry_id
        WHERE nl.target_entry_id = ?
        """,
        (entry_id,),
    ).fetchall()

    # Get semantic neighbors if requested
    semantic_neighbors = []
    if include_semantic:
        service = get_embedding_service()
        if service:
            try:
                # Search for similar notes
                search_result = service.hybrid_search(
                    query=entry["title"] + " " + (entry["content"][:500] if entry["content"] else ""),
                    entry_type="knowledge",
                    limit=semantic_limit + 1,  # +1 to exclude self
                    include_low_confidence=False,
                )
                for r in search_result.get("results", []):
                    if r["entry_id"] != entry_id:
                        semantic_neighbors.append(
                            {
                                "entry_id": r["entry_id"],
                                "title": r["title"],
                                "category": r.get("category"),
                                "similarity": round(r.get("similarity", 0), 3),
                            }
                        )
                        if len(semantic_neighbors) >= semantic_limit:
                            break
            except Exception as e:
                logger.warning(f"Could not get semantic neighbors: {e}")

    # Get related projects from agent queue (filtered by user for multi-tenant)
    from .rag.database import get_connection, get_db_path

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None
    try:
        agents_db = get_connection(get_db_path("agents.db"))
        projects = agents_db.execute(
            """
            SELECT queue_id, project_name, project_type, status, title
            FROM agent_queue
            WHERE related_entry_id LIKE ? AND user_id = ?
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (f"%{entry_id}%", user_id),
        ).fetchall()
    except Exception:
        projects = []

    return {
        "note": {
            "entry_id": entry["entry_id"],
            "title": entry["title"],
            "category": entry["category"],
            "content": entry["content"],
            "file_path": entry["file_path"],
            "task_status": entry["task_status"],
            "due_date": entry["due_date"],
            "created_at": entry["created_at"],
            "updated_at": entry["updated_at"],
            "chord_status": entry["chord_status"],
            "chord_repo": entry["chord_repo"],
        },
        "links": {
            "outgoing": [
                {
                    "entry_id": lnk["target_entry_id"],
                    "title": lnk["title"],
                    "category": lnk["category"],
                    "link_type": lnk["link_type"],
                    "description": lnk["description"],
                }
                for lnk in outgoing
            ],
            "incoming": [
                {
                    "entry_id": lnk["source_entry_id"],
                    "title": lnk["title"],
                    "category": lnk["category"],
                    "link_type": lnk["link_type"],
                    "description": lnk["description"],
                }
                for lnk in incoming
            ],
        },
        "semantic_neighbors": semantic_neighbors,
        "related_projects": [
            {
                "queue_id": p["queue_id"],
                "project_name": p["project_name"],
                "project_type": p["project_type"],
                "status": p["status"],
                "title": p["title"],
            }
            for p in projects
        ],
    }


def tool_process_motif(args: dict) -> dict:
    """Push content into the transcript processing pipeline.

    Uses the Pit-native MotifProcessor which:
    - Parses the transcript into threads
    - Classifies each thread using Claude
    - Correlates with existing entries
    - Extracts markdown artifacts
    - Writes to the user's Library
    """
    from flask import g

    from .motif_processor import process_motif_sync

    content = args.get("content", "").strip()
    source_label = args.get("source_label", "mcp-direct")

    if not content:
        return {"error": "content is required"}

    if len(content) < 10:
        return {"error": "Content too short. Minimum 10 characters required."}

    # Get user_id from MCP context
    if not hasattr(g, "mcp_user") or not g.mcp_user:
        return {"error": "Authentication required"}

    user_id = g.mcp_user.get("user_id")
    if not user_id:
        return {"error": "User ID not found in token"}

    try:
        # Use the new Pit-native motif processor
        # This processes synchronously using the user's own Anthropic API key
        result = process_motif_sync(content, user_id, source_label)

        if result.get("status") == "completed":
            return {
                "success": True,
                "job_id": result.get("job_id"),
                "status": "completed",
                "result": {
                    "entry_ids": result.get("entry_ids", []),
                    "notes_created": len(result.get("entry_ids", [])),
                },
            }
        elif result.get("status") == "failed":
            return {
                "success": False,
                "job_id": result.get("job_id"),
                "status": "failed",
                "error": result.get("error", "Processing failed"),
            }
        else:
            # Pending/processing - should not happen in sync mode
            return {
                "success": True,
                "job_id": result.get("job_id"),
                "status": result.get("status", "pending"),
                "message": "Processing in progress",
            }

    except Exception as e:
        logger.error(f"Failed to process motif: {e}")
        return {"error": f"Failed to process motif: {str(e)}"}


def tool_get_processing_status(args: dict) -> dict:
    """Check the status of an async processing job."""
    job_id = args.get("job_id", "").strip()

    if not job_id:
        return {"error": "job_id is required"}

    db = get_db()

    job = db.execute(
        """
        SELECT job_id, job_type, status, input_format, result_entry_ids, error_message,
               created_at, updated_at, completed_at
        FROM processing_jobs
        WHERE job_id = ?
        """,
        (job_id,),
    ).fetchone()

    if not job:
        return {"error": f"Job not found: {job_id}"}

    result = {
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }

    if job["status"] == "completed":
        result["completed_at"] = job["completed_at"]
        result["result_entry_ids"] = job["result_entry_ids"].split(",") if job["result_entry_ids"] else []

    if job["status"] == "failed":
        result["error"] = job["error_message"]

    return result


def tool_check_connection(args: dict) -> dict:
    """Diagnostic tool to check MCP connection and user state."""
    from .auth import _get_db as get_auth_db
    from .auth import get_user_installation_token

    result = {"mcp_auth": {}, "github_app": {}, "database": {}, "recommendations": []}

    # Check MCP authentication
    if hasattr(g, "mcp_user") and g.mcp_user:
        result["mcp_auth"]["authenticated"] = True
        result["mcp_auth"]["user_id"] = g.mcp_user.get("user_id")
        result["mcp_auth"]["github_login"] = g.mcp_user.get("sub")
        result["mcp_auth"]["github_id"] = g.mcp_user.get("github_id")

        # Show if canonical user_id lookup was performed
        # Note: The middleware already resolved canonical user_id before this runs
        auth_db = get_auth_db()
        github_id = g.mcp_user.get("github_id")
        if github_id:
            canonical = auth_db.execute("SELECT user_id FROM users WHERE github_id = ?", (github_id,)).fetchone()
            if canonical:
                result["mcp_auth"]["canonical_user_id"] = canonical["user_id"]
                if canonical["user_id"] != g.mcp_user.get("user_id"):
                    result["mcp_auth"]["user_id_corrected"] = True
    else:
        result["mcp_auth"]["authenticated"] = False
        result["recommendations"].append("MCP authentication failed - re-authenticate the MCP client")
        return result

    user_id = g.mcp_user.get("user_id")

    # Check user_repos table for library configuration
    auth_db = get_auth_db()
    user_repo = auth_db.execute(
        """
        SELECT repo_full_name, installation_id, created_at
        FROM user_repos
        WHERE user_id = ? AND repo_type = 'library'
        """,
        (user_id,),
    ).fetchone()

    if user_repo:
        result["github_app"]["library_configured"] = True
        result["github_app"]["library_repo"] = user_repo["repo_full_name"]
        result["github_app"]["installation_id"] = user_repo["installation_id"]

        # Try to get installation token
        token = get_user_installation_token(user_id, "library")
        if token:
            result["github_app"]["token_valid"] = True
        else:
            result["github_app"]["token_valid"] = False
            result["recommendations"].append(
                "GitHub App token is invalid - the installation may have been removed."
                "Re-install the GitHub App via the web interface."
            )
    else:
        result["github_app"]["library_configured"] = False
        result["recommendations"].append(
            f"No library repo configured for user {user_id}."
            " Complete GitHub App setup via the Legate Studio web interface."
        )

        # Check if there's a user record at all
        user_record = auth_db.execute(
            "SELECT github_id, github_login FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()

        if user_record:
            result["database"]["user_exists"] = True
            result["database"]["db_github_login"] = user_record["github_login"]
        else:
            result["database"]["user_exists"] = False
            result["recommendations"].append("User record not found in database - this is unexpected")

    # Check for Anthropic API key (needed for process_motif)
    from .auth import get_user_api_key

    try:
        api_key = get_user_api_key(user_id, "anthropic")
        if api_key:
            result["database"]["anthropic_api_key_set"] = True
        else:
            result["database"]["anthropic_api_key_set"] = False
            result["recommendations"].append(
                "Anthropic API key not configured. Add it in Legate Studio Settings to enableprocess_motif."
            )
    except Exception as e:
        result["database"]["anthropic_api_key_set"] = False
        result["database"]["api_key_error"] = str(e)

    # Count notes in library
    try:
        user_db = get_db()  # Gets user's legato database
        note_count = user_db.execute("SELECT COUNT(*) as count FROM knowledge_entries").fetchone()
        result["database"]["note_count"] = note_count["count"] if note_count else 0
    except Exception as e:
        result["database"]["note_count"] = "error"
        result["database"]["note_count_error"] = str(e)

    return result


# ============ Sync State Tools ============


def tool_verify_sync_state(args: dict) -> dict:
    """Check consistency between database entries and GitHub files.

    Identifies notes that exist in the database but are missing from GitHub
    (orphaned DB entries) which can cause update/rename operations to fail.

    When library_id is provided, checks the shared library database instead of
    the caller's personal library. The caller must be the owner.
    """
    from .rag.github_service import file_exists

    category = args.get("category", "").strip().lower() if args.get("category") else None
    limit = min(int(args.get("limit", 100)), 500)
    library_id = args.get("library_id", "").strip() if args.get("library_id") else None

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None

    from .auth import get_user_installation_token

    if library_id:
        # Shared library mode — use the shared library's DB and owner's token
        from .rag.database import get_shared_library_db, init_db as init_shared_db

        shared_meta = init_shared_db()
        row = shared_meta.execute(
            "SELECT owner_user_id, repo_full_name FROM shared_libraries WHERE id = ? AND status = 'active'",
            (library_id,),
        ).fetchone()
        if not row:
            return {"error": f"Shared library '{library_id}' not found"}
        if row["owner_user_id"] != user_id:
            return {"error": "Only the library owner can verify sync state for a shared library"}
        repo = row["repo_full_name"]
        if not repo:
            return {"error": "Shared library has no GitHub repository configured"}
        token = get_user_installation_token(row["owner_user_id"], "library")
        db = get_shared_library_db(library_id)
    else:
        # Personal library mode (original behaviour)
        from .core import get_user_library_repo

        db = get_db()
        token = get_user_installation_token(user_id, "library") if user_id else None
        repo = get_user_library_repo(user_id)

    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    if not repo:
        return {"error": "No GitHub repository configured for this library."}

    # Get entries from database
    if category:
        entries = db.execute(
            """
            SELECT entry_id, title, category, file_path, content_hash
            FROM knowledge_entries
            WHERE category = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (category, limit),
        ).fetchall()
    else:
        entries = db.execute(
            """
            SELECT entry_id, title, category, file_path, content_hash
            FROM knowledge_entries
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    # Check each entry against GitHub
    orphaned_db_entries = []
    healthy_entries = []
    errors = []

    for entry in entries:
        entry_id = entry["entry_id"]
        file_path = entry["file_path"]

        try:
            exists = file_exists(repo, file_path, token)
            if exists:
                healthy_entries.append({"entry_id": entry_id, "title": entry["title"], "file_path": file_path})
            else:
                orphaned_db_entries.append(
                    {
                        "entry_id": entry_id,
                        "title": entry["title"],
                        "category": entry["category"],
                        "file_path": file_path,
                        "issue": ("File missing in GitHub - database entry exists but GitHub file does not"),
                    }
                )
        except Exception as e:
            errors.append({"entry_id": entry_id, "file_path": file_path, "error": str(e)})

    result = {
        "total_checked": len(entries),
        "healthy_count": len(healthy_entries),
        "orphaned_db_count": len(orphaned_db_entries),
        "error_count": len(errors),
        "sync_status": "healthy" if len(orphaned_db_entries) == 0 else "mismatch_detected",
    }

    if orphaned_db_entries:
        result["orphaned_db_entries"] = orphaned_db_entries
        result["recommendation"] = "Use repair_sync_state to recreate missing GitHub files from database content."

    if errors:
        result["errors"] = errors

    return result


def tool_repair_sync_state(args: dict) -> dict:
    """Repair sync mismatches by recreating missing GitHub files from database content.

    For entries that exist in the database but are missing from GitHub,
    this recreates the GitHub file using the content stored in the database.
    """
    from .rag.github_service import create_file, file_exists

    entry_ids = args.get("entry_ids", [])
    limit = min(int(args.get("limit", 10)), 50)
    dry_run = args.get("dry_run", False)

    db = get_db()
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None

    # Get user's installation token
    from .auth import get_user_installation_token
    from .core import get_user_library_repo

    token = get_user_installation_token(user_id, "library") if user_id else None
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    repo = get_user_library_repo(user_id)

    # Get entries to repair
    if entry_ids:
        # Repair specific entries
        placeholders = ",".join("?" * len(entry_ids))
        entries = db.execute(
            f"""
            SELECT id, entry_id, title, category, content, file_path,
            subfolder, content_hash, task_status, due_date
            FROM knowledge_entries
            WHERE entry_id IN ({placeholders})
            """,
            entry_ids,
        ).fetchall()
    else:
        # Find orphaned entries (exist in DB but not in GitHub)
        all_entries = db.execute(
            """
            SELECT id, entry_id, title, category, content, file_path,
            subfolder, content_hash, task_status, due_date
            FROM knowledge_entries
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit * 5,),  # Check more than limit to find orphaned ones
        ).fetchall()

        # Filter to only orphaned entries
        entries = []
        for entry in all_entries:
            if len(entries) >= limit:
                break
            try:
                if not file_exists(repo, entry["file_path"], token):
                    entries.append(entry)
            except Exception:
                # If we can't check, skip it
                pass

    repaired = []
    skipped = []
    errors = []

    for entry in entries:
        entry_id = entry["entry_id"]
        file_path = entry["file_path"]
        title = entry["title"]
        category = entry["category"]
        content = entry["content"]
        content_hash = entry["content_hash"] or compute_content_hash(content)
        subfolder = entry["subfolder"]
        task_status = entry["task_status"]
        due_date = entry["due_date"]

        # Check if file actually exists in GitHub
        try:
            exists = file_exists(repo, file_path, token)
            if exists:
                skipped.append(
                    {
                        "entry_id": entry_id,
                        "file_path": file_path,
                        "reason": "File already exists in GitHub - no repair needed",
                    }
                )
                continue
        except Exception as e:
            errors.append(
                {
                    "entry_id": entry_id,
                    "file_path": file_path,
                    "error": f"Failed to check file existence: {str(e)}",
                }
            )
            continue

        if dry_run:
            repaired.append(
                {
                    "entry_id": entry_id,
                    "title": title,
                    "file_path": file_path,
                    "action": "would_create",
                    "dry_run": True,
                }
            )
            continue

        # Build the file content with frontmatter
        timestamp = datetime.utcnow().isoformat() + "Z"
        frontmatter_lines = [
            "---",
            f"id: {entry_id}",
            f'title: "{title}"',
            f"category: {category}",
            f"created: {timestamp}",
            f"content_hash: {content_hash}",
            "source: mcp-repair",
            "domain_tags: []",
            "key_phrases: []",
        ]

        if subfolder:
            frontmatter_lines.append(f"subfolder: {subfolder}")
        if task_status:
            frontmatter_lines.append(f"task_status: {task_status}")
        if due_date:
            frontmatter_lines.append(f"due_date: {due_date}")

        frontmatter_lines.append("---")
        frontmatter_lines.append("")
        frontmatter = "\n".join(frontmatter_lines)
        full_content = frontmatter + content

        # Create the file in GitHub
        try:
            create_file(
                repo=repo,
                path=file_path,
                content=full_content,
                message=f"Repair sync: recreate note {title}",
                token=token,
            )
            repaired.append({"entry_id": entry_id, "title": title, "file_path": file_path, "action": "created"})
            logger.info(f"Repaired sync for {entry_id}: created {file_path}")
        except Exception as e:
            errors.append(
                {
                    "entry_id": entry_id,
                    "file_path": file_path,
                    "error": f"Failed to create file: {str(e)}",
                }
            )

    result = {
        "dry_run": dry_run,
        "repaired_count": len(repaired),
        "skipped_count": len(skipped),
        "error_count": len(errors),
    }

    if repaired:
        result["repaired"] = repaired
    if skipped:
        result["skipped"] = skipped
    if errors:
        result["errors"] = errors

    if dry_run and repaired:
        result["message"] = (
            f"Dry run complete. {len(repaired)} entries would be repaired. Run with dry_run=false to apply changes."
        )
    elif repaired:
        result["message"] = f"Successfully repaired {len(repaired)} entries by creating missing GitHub files."

    return result


def tool_sync_shared_library(args: dict) -> dict:
    """Sync a shared library's database from its GitHub repository.

    Only callable by the library owner. Pulls the latest content from the
    'main' branch and updates the per-library SQLite database.
    """
    from .rag.database import init_db
    from .rag.library_sync import sync_shared_library

    library_id = args.get("library_id", "").strip()
    if not library_id:
        return {"error": "library_id is required"}

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None
    if not user_id:
        return {"error": "Authentication required"}

    # Verify caller is the library owner
    shared_db = init_db()
    row = shared_db.execute(
        "SELECT owner_user_id FROM shared_libraries WHERE id = ? AND status = 'active'",
        (library_id,),
    ).fetchone()

    if not row:
        return {"error": f"Shared library '{library_id}' not found"}

    if row["owner_user_id"] != user_id:
        return {"error": "Only the library owner can trigger a sync"}

    try:
        stats = sync_shared_library(library_id)
        return stats
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"sync_shared_library failed for {library_id}: {e}", exc_info=True)
        return {"error": f"Sync failed: {str(e)}"}


# ============ Asset Tools ============


def tool_list_assets(args: dict) -> dict:
    """List assets in the library, optionally filtered by category."""
    category = args.get("category", "").strip().lower()
    limit = min(int(args.get("limit", 50)), 100)

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    if category:
        rows = db.execute(
            """
            SELECT asset_id, category, filename, file_path, mime_type, file_size,
                   alt_text, description, created_at
            FROM library_assets
            WHERE category = ?
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (category, limit),
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT asset_id, category, filename, file_path, mime_type, file_size,
                   alt_text, description, created_at
            FROM library_assets
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (limit,),
        ).fetchall()

    assets = []
    for row in rows:
        asset = dict(row)
        asset["markdown_ref"] = f"![{asset['alt_text'] or asset['filename']}](assets/{asset['filename']})"
        assets.append(asset)

    return {
        "assets": assets,
        "count": len(assets),
        "usage_hint": "Use markdown references in notes like: ![alt text](assets/filename.png)",
    }


def tool_get_asset(args: dict) -> dict:
    """Get metadata for a specific asset."""
    asset_id = args.get("asset_id", "").strip()

    if not asset_id:
        return {"error": "asset_id is required"}

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    row = db.execute(
        """
        SELECT asset_id, category, filename, file_path, mime_type, file_size,
               alt_text, description, created_at
        FROM library_assets
        WHERE asset_id = ?
    """,
        (asset_id,),
    ).fetchone()

    if not row:
        return {"error": f"Asset not found: {asset_id}"}

    asset = dict(row)
    asset["markdown_ref"] = f"![{asset['alt_text'] or asset['filename']}](assets/{asset['filename']})"

    return asset


def tool_delete_asset(args: dict) -> dict:
    """Delete an asset from the library."""
    from .auth import get_user_installation_token
    from .rag.github_service import delete_file

    asset_id = args.get("asset_id", "").strip()
    confirm = args.get("confirm", False)

    if not asset_id:
        return {"error": "asset_id is required"}

    if not confirm:
        return {"error": "confirm must be true to delete an asset"}

    db = get_db()

    # Get asset info
    row = db.execute(
        """
        SELECT file_path, filename
        FROM library_assets
        WHERE asset_id = ?
    """,
        (asset_id,),
    ).fetchone()

    if not row:
        return {"error": f"Asset not found: {asset_id}"}

    # Get user credentials
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") and g.mcp_user else None
    if not user_id:
        return {"error": "Authentication required"}

    token = get_user_installation_token(user_id, "library")
    if not token:
        return {"error": "GitHub authorization required"}

    from .core import get_user_library_repo

    library_repo = get_user_library_repo()
    if not library_repo:
        return {"error": "Library repo not configured"}

    try:
        # Delete from GitHub
        try:
            delete_file(
                repo=library_repo,
                path=row["file_path"],
                message=f"Delete asset: {row['filename']}",
                token=token,
            )
        except Exception as e:
            if "404" not in str(e):
                raise
            # File already deleted from GitHub

        # Delete from database
        db.execute("DELETE FROM library_assets WHERE asset_id = ?", (asset_id,))
        commit_and_checkpoint(db)

        logger.info(f"MCP deleted asset: {asset_id}")

        return {"success": True, "deleted": asset_id, "filename": row["filename"]}

    except Exception as e:
        logger.error(f"Failed to delete asset {asset_id}: {e}")
        return {"error": str(e)}


def tool_get_asset_reference(args: dict) -> dict:
    """Get the markdown reference for an asset."""
    asset_id = args.get("asset_id", "").strip()
    custom_alt = args.get("alt_text", "").strip()

    if not asset_id:
        return {"error": "asset_id is required"}

    try:
        db, _role = get_library_db(args)
    except ValueError as e:
        return {"error": str(e)}

    row = db.execute(
        """
        SELECT filename, alt_text, mime_type, category
        FROM library_assets
        WHERE asset_id = ?
    """,
        (asset_id,),
    ).fetchone()

    if not row:
        return {"error": f"Asset not found: {asset_id}"}

    alt_text = custom_alt or row["alt_text"] or row["filename"]
    markdown_ref = f"![{alt_text}](assets/{row['filename']})"

    # Determine if it's an image or a link
    is_image = row["mime_type"] and row["mime_type"].startswith("image/")

    return {
        "asset_id": asset_id,
        "category": row["category"],
        "filename": row["filename"],
        "markdown_ref": markdown_ref,
        "is_image": is_image,
        "usage": f"Add this to your note content: {markdown_ref}",
    }


def tool_upload_asset(args: dict) -> dict:
    """Upload an image or file to a category's assets folder."""
    import base64
    import mimetypes
    import secrets

    from .auth import get_user_installation_token
    from .core import get_user_library_repo
    from .rag.github_service import create_binary_file, create_file, file_exists

    category = args.get("category", "").strip().lower()
    filename = args.get("filename", "").strip()
    content_base64 = args.get("content_base64", "").strip()
    alt_text = args.get("alt_text", "").strip()
    description = args.get("description", "").strip()

    # Validation
    if not category:
        return {"error": "category is required"}
    if not filename:
        return {"error": "filename is required"}
    if not content_base64:
        return {"error": "content_base64 is required"}

    # Decode base64 content
    try:
        content = base64.b64decode(content_base64)
    except Exception as e:
        return {"error": f"Invalid base64 content: {e}"}

    # Check file size (10MB max)
    max_file_size = 10 * 1024 * 1024
    if len(content) > max_file_size:
        return {"error": f"File too large. Maximum size is {max_file_size // 1024 // 1024}MB"}

    # Determine and validate MIME type
    mime_type = mimetypes.guess_type(filename)[0]
    allowed_mime_types = {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/svg+xml",
        "application/pdf",
        "text/csv",
        "application/json",
    }
    if mime_type not in allowed_mime_types:
        return {
            "error": f"File type not allowed: {mime_type}",
            "allowed_types": list(allowed_mime_types),
        }

    # Get user credentials
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") and g.mcp_user else None
    if not user_id:
        return {"error": "Authentication required"}

    token = get_user_installation_token(user_id, "library")
    if not token:
        return {"error": "GitHub authorization required"}

    library_repo = get_user_library_repo()
    if not library_repo:
        return {"error": "Library repo not configured"}

    try:
        db = get_db()

        # Generate asset ID and sanitized filename
        asset_id = f"asset-{secrets.token_hex(6)}"

        # Sanitize filename
        safe_filename = os.path.basename(filename).replace(" ", "-")
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._-")
        name_part, ext = os.path.splitext(safe_filename)
        if len(name_part) > 30:
            name_part = name_part[:30]

        # Create final filename with asset ID
        final_filename = f"{name_part}-{asset_id[-6:]}{ext}"
        file_path = f"{category}/assets/{final_filename}"

        # Ensure assets folder exists
        assets_folder = f"{category}/assets"
        gitkeep_path = f"{assets_folder}/.gitkeep"
        if not file_exists(library_repo, gitkeep_path, token):
            try:
                create_file(
                    repo=library_repo,
                    path=gitkeep_path,
                    content="# Assets folder for images and files\n",
                    message=f"Create assets folder for {category}",
                    token=token,
                )
                logger.info(f"Created assets folder: {assets_folder}")
            except Exception:
                pass  # Folder might already exist

        # Upload the file to GitHub
        result = create_binary_file(
            repo=library_repo,
            path=file_path,
            content=content,
            message=f"Add asset: {final_filename}",
            token=token,
        )

        # Store in database
        db.execute(
            """
            INSERT INTO library_assets
            (asset_id, category, filename, file_path, mime_type, file_size, alt_text, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                asset_id,
                category,
                final_filename,
                file_path,
                mime_type,
                len(content),
                alt_text,
                description,
            ),
        )
        commit_and_checkpoint(db)

        logger.info(f"MCP uploaded asset: {asset_id} -> {file_path}")

        # Generate markdown reference
        markdown_ref = f"![{alt_text or final_filename}](assets/{final_filename})"

        return {
            "success": True,
            "asset_id": asset_id,
            "filename": final_filename,
            "file_path": file_path,
            "mime_type": mime_type,
            "file_size": len(content),
            "markdown_ref": markdown_ref,
            "usage": f"Add this to your note content: {markdown_ref}",
            "commit_sha": result.get("commit", {}).get("sha", "")[:7],
        }

    except Exception as e:
        logger.error(f"Failed to upload asset: {e}", exc_info=True)
        return {"error": str(e)}


def tool_upload_markdown_as_note(args: dict) -> dict:
    """Upload a markdown file directly as a note, parsing and augmenting frontmatter."""
    from .auth import get_user_installation_token
    from .core import get_user_library_repo
    from .rag.database import get_user_categories
    from .rag.github_service import create_file

    markdown_content = args.get("markdown_content", "").strip()
    category_arg = args.get("category", "").lower().strip() if args.get("category") else None
    title_arg = args.get("title", "").strip() if args.get("title") else None
    subfolder = args.get("subfolder", "").strip() if args.get("subfolder") else None
    preserve_frontmatter = args.get("preserve_frontmatter", True)

    if not markdown_content:
        return {"error": "markdown_content is required"}

    # Parse existing frontmatter if present
    existing_frontmatter = {}
    body_content = markdown_content

    if markdown_content.startswith("---"):
        parts = markdown_content.split("---", 2)
        if len(parts) >= 3:
            # Has frontmatter
            frontmatter_text = parts[1].strip()
            body_content = parts[2].strip()

            # Parse YAML frontmatter (simple parser)
            for line in frontmatter_text.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    # Strip quotes from strings
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    # Handle booleans and nulls
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.lower() in ("null", "~", ""):
                        value = None
                    # Handle JSON arrays (simple check)
                    elif value.startswith("[") and value.endswith("]"):
                        try:
                            import json

                            value = json.loads(value)
                        except Exception:
                            pass  # Keep as string
                    existing_frontmatter[key] = value

    # Determine title: arg > frontmatter > first heading > error
    title = title_arg
    if not title:
        title = existing_frontmatter.get("title")
    if not title:
        # Try to extract from first heading
        import re

        heading_match = re.search(r"^#\s+(.+)$", body_content, re.MULTILINE)
        if heading_match:
            title = heading_match.group(1).strip()
    if not title:
        return {
            "error": (
                "Could not determine title. Provide 'title' parameter, include 'title' in "
                "frontmatter, or start content with a # heading."
            )
        }

    # Determine category: arg > frontmatter > error
    category = category_arg
    if not category:
        category = (
            existing_frontmatter.get("category", "").lower().strip() if existing_frontmatter.get("category") else None
        )
    if not category:
        return {"error": ("Category is required. Provide 'category' parameter or include 'category' in frontmatter.")}

    # Validate subfolder
    if subfolder and ("/" in subfolder or "\\" in subfolder):
        return {"error": ("Subfolder name cannot contain slashes. Use a simple name like 'projects' or 'backlog'.")}

    # Get user credentials
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") and g.mcp_user else None
    if not user_id:
        return {"error": "Authentication required"}

    token = get_user_installation_token(user_id, "library")
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    # Validate category
    db = get_db()
    categories = get_user_categories(db, user_id or "default")
    valid_categories = {c["name"] for c in categories}
    category_folders = {c["name"]: c["folder_name"] for c in categories}

    if category not in valid_categories:
        valid_cats = ", ".join(sorted(valid_categories))
        return {
            "error": f"Invalid category '{category}'. Must be one of: {valid_cats}",
            "hint": "Use the create_category tool to create a new category first.",
        }

    # Compute content hash
    content_hash = compute_content_hash(body_content)

    # Generate entry_id
    entry_id = generate_entry_id(category, title)

    # Check for collision
    collision = db.execute("SELECT entry_id FROM knowledge_entries WHERE entry_id = ?", (entry_id,)).fetchone()
    if collision:
        logger.info(f"Entry ID collision for '{title}', disambiguating with content hash")
        entry_id = generate_entry_id(category, title, content_hash)

    # Build file path
    slug = generate_slug(title)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    folder = category_folders.get(category, category)
    if subfolder:
        file_path = f"{folder}/{subfolder}/{date_str}-{slug}.md"
    else:
        file_path = f"{folder}/{date_str}-{slug}.md"

    # Build new frontmatter
    timestamp = datetime.utcnow().isoformat() + "Z"
    frontmatter_dict = {
        "id": entry_id,
        "title": title,
        "category": category,
        "created": timestamp,
        "content_hash": content_hash,
        "source": "mcp-claude",
        "domain_tags": [],
        "key_phrases": [],
    }

    # Preserve certain existing frontmatter fields if requested
    if preserve_frontmatter:
        preservable_fields = [
            "tags",
            "domain_tags",
            "key_phrases",
            "task_status",
            "due_date",
            "needs_chord",
            "chord_name",
            "chord_scope",
            "author",
            "aliases",
            "related",
            "links",
            "references",
            "source_url",
            "original_date",
        ]
        for field in preservable_fields:
            if field in existing_frontmatter and existing_frontmatter[field] is not None:
                frontmatter_dict[field] = existing_frontmatter[field]

    # Add optional fields
    if subfolder:
        frontmatter_dict["subfolder"] = subfolder

    # Get task_status and due_date from preserved frontmatter
    task_status = frontmatter_dict.get("task_status")
    due_date = frontmatter_dict.get("due_date")

    # Build frontmatter string
    frontmatter_lines = ["---"]
    for key, value in frontmatter_dict.items():
        if value is None:
            continue
        if isinstance(value, bool):
            frontmatter_lines.append(f"{key}: {str(value).lower()}")
        elif isinstance(value, list):
            frontmatter_lines.append(f"{key}: {json.dumps(value)}")
        elif isinstance(value, str) and ('"' in value or ":" in value or value.startswith("#")):
            frontmatter_lines.append(f'{key}: "{value}"')
        else:
            frontmatter_lines.append(f"{key}: {value}")
    frontmatter_lines.append("---")
    frontmatter_lines.append("")

    full_content = "\n".join(frontmatter_lines) + body_content

    # Create file in GitHub
    repo = get_user_library_repo(user_id)

    try:
        create_file(
            repo=repo,
            path=file_path,
            content=full_content,
            message=f"Upload markdown note via MCP: {title}",
            token=token,
        )
    except Exception as e:
        logger.error(f"Failed to create file in GitHub: {e}", exc_info=True)
        return {"error": f"Failed to save to GitHub: {str(e)}"}

    # Insert into database
    try:
        if task_status:
            cursor = db.execute(
                """
                INSERT INTO knowledge_entries
                (entry_id, title, category, content, file_path, source_transcript,
                task_status, due_date, content_hash, subfolder, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'mcp-claude',
                ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                (
                    entry_id,
                    title,
                    category,
                    body_content,
                    file_path,
                    task_status,
                    due_date,
                    content_hash,
                    subfolder,
                ),
            )
        else:
            cursor = db.execute(
                """
                INSERT INTO knowledge_entries
                (entry_id, title, category, content, file_path, source_transcript,
                content_hash, subfolder, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'mcp-claude', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                (entry_id, title, category, body_content, file_path, content_hash, subfolder),
            )
        row = cursor.fetchone()
        entry_db_id = row[0]
        commit_and_checkpoint(db)

        # Generate embedding
        _generate_embedding_for_entry(entry_db_id, entry_id, body_content)

    except Exception as e:
        logger.error(f"Failed to insert into database: {e}", exc_info=True)
        return {"error": f"Note saved to GitHub but failed to index locally: {str(e)}"}

    logger.info(f"MCP uploaded markdown note: {entry_id} - {title}")

    result = {
        "success": True,
        "entry_id": entry_id,
        "title": title,
        "category": category,
        "file_path": file_path,
        "frontmatter_preserved": preserve_frontmatter,
    }

    if subfolder:
        result["subfolder"] = subfolder
    if task_status:
        result["task_status"] = task_status
    if due_date:
        result["due_date"] = due_date

    return result


def tool_create_category(args: dict) -> dict:
    """Create a new category in the Legate Studio library.

    This is consistent with the categories.py web UI endpoint.
    Uses the same validation, folder naming conventions, and helper functions.
    """
    from .auth import get_user_installation_token
    from .categories import create_category_folder

    name = args.get("name", "").lower().strip()
    display_name = args.get("display_name", "").strip()
    description = args.get("description", "").strip() if args.get("description") else ""
    color = args.get("color", "#6366f1").strip()  # Default to indigo

    # Validation
    if not name:
        return {"error": "Category name is required"}
    if not display_name:
        return {"error": "Display name is required"}

    # Validate name format - match categories.py validation
    import re

    if not re.match(r"^[a-z][a-z0-9-]*$", name):
        return {
            "error": ("Category name must start with a letter and contain only lowercase letters, numbers, and hyphens")
        }

    if len(name) > 30:
        return {"error": "Category name must be 30 characters or less"}

    # Validate color format
    if color and not re.match(r"^#[0-9a-fA-F]{6}$", color):
        return {"error": "Color must be a valid hex code (e.g., '#10b981')"}

    # Get user credentials
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") and g.mcp_user else None
    if not user_id:
        return {"error": "Authentication required"}

    token = get_user_installation_token(user_id, "library")
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    db = get_db()

    # Check if category already exists (including inactive ones)
    existing = db.execute(
        "SELECT id, is_active FROM user_categories WHERE user_id = ? AND name = ?", (user_id, name)
    ).fetchone()

    if existing:
        if existing["is_active"] == 1:
            return {"error": f"Category '{name}' already exists"}
        else:
            # Reactivate the inactive category
            db.execute(
                """
                UPDATE user_categories
                SET is_active = 1, display_name = ?, description = ?,
                color = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (display_name, description, color, existing["id"]),
            )
            commit_and_checkpoint(db)

            logger.info(f"MCP reactivated category: {name} (id={existing['id']})")

            return {
                "success": True,
                "name": name,
                "display_name": display_name,
                "reactivated": True,
            }

    # Folder name convention: use category name directly (singular, matching DB defaults)
    folder_name = name

    # Determine sort order (after existing categories)
    max_order = (
        db.execute("SELECT MAX(sort_order) FROM user_categories WHERE user_id = ?", (user_id,)).fetchone()[0] or 0
    )
    sort_order = max_order + 1

    try:
        # Insert into database
        cursor = db.execute(
            """
            INSERT INTO user_categories
            (user_id, name, display_name, description, folder_name, sort_order, color)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (user_id, name, display_name, description, folder_name, sort_order, color),
        )
        commit_and_checkpoint(db)
        category_id = cursor.lastrowid

        # Create folder in GitHub using the shared helper
        folder_result = create_category_folder(folder_name, token=token, user_id=user_id)

        logger.info(f"MCP created category: {name} ({display_name}) for user {user_id}")

        return {
            "success": True,
            "id": category_id,
            "name": name,
            "display_name": display_name,
            "description": description,
            "folder_name": folder_name,
            "color": color,
            "sort_order": sort_order,
            "folder_created": folder_result.get("created", False),
        }

    except Exception as e:
        if "UNIQUE constraint" in str(e):
            return {"error": f"Category '{name}' already exists"}
        logger.error(f"Failed to create category: {e}", exc_info=True)
        return {"error": str(e)}


def _strip_yaml_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content.

    Frontmatter is delimited by --- at the start and end,
    appearing at the beginning of the file.
    """
    if not content.startswith("---"):
        return content

    # Find the closing ---
    lines = content.split("\n")
    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        # No closing ---, return original
        return content

    # Return content after frontmatter, stripping leading newlines
    result = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return result


def tool_download_note(args: dict) -> dict:
    """Download a single note to a local filesystem path."""

    entry_id = args.get("entry_id", "").strip() if args.get("entry_id") else None
    file_path_lookup = args.get("file_path", "").strip() if args.get("file_path") else None
    destination = args.get("destination", "").strip()
    strip_frontmatter = args.get("strip_frontmatter", False)

    if not destination:
        return {"error": "destination is required"}

    if not entry_id and not file_path_lookup:
        return {"error": "Either entry_id or file_path is required"}

    db = get_db()
    entry = None
    lookup_method = None

    # Try entry_id first
    if entry_id:
        entry = db.execute(
            """
            SELECT entry_id, title, category, content, file_path
            FROM knowledge_entries
            WHERE entry_id = ?
            """,
            (entry_id,),
        ).fetchone()
        lookup_method = "entry_id"

    # Fallback to file_path
    if not entry and file_path_lookup:
        entry = db.execute(
            """
            SELECT entry_id, title, category, content, file_path
            FROM knowledge_entries
            WHERE file_path = ?
            """,
            (file_path_lookup,),
        ).fetchone()
        lookup_method = "file_path"

    if not entry:
        search_term = entry_id or file_path_lookup
        return {"error": f"Note not found: {search_term}"}

    content = entry["content"] or ""

    if strip_frontmatter:
        content = _strip_yaml_frontmatter(content)

    # Ensure destination directory exists
    dest_dir = os.path.dirname(destination)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    # Write file with UTF-8 encoding
    try:
        with open(destination, "w", encoding="utf-8") as f:
            f.write(content)
        bytes_written = len(content.encode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to write file {destination}: {e}")
        return {"error": f"Failed to write file: {str(e)}"}

    return {
        "success": True,
        "source": entry["entry_id"],
        "destination": destination,
        "bytes_written": bytes_written,
        "lookup_method": lookup_method,
    }


def tool_download_notes(args: dict) -> dict:
    """Bulk download notes from a category/subfolder to a local directory."""
    import fnmatch

    from .rag.database import get_user_categories

    category = args.get("category", "").lower().strip()
    subfolder = args.get("subfolder", "").strip() if args.get("subfolder") else None
    destination_dir = args.get("destination_dir", "").strip()
    pattern = args.get("pattern", "").strip() if args.get("pattern") else None
    strip_frontmatter = args.get("strip_frontmatter", False)
    flatten = args.get("flatten", True)

    if not category:
        return {"error": "category is required"}

    if not destination_dir:
        return {"error": "destination_dir is required"}

    db = get_db()
    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") else None

    # Validate category
    categories = get_user_categories(db, user_id or "default")
    valid_categories = {c["name"] for c in categories}

    if category not in valid_categories:
        return {"error": f"Invalid category. Must be one of: {', '.join(sorted(valid_categories))}"}

    # Query notes in this category/subfolder combination
    if subfolder:
        rows = db.execute(
            """
            SELECT entry_id, title, content, file_path, subfolder
            FROM knowledge_entries
            WHERE category = ? AND subfolder = ?
            ORDER BY file_path ASC
            """,
            (category, subfolder),
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT entry_id, title, content, file_path, subfolder
            FROM knowledge_entries
            WHERE category = ?
            ORDER BY file_path ASC
            """,
            (category,),
        ).fetchall()

    # Apply pattern filter if specified
    if pattern:
        filtered_rows = []
        for row in rows:
            file_path = row["file_path"] or ""
            filename = os.path.basename(file_path)
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(file_path, pattern):
                filtered_rows.append(row)
        rows = filtered_rows

    if not rows:
        return {
            "success": True,
            "files_written": 0,
            "total_bytes": 0,
            "files": [],
            "message": "No matching notes found",
        }

    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    files_written = []
    total_bytes = 0
    errors = []

    for row in rows:
        content = row["content"] or ""

        if strip_frontmatter:
            content = _strip_yaml_frontmatter(content)

        # Determine destination filename
        source_path = row["file_path"] or f"{row['entry_id']}.md"
        filename = os.path.basename(source_path)

        if flatten:
            dest_path = os.path.join(destination_dir, filename)
        else:
            # Preserve subfolder structure
            row_subfolder = row["subfolder"]
            if row_subfolder:
                dest_path = os.path.join(destination_dir, row_subfolder, filename)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            else:
                dest_path = os.path.join(destination_dir, filename)

        try:
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(content)
            bytes_written = len(content.encode("utf-8"))
            total_bytes += bytes_written
            files_written.append({"source": row["entry_id"], "destination": dest_path, "bytes": bytes_written})
        except Exception as e:
            logger.error(f"Failed to write file {dest_path}: {e}")
            errors.append({"source": row["entry_id"], "destination": dest_path, "error": str(e)})

    result = {
        "success": len(errors) == 0,
        "files_written": len(files_written),
        "total_bytes": total_bytes,
        "files": files_written,
    }

    if errors:
        result["errors"] = errors

    return result


def tool_download_notes_batch(args: dict) -> dict:
    """Download specific notes by entry ID to specified destinations."""

    notes = args.get("notes", [])
    strip_frontmatter = args.get("strip_frontmatter", False)

    if not notes:
        return {"error": "notes array is required and cannot be empty"}

    if not isinstance(notes, list):
        return {"error": "notes must be an array"}

    # Validate all notes have required fields
    for i, note in enumerate(notes):
        if not isinstance(note, dict):
            return {"error": f"notes[{i}] must be an object"}
        if not note.get("entry_id"):
            return {"error": f"notes[{i}].entry_id is required"}
        if not note.get("destination"):
            return {"error": f"notes[{i}].destination is required"}

    db = get_db()

    # Fetch all requested notes in one query
    entry_ids = [n["entry_id"].strip() for n in notes]
    placeholders = ",".join(["?" for _ in entry_ids])

    rows = db.execute(
        f"""
        SELECT entry_id, title, content, file_path
        FROM knowledge_entries
        WHERE entry_id IN ({placeholders})
        """,
        entry_ids,
    ).fetchall()

    # Create lookup map
    entries_map = {row["entry_id"]: row for row in rows}

    files_written = []
    total_bytes = 0
    errors = []

    for note in notes:
        entry_id = note["entry_id"].strip()
        destination = note["destination"].strip()

        if entry_id not in entries_map:
            errors.append({"source": entry_id, "destination": destination, "error": "Note not found"})
            continue

        entry = entries_map[entry_id]
        content = entry["content"] or ""

        if strip_frontmatter:
            content = _strip_yaml_frontmatter(content)

        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        try:
            with open(destination, "w", encoding="utf-8") as f:
                f.write(content)
            bytes_written = len(content.encode("utf-8"))
            total_bytes += bytes_written
            files_written.append({"source": entry_id, "destination": destination, "bytes": bytes_written})
        except Exception as e:
            logger.error(f"Failed to write file {destination}: {e}")
            errors.append({"source": entry_id, "destination": destination, "error": str(e)})

    result = {
        "success": len(errors) == 0,
        "files_written": len(files_written),
        "total_bytes": total_bytes,
        "files": files_written,
    }

    if errors:
        result["errors"] = errors

    return result


# ============ Library Management Tools ============


def tool_create_shared_library(args: dict) -> dict:
    """Create a new shared library with a private GitHub repo and per-library SQLite DB.

    Requires managed subscription tier. Provisions Legate.Library.{slug} on GitHub
    under the owner's account, inserts into shared_libraries, and inserts owner membership.
    """
    import re
    import uuid

    import requests

    from .auth import get_user_installation_token
    from .rag.database import init_db as init_shared_meta_db
    from .rag.database import init_shared_library_db

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") and g.mcp_user else None
    github_login = g.mcp_user.get("sub") if hasattr(g, "mcp_user") and g.mcp_user else None

    if not user_id:
        return {"error": "Authentication required"}

    # Tier gate
    try:
        require_managed_tier(user_id)
    except ValueError as e:
        return {"error": str(e)}

    name = (args.get("name") or "").strip()
    if not name:
        return {"error": "Library name is required"}
    if len(name) > 100:
        return {"error": "Library name must be 100 characters or less"}

    # Generate slug from provided slug or name
    raw_slug = (args.get("slug") or "").strip()
    if raw_slug:
        slug = re.sub(r"[^a-z0-9-]", "-", raw_slug.lower()).strip("-")
        slug = re.sub(r"-+", "-", slug)
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        slug = re.sub(r"-+", "-", slug)
    slug = slug[:50]

    if not slug or not re.match(r"^[a-z0-9][a-z0-9-]*$", slug):
        return {"error": "Could not generate a valid slug. Use only letters, numbers, and hyphens."}

    description = (args.get("description") or "").strip()

    # Get owner's installation token to create the GitHub repo
    token = get_user_installation_token(user_id, "library")
    if not token:
        return {"error": "GitHub authorization required. Please re-authenticate."}

    shared_db = init_shared_meta_db()

    # Check slug uniqueness for this owner
    existing = shared_db.execute(
        "SELECT id FROM shared_libraries WHERE owner_user_id = ? AND slug = ? AND status = 'active'",
        (user_id, slug),
    ).fetchone()
    if existing:
        return {"error": f"You already have a shared library with slug '{slug}'. Choose a different slug."}

    library_id = str(uuid.uuid4())
    repo_name = f"Legate.Library.{slug}"
    repo_full_name = f"{github_login}/{repo_name}"

    # Create GitHub repo (private, auto_init so it has a main branch)
    try:
        gh_resp = requests.post(
            "https://api.github.com/user/repos",
            json={
                "name": repo_name,
                "description": description or f"Legate shared library: {name}",
                "private": True,
                "auto_init": True,
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=20,
        )
        if gh_resp.status_code == 422:
            # Repo already exists
            error_data = gh_resp.json()
            errors = error_data.get("errors", [])
            already_exists = any(e.get("message", "").startswith("name already exists") for e in errors)
            if already_exists:
                return {"error": f"GitHub repo '{repo_name}' already exists on your account. Choose a different slug."}
        if not gh_resp.ok:
            logger.error(f"GitHub repo creation failed: {gh_resp.status_code} {gh_resp.text[:500]}")
            return {"error": f"Failed to create GitHub repo: {gh_resp.status_code}"}
        repo_data = gh_resp.json()
        repo_full_name = repo_data.get("full_name", repo_full_name)
    except requests.RequestException as e:
        logger.error(f"GitHub API error creating shared library repo: {e}")
        return {"error": f"Failed to create GitHub repo: {str(e)}"}

    # Insert shared_libraries row
    try:
        shared_db.execute(
            """
            INSERT INTO shared_libraries (id, name, slug, owner_user_id, repo_full_name, description, status)
            VALUES (?, ?, ?, ?, ?, ?, 'active')
            """,
            (library_id, name, slug, user_id, repo_full_name, description or None),
        )
        # Insert owner membership
        shared_db.execute(
            """
            INSERT INTO shared_library_members
            (shared_library_id, user_id, role, status, accepted_at)
            VALUES (?, ?, 'owner', 'active', CURRENT_TIMESTAMP)
            """,
            (library_id, user_id),
        )
        shared_db.commit()
    except Exception as e:
        logger.error(f"DB error inserting shared library: {e}", exc_info=True)
        return {"error": f"Failed to save library to database: {str(e)}"}

    # Initialize per-library SQLite DB
    try:
        init_shared_library_db(library_id)
    except Exception as e:
        logger.error(f"Failed to init shared library DB for {library_id}: {e}", exc_info=True)
        # Non-fatal — the DB will be initialized on first access
        logger.warning("Shared library DB init deferred — will init on first access")

    logger.info(f"Created shared library '{name}' ({library_id}) for user {user_id}, repo {repo_full_name}")

    return {
        "success": True,
        "library_id": library_id,
        "name": name,
        "slug": slug,
        "repo_full_name": repo_full_name,
        "description": description or None,
        "message": (
            f"Shared library '{name}' created. Use library_id='{library_id}' to invite collaborators "
            f"and access the library with other tools."
        ),
    }


def tool_list_libraries(args: dict) -> dict:
    """List personal library plus all shared libraries where the user is an active member."""
    from .rag.database import init_db as init_shared_meta_db

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") and g.mcp_user else None
    github_login = g.mcp_user.get("sub") if hasattr(g, "mcp_user") and g.mcp_user else None

    if not user_id:
        return {"error": "Authentication required"}

    # Personal library entry (always present)
    libraries = [
        {
            "library_id": None,
            "name": "Personal Library",
            "slug": None,
            "role": "owner",
            "type": "personal",
            "member_count": 1,
        }
    ]

    try:
        shared_db = init_shared_meta_db()

        # Fetch all shared libraries where user is an active member
        rows = shared_db.execute(
            """
            SELECT
                sl.id AS library_id,
                sl.name,
                sl.slug,
                sl.repo_full_name,
                sl.description,
                sl.owner_user_id,
                slm.role,
                (
                    SELECT COUNT(*)
                    FROM shared_library_members m2
                    WHERE m2.shared_library_id = sl.id AND m2.status = 'active'
                ) AS member_count
            FROM shared_libraries sl
            JOIN shared_library_members slm
                ON sl.id = slm.shared_library_id
            WHERE slm.user_id = ? AND slm.status = 'active' AND sl.status = 'active'
            ORDER BY sl.name
            """,
            (user_id,),
        ).fetchall()

        for row in rows:
            libraries.append({
                "library_id": row["library_id"],
                "name": row["name"],
                "slug": row["slug"],
                "role": row["role"],
                "type": "shared",
                "repo_full_name": row["repo_full_name"],
                "description": row["description"],
                "member_count": row["member_count"],
            })

        # Also fetch pending invitations so the user can see what they can accept
        pending_rows = shared_db.execute(
            """
            SELECT sl.id AS library_id, sl.name, sl.slug, sl.description
            FROM shared_libraries sl
            JOIN shared_library_members slm ON sl.id = slm.shared_library_id
            WHERE slm.user_id = ? AND slm.status = 'invited' AND sl.status = 'active'
            ORDER BY sl.name
            """,
            (user_id,),
        ).fetchall()

        pending_invitations = [
            {
                "library_id": row["library_id"],
                "name": row["name"],
                "slug": row["slug"],
                "description": row["description"],
            }
            for row in pending_rows
        ]

    except Exception as e:
        logger.error(f"list_libraries failed for {user_id}: {e}", exc_info=True)
        return {"error": f"Failed to list libraries: {str(e)}"}

    result = {
        "libraries": libraries,
        "total": len(libraries),
    }
    if pending_invitations:
        result["pending_invitations"] = pending_invitations
        result["invitation_hint"] = "Call accept_invitation(library_id) to join a pending library."

    return result


def tool_invite_collaborator(args: dict) -> dict:
    """Invite a GitHub user to a shared library.

    Verifies caller is owner, checks target user exists in legato.db, inserts
    membership row (status='invited'), and adds them as a GitHub collaborator.
    """
    import requests

    from .auth import get_user_installation_token
    from .rag.database import init_db as init_shared_meta_db

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") and g.mcp_user else None
    if not user_id:
        return {"error": "Authentication required"}

    library_id = (args.get("library_id") or "").strip()
    github_login = (args.get("github_login") or "").strip()

    if not library_id:
        return {"error": "library_id is required"}
    if not github_login:
        return {"error": "github_login is required"}

    shared_db = init_shared_meta_db()

    # Verify caller is owner
    lib_row = shared_db.execute(
        """
        SELECT sl.repo_full_name, sl.name, sl.owner_user_id
        FROM shared_libraries sl
        JOIN shared_library_members slm ON sl.id = slm.shared_library_id
        WHERE sl.id = ? AND slm.user_id = ? AND slm.role = 'owner' AND slm.status = 'active'
        """,
        (library_id, user_id),
    ).fetchone()

    if not lib_row:
        return {"error": "Library not found or you are not the owner"}

    repo_full_name = lib_row["repo_full_name"]
    library_name = lib_row["name"]

    # Verify target user exists in our system
    target_user = shared_db.execute(
        "SELECT user_id FROM users WHERE github_login = ?",
        (github_login,),
    ).fetchone()

    if not target_user:
        return {
            "error": (
                f"User '{github_login}' not found in Legate Studio. "
                "They must sign up at legate.studio first."
            )
        }

    target_user_id = target_user["user_id"]

    # Check they're not already a member
    existing = shared_db.execute(
        "SELECT status FROM shared_library_members WHERE shared_library_id = ? AND user_id = ?",
        (library_id, target_user_id),
    ).fetchone()

    if existing:
        status = existing["status"]
        if status == "active":
            return {"error": f"'{github_login}' is already an active member of this library"}
        if status == "invited":
            return {"error": f"'{github_login}' already has a pending invitation to this library"}
        # Revoked — allow re-invitation
        shared_db.execute(
            """
            UPDATE shared_library_members
            SET status = 'invited', invited_at = CURRENT_TIMESTAMP, accepted_at = NULL
            WHERE shared_library_id = ? AND user_id = ?
            """,
            (library_id, target_user_id),
        )
    else:
        shared_db.execute(
            """
            INSERT INTO shared_library_members (shared_library_id, user_id, role, status)
            VALUES (?, ?, 'collaborator', 'invited')
            """,
            (library_id, target_user_id),
        )

    shared_db.commit()

    # Add as GitHub collaborator (best-effort — don't fail if GitHub API errors)
    github_error = None
    try:
        token = get_user_installation_token(user_id, "library")
        if token and repo_full_name:
            gh_resp = requests.put(
                f"https://api.github.com/repos/{repo_full_name}/collaborators/{github_login}",
                json={"permission": "push"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=15,
            )
            if not gh_resp.ok and gh_resp.status_code != 201:
                github_error = f"GitHub collaborator invite returned {gh_resp.status_code}"
                logger.warning(f"invite_collaborator GitHub error: {github_error}")
    except Exception as e:
        github_error = str(e)
        logger.warning(f"invite_collaborator GitHub API error: {e}")

    result = {
        "success": True,
        "library_id": library_id,
        "library_name": library_name,
        "github_login": github_login,
        "status": "invited",
        "message": (
            f"'{github_login}' has been invited to '{library_name}'. "
            "They can accept with accept_invitation(library_id)."
        ),
    }
    if github_error:
        result["github_warning"] = f"DB updated but GitHub collaborator invite may have failed: {github_error}"

    return result


def tool_accept_invitation(args: dict) -> dict:
    """Accept a pending shared library invitation.

    Verifies current user has status='invited', then sets to 'active'.
    """
    from .rag.database import init_db as init_shared_meta_db

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") and g.mcp_user else None
    if not user_id:
        return {"error": "Authentication required"}

    library_id = (args.get("library_id") or "").strip()
    if not library_id:
        return {"error": "library_id is required"}

    shared_db = init_shared_meta_db()

    # Verify there's a pending invitation
    row = shared_db.execute(
        """
        SELECT slm.id, sl.name, sl.slug
        FROM shared_library_members slm
        JOIN shared_libraries sl ON sl.id = slm.shared_library_id
        WHERE slm.shared_library_id = ? AND slm.user_id = ? AND slm.status = 'invited'
        """,
        (library_id, user_id),
    ).fetchone()

    if not row:
        # Check if already active
        active = shared_db.execute(
            "SELECT 1 FROM shared_library_members WHERE shared_library_id = ? AND user_id = ? AND status = 'active'",
            (library_id, user_id),
        ).fetchone()
        if active:
            return {"error": "You are already an active member of this library"}
        return {"error": "No pending invitation found for this library"}

    library_name = row["name"]
    library_slug = row["slug"]

    shared_db.execute(
        """
        UPDATE shared_library_members
        SET status = 'active', accepted_at = CURRENT_TIMESTAMP
        WHERE shared_library_id = ? AND user_id = ?
        """,
        (library_id, user_id),
    )
    shared_db.commit()

    return {
        "success": True,
        "library_id": library_id,
        "library_name": library_name,
        "library_slug": library_slug,
        "message": (
            f"You are now an active collaborator in '{library_name}'. "
            f"Use library_id='{library_id}' with other tools to access this library. "
            "Create drafts with create_draft() — owners merge them into the library."
        ),
    }


def tool_remove_collaborator(args: dict) -> dict:
    """Remove a collaborator from a shared library.

    Owner-only. Sets membership status to 'revoked', removes GitHub collaborator,
    and deletes any unsubmitted drafts by the removed user.
    """
    import requests

    from .auth import get_user_installation_token
    from .rag.database import get_shared_library_db
    from .rag.database import init_db as init_shared_meta_db

    user_id = g.mcp_user.get("user_id") if hasattr(g, "mcp_user") and g.mcp_user else None
    if not user_id:
        return {"error": "Authentication required"}

    library_id = (args.get("library_id") or "").strip()
    github_login = (args.get("github_login") or "").strip()

    if not library_id:
        return {"error": "library_id is required"}
    if not github_login:
        return {"error": "github_login is required"}

    shared_db = init_shared_meta_db()

    # Verify caller is owner
    lib_row = shared_db.execute(
        """
        SELECT sl.repo_full_name, sl.name
        FROM shared_libraries sl
        JOIN shared_library_members slm ON sl.id = slm.shared_library_id
        WHERE sl.id = ? AND slm.user_id = ? AND slm.role = 'owner' AND slm.status = 'active'
        """,
        (library_id, user_id),
    ).fetchone()

    if not lib_row:
        return {"error": "Library not found or you are not the owner"}

    repo_full_name = lib_row["repo_full_name"]
    library_name = lib_row["name"]

    # Look up target user
    target_user = shared_db.execute(
        "SELECT user_id FROM users WHERE github_login = ?",
        (github_login,),
    ).fetchone()

    if not target_user:
        return {"error": f"User '{github_login}' not found in Legate Studio"}

    target_user_id = target_user["user_id"]

    # Cannot remove the owner
    if target_user_id == user_id:
        return {"error": "You cannot remove yourself as owner. Transfer ownership or archive the library instead."}

    member_row = shared_db.execute(
        "SELECT status, role FROM shared_library_members WHERE shared_library_id = ? AND user_id = ?",
        (library_id, target_user_id),
    ).fetchone()

    if not member_row:
        return {"error": f"'{github_login}' is not a member of this library"}

    if member_row["status"] == "revoked":
        return {"error": f"'{github_login}' has already been removed from this library"}

    # Set status to revoked
    shared_db.execute(
        """
        UPDATE shared_library_members
        SET status = 'revoked'
        WHERE shared_library_id = ? AND user_id = ?
        """,
        (library_id, target_user_id),
    )
    shared_db.commit()

    # Delete pending (unsubmitted) drafts from the shared library DB
    drafts_deleted = 0
    try:
        lib_db = get_shared_library_db(library_id)
        cursor = lib_db.execute(
            "DELETE FROM drafts WHERE author_user_id = ? AND status = 'draft'",
            (target_user_id,),
        )
        drafts_deleted = cursor.rowcount
        lib_db.commit()
    except Exception as e:
        logger.warning(f"remove_collaborator: failed to delete drafts for {target_user_id}: {e}")

    # Remove GitHub collaborator (best-effort)
    github_error = None
    try:
        token = get_user_installation_token(user_id, "library")
        if token and repo_full_name:
            gh_resp = requests.delete(
                f"https://api.github.com/repos/{repo_full_name}/collaborators/{github_login}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=15,
            )
            if not gh_resp.ok and gh_resp.status_code != 204:
                github_error = f"GitHub collaborator removal returned {gh_resp.status_code}"
                logger.warning(f"remove_collaborator GitHub error: {github_error}")
    except Exception as e:
        github_error = str(e)
        logger.warning(f"remove_collaborator GitHub API error: {e}")

    result = {
        "success": True,
        "library_id": library_id,
        "library_name": library_name,
        "github_login": github_login,
        "drafts_deleted": drafts_deleted,
        "message": f"'{github_login}' has been removed from '{library_name}'.",
    }
    if github_error:
        result["github_warning"] = f"Membership revoked but GitHub removal may have failed: {github_error}"

    return result


# ============ Resource Handlers ============

RESOURCES = [
    {
        "uri": "legato://library/stats",
        "name": "Library Statistics",
        "description": "Overview of the Legate Studio library - note counts, categories, etc.",
        "mimeType": "application/json",
    }
]


def handle_resources_list(params: dict) -> dict:
    """Return list of available resources."""
    return {"resources": RESOURCES}


def handle_resource_read(params: dict) -> dict:
    """Read a specific resource."""
    uri = params.get("uri", "")

    if uri == "legato://library/stats":
        db = get_db()

        # Get total count
        total = db.execute("SELECT COUNT(*) FROM knowledge_entries").fetchone()[0]

        # Get category counts
        categories = db.execute("""
            SELECT category, COUNT(*) as count
            FROM knowledge_entries
            GROUP BY category
            ORDER BY count DESC
        """).fetchall()

        # Get recent activity
        recent = db.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM knowledge_entries
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            LIMIT 7
        """).fetchall()

        content = json.dumps(
            {
                "total_notes": total,
                "categories": [{"name": c["category"], "count": c["count"]} for c in categories],
                "recent_activity": [{"date": r["date"], "count": r["count"]} for r in recent],
            },
            indent=2,
        )

        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": content}]}

    raise MCPError(-32602, f"Unknown resource: {uri}")


# ============ Prompt Handlers ============

PROMPTS = [
    {
        "name": "summarize_notes",
        "description": "Summarize notes from a category or search results",
        "arguments": [
            {"name": "category", "description": "Category to summarize", "required": False},
            {
                "name": "query",
                "description": "Search query to find notes to summarize",
                "required": False,
            },
        ],
    }
]


def handle_prompts_list(params: dict) -> dict:
    """Return list of available prompts."""
    return {"prompts": PROMPTS}


def handle_prompt_get(params: dict) -> dict:
    """Get a specific prompt template."""
    name = params.get("name")
    arguments = params.get("arguments", {})

    if name == "summarize_notes":
        category = arguments.get("category")
        query = arguments.get("query")

        if category:
            context = f"Summarize all notes in the '{category}' category."
        elif query:
            context = f"Search for notes about '{query}' and summarize the key insights."
        else:
            context = "Summarize the recent notes in the library."

        return {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": (
                            f"{context}\n\nUse the search_library and get_note"
                            " tools to find and read the notes,"
                            " then provide a comprehensive summary."
                        ),
                    },
                }
            ]
        }

    raise MCPError(-32602, f"Unknown prompt: {name}")
