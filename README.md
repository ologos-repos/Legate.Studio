# Legate Studio

A personal knowledge platform with AI-powered capture, search, and chat. Your second brain — backed by GitHub, served by Flask, secured by design. Supports shared libraries for collaborative knowledge sharing, a full MCP integration for Claude, and a draft/merge workflow for team contributions.

## Stack

- **Runtime**: Python 3.11+, [uv](https://github.com/astral-sh/uv) for dependency management (never pip, never npm)
- **Web**: Flask (application factory pattern, blueprints, Jinja2 templates)
- **Database**: SQLite in WAL mode (`legato.db`, `agents.db`, `chat.db` — per-user in multi-tenant)
- **Auth**: GitHub App OAuth (multi-tenant) or GitHub OAuth App (single-tenant)
- **Billing**: Stripe (subscriptions + one-time top-up credits)
- **Encryption**: Fernet per-user encryption for stored API keys (via `crypto.py`)
- **Rate limiting**: Flask-Limiter (in-memory by default, Redis optional)
- **Embeddings**: Gemini `text-embedding-004` (768-dim, default), OpenAI `text-embedding-ada-002` (1536-dim), or Ollama (local)
- **Transcription**: Gemini Flash multimodal (replaces Whisper)
- **AI Chat**: Anthropic Claude, OpenAI, and Google Gemini (provider abstraction layer)
- **MCP**: OAuth 2.1 Authorization Server with Dynamic Client Registration (protocol version `2025-06-18`)
- **Deployment**: Fly.io (persistent `/data` volume for SQLite + secrets) or Docker
- **Error tracking**: Sentry (optional, set `SENTRY_DSN`)
- **Theme**: Catppuccin Mocha dark theme across all UI

## Deployment Modes

| Mode | Description |
|------|-------------|
| `single-tenant` (default) | One user, DIY install, uses `SYSTEM_PAT` and `GH_OAUTH_CLIENT_ID` |
| `multi-tenant` | SaaS mode, GitHub App auth, per-user DBs, Stripe billing, trial/BYOK/managed tiers |

Set `LEGATO_MODE=multi-tenant` to enable SaaS mode.

## Pricing Tiers

| Tier | Price | Description |
|------|-------|-------------|
| `trial` | Free (14 days) | Full platform access, bring your own API keys |
| `byok` | $0.99/mo | Unlimited, bring your own Gemini/Anthropic/OpenAI keys |
| `managed_lite` | $2.99/mo | Platform keys, $2.69/mo token credits |
| `managed_standard` | $10/mo | Platform keys, $9.00/mo token credits |
| `managed_plus` | $20/mo | Platform keys, $18.00/mo token credits |

Top-ups: $2.99 per purchase adds $2.69 in token credits (10% platform margin).

Shared libraries require a managed tier (`managed_lite` or above).

## Features

### Core Knowledge Management
- **Library**: Full CRUD for notes with markdown rendering, YAML frontmatter, and GitHub-backed storage
- **Categories**: Organize notes into typed categories (concept, epiphany, reflection, worklog, etc.)
- **Subfolders**: Nested organization within categories for deeper structure
- **Search**: Hybrid semantic (AI embeddings) + keyword search with confidence bucketing
- **Note linking**: Bidirectional relationships between notes (related, depends_on, blocks, implements, references, contradicts, supports)
- **Note context**: View a note with its full graph — linked notes, semantic neighbors, related projects
- **Tasks**: Any note can be a task with status tracking (pending, in_progress, blocked, done) and due dates

### Shared Libraries
- **Collaborative knowledge sharing**: Create shared libraries backed by private GitHub repos
- **Role-based access**: Owners have full write access; collaborators use the draft/merge workflow
- **Draft & Merge workflow**: Collaborators propose changes (new notes, edits, deletions) as drafts; owners review, merge, or reject with feedback
- **Invitations**: Invite GitHub users to collaborate; they accept to gain access
- **Managed tier required**: Shared libraries are gated to managed subscription tiers

### AI & Processing
- **Chat**: AI chat with library context (RAG) — Anthropic, OpenAI, and Gemini providers
- **Motif**: Voice and text capture pipeline — upload audio/text, process into structured notes
- **Embeddings**: Automatic embedding generation for semantic search (Gemini, OpenAI, or Ollama)
- **Transcript processing**: Background worker for async motif processing with job queue

### Assets
- **File attachments**: Upload images and files (PNG, JPEG, GIF, WebP, SVG, PDF) to category asset folders
- **Markdown references**: Get properly formatted markdown references for embedding assets in notes
- **GitHub-backed**: Assets stored alongside notes in the GitHub repo

### Chord (GitHub Copilot Agent)
- **Project spawning**: Queue projects from library notes for GitHub Copilot implementation
- **Agent queue**: Human-in-the-loop approval before Copilot execution
- **Multi-phase projects**: Support for simple (single-PR) and complex (multi-phase chord) projects

### Import & Export
- **Markdown ZIP import**: Upload a ZIP of markdown files, classify into categories, preview, and confirm
- **Markdown upload**: Direct markdown-to-note upload with frontmatter parsing
- **Bulk note retrieval**: Fetch multiple notes by ID, category, or glob pattern

### Marketing & SEO
- Landing page, pricing, features, FAQ, about, contact, privacy, terms, and security pages
- SEO content pages: MCP-first PKM, memory layer for AI, persistent memory for AI, PKB for AI, voice notes to KB
- MCP documentation page at `/docs/mcp`

### Admin & Operations
- **Admin console**: User management, feature flags, system overview
- **Recovery CLI**: Validate library integrity, normalize IDs, fix frontmatter, merge categories, rebuild hashes (`legate-recovery`)
- **Sync diagnostics**: Verify and repair consistency between database entries and GitHub files
- **Health check**: `/health` endpoint for load balancer probes

## Local Development

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone https://github.com/ologos-repos/Legate.Studio
cd Legate.Studio

# Install dependencies
uv sync

# Copy and edit environment
cp .env.example .env
# Edit .env with your values

# Run locally
uv run python src/main.py
```

## Environment Variables

### Required (all modes)

```bash
FLASK_SECRET_KEY=<hex-64-chars>     # Session signing key (auto-generated if unset)
FLASK_ENV=development                # or 'production'
```

### Single-tenant mode

```bash
LEGATO_MODE=single-tenant            # Default
GH_OAUTH_CLIENT_ID=<github-client-id>
GH_OAUTH_CLIENT_SECRET=<github-secret>
GH_ALLOWED_USERS=yourusername        # Comma-separated allowlist
SYSTEM_PAT=<github-pat>              # PAT with repo scope, for library sync
```

### Multi-tenant mode

```bash
LEGATO_MODE=multi-tenant
GITHUB_APP_ID=<app-id>
GITHUB_APP_CLIENT_ID=<app-client-id>
GITHUB_APP_CLIENT_SECRET=<app-client-secret>
GITHUB_APP_PRIVATE_KEY=<pem-contents-or-path>
GITHUB_APP_SLUG=legate-studio
GITHUB_WEBHOOK_SECRET=<webhook-secret>
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<strong-password>
ADMIN_USERS=yourgithubusername
```

### Security (recommended in production)

```bash
# Separate JWT signing key (falls back to FLASK_SECRET_KEY with a warning if unset)
JWT_SECRET_KEY=<hex-64-chars>

# Master encryption key for per-user API key storage
# If unset, key is stored in legato.db (less secure — set this in production)
LEGATE_MASTER_KEY=<base64-fernet-key>

# Redis for distributed rate limiting (falls back to in-memory if unset)
REDIS_URL=redis://localhost:6379/0
```

### AI providers (platform keys for managed tier)

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIzaSy...
```

### Platform backend: Vertex AI (optional)

Instead of direct provider keys, the managed/platform tier can serve inference
through Google Vertex AI — letting you serve any Vertex-hosted model (Claude via
`AnthropicVertex`, Gemini via `google-genai`) behind one GCP billing + auth
surface. The model family is inferred from the `VERTEX_MODEL` prefix, so the
pipeline stays provider-agnostic. BYOK users keep using their own direct provider
keys unchanged. Inert unless both `LEGATE_PLATFORM_BACKEND=vertex` and
`VERTEX_PROJECT` are set.

```bash
LEGATE_PLATFORM_BACKEND=vertex          # default 'direct'
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-central1             # Claude also accepts global/us/eu
VERTEX_MODEL=gemini-2.5-flash           # or e.g. claude-sonnet-4-5@20250929
GOOGLE_SERVICE_ACCOUNT_JSON={...}       # inline SA JSON; omit to use ambient ADC
```

### Optional

```bash
SENTRY_DSN=https://...@sentry.io/...   # Error tracking
SENTRY_ENVIRONMENT=production
DATA_DIR=/data                          # Persistent storage path (Fly.io: /data)
```

Note: Several env vars and database filenames still use the legacy `legato` name (e.g., `LEGATO_MODE`, `legato.db`). The product name is **Legate** (or **Legate Studio**); the internal identifiers are kept for backward compatibility.

## GitHub App Setup (multi-tenant)

1. Go to [GitHub Developer Settings > GitHub Apps](https://github.com/settings/apps)
2. Create a new GitHub App with:
   - **Callback URL**: `https://your-domain.com/auth/github/app/callback`
   - **Webhook URL**: `https://your-domain.com/auth/github/webhook`
   - **Permissions**: Repository contents (read/write), Metadata (read)
   - **Subscribe to events**: Installation, Push
3. Generate a private key and download the `.pem` file
4. Set all `GITHUB_APP_*` env vars from above

## Fly.io Deployment

```bash
# First time
fly launch --name legate-studio

# Create persistent volume for SQLite
fly volumes create legate_data --size 10

# Set secrets
fly secrets set \
  FLASK_SECRET_KEY="$(openssl rand -hex 32)" \
  JWT_SECRET_KEY="$(openssl rand -hex 32)" \
  LEGATE_MASTER_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" \
  LEGATO_MODE=multi-tenant \
  GITHUB_APP_ID=... \
  GITHUB_APP_CLIENT_ID=... \
  GITHUB_APP_CLIENT_SECRET=... \
  STRIPE_SECRET_KEY=... \
  STRIPE_WEBHOOK_SECRET=...

# Deploy
fly deploy
```

## MCP Integration

Legate Studio exposes an MCP (Model Context Protocol) server at `/mcp`. It implements OAuth 2.1 with Dynamic Client Registration so Claude Desktop, Claude Code, and other MCP clients can authenticate without manual token management. Protocol version: `2025-06-18`.

Full MCP documentation is available at `/docs/mcp` on any running instance.

### Claude Desktop setup

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "legate": {
      "url": "https://your-domain.com/mcp",
      "transport": "http"
    }
  }
}
```

The OAuth flow handles authentication automatically on first use.

### Available MCP Tools

Legate exposes 43 MCP tools organized into functional groups. All note-related tools accept an optional `library_id` parameter to target a shared library instead of your personal library.

#### Notes — CRUD & Organization

| Tool | Description |
|------|-------------|
| `create_note` | Create a new note with category, content, tags. Supports task metadata (status, due date) and subfolder placement. |
| `get_note` | Retrieve a note by entry_id, file_path, or title (fuzzy match). Fallback chain: entry_id > file_path > title. |
| `get_notes` | Batch-fetch multiple notes by entry_ids, or by category/subfolder with glob pattern filtering. |
| `update_note` | Update a note via full content replacement or precise diff-based edits (old_string/new_string). |
| `append_to_note` | Append content to an existing note — useful for journals and incremental builds. |
| `rename_note` | Rename a note's title, regenerating entry_id and file path with an atomic Git commit. |
| `delete_note` | Delete a note from GitHub and the database. Requires confirmation flag. |
| `move_category` | Move a note to a different category, updating file path and entry_id. |
| `upload_markdown_as_note` | Upload raw markdown (with optional frontmatter) directly as a note. |
| `create_category` | Create a new category folder in the library with display name, description, and color. |
| `list_categories` | List all available note categories. |

#### Search & Discovery

| Tool | Description |
|------|-------------|
| `search_library` | Hybrid semantic + keyword search with confidence bucketing, query expansion, and category filtering. |
| `get_related_notes` | Find semantically similar notes to a given note using embeddings. |
| `get_note_context` | Get a note with its full context: linked notes, semantic neighbors, and related projects. |
| `get_library_stats` | Library statistics — note counts by category, total notes, recent activity. |
| `list_recent_notes` | List the most recently created notes, with optional category filter. |

#### Subfolders

| Tool | Description |
|------|-------------|
| `create_subfolder` | Create a subfolder under a category (creates `.gitkeep` in GitHub). |
| `list_subfolders` | List all subfolders under a category. |
| `list_subfolder_contents` | List all notes within a specific subfolder. |
| `move_to_subfolder` | Move a note to a different subfolder within its category. |
| `rename_subfolder` | Rename a subfolder, moving all contained notes with atomic Git commits. |

#### Tasks

| Tool | Description |
|------|-------------|
| `list_tasks` | List notes marked as tasks, with filters for status, due date range, and category. |
| `update_task_status` | Set or change task status (pending, in_progress, blocked, done) and due date. |

#### Note Linking

| Tool | Description |
|------|-------------|
| `link_notes` | Create a bidirectional relationship between two notes (related, depends_on, blocks, implements, references, contradicts, supports). |

#### Assets

| Tool | Description |
|------|-------------|
| `upload_asset` | Upload an image or file (base64-encoded) to a category's assets folder. Returns a markdown reference. |
| `list_assets` | List assets in a category's assets folder. |
| `get_asset` | Get metadata for a specific asset including its markdown reference. |
| `get_asset_reference` | Get a properly formatted markdown image/link reference for an asset. |
| `delete_asset` | Delete an asset from GitHub and the database. |

#### Motif (Voice/Text Capture)

| Tool | Description |
|------|-------------|
| `process_motif` | Push text/markdown content into the transcript processing pipeline. Returns a job ID. |
| `get_processing_status` | Check the status of an async processing job by job ID. |

#### Shared Libraries

| Tool | Description |
|------|-------------|
| `create_shared_library` | Create a new shared library (provisions a private GitHub repo). Managed tier required. |
| `list_libraries` | List all libraries you have access to — personal + shared (with roles and member counts). |
| `sync_shared_library` | Sync a shared library's database from its GitHub repo. Owner only. |
| `invite_collaborator` | Invite a GitHub user to collaborate on a shared library. |
| `accept_invitation` | Accept a pending invitation to join a shared library. |
| `remove_collaborator` | Remove a collaborator and revoke their access. Owner only. |

#### Draft & Merge Workflow

| Tool | Description |
|------|-------------|
| `create_draft` | Create a draft for a shared library — new_note, edit, or delete. Collaborators must use this instead of direct writes. |
| `submit_draft` | Submit a draft for owner review. |
| `list_drafts` | List drafts with optional status/author filters. Owners see all submitted; collaborators see their own. |
| `review_draft` | Review a draft with side-by-side comparison for edits. |
| `merge_draft` | Merge a submitted draft into the library. Includes conflict detection. Owner only. |
| `reject_draft` | Reject a draft with optional feedback. Owner only. |

#### Chord (Agent Spawning)

| Tool | Description |
|------|-------------|
| `spawn_agent` | Queue a project from 1-5 library notes for GitHub Copilot implementation. Appears in agent queue for approval. |

#### Diagnostics & Sync

| Tool | Description |
|------|-------------|
| `check_connection` | Check MCP connection status, user auth, and GitHub App setup. |
| `verify_sync_state` | Check consistency between database entries and GitHub files. Identifies orphaned or missing entries. |
| `repair_sync_state` | Repair sync mismatches by recreating missing GitHub files from database content. Supports dry-run. |

### MCP Prompts

| Prompt | Description |
|--------|-------------|
| `summarize_notes` | Summarize notes from a category or search results. |

## Architecture

```
src/
├── main.py                          # Gunicorn entry point (creates Flask app)
└── legate_studio/
    ├── main.py                      # CLI entry point (legate-studio command)
    ├── core.py                      # App factory, decorators (login_required, paid_required, beta_gate)
    ├── auth.py                      # GitHub App + OAuth auth, session management
    ├── github_app.py                # GitHub App JWT generation, installation token caching
    ├── admin.py                     # Admin console (user management, feature flags)
    ├── chat.py                      # AI chat with library context (RAG)
    ├── library.py                   # Knowledge entry CRUD + markdown rendering
    ├── categories.py                # Category management
    ├── assets.py                    # File attachment handling (upload, reference, delete)
    ├── shared_libraries.py          # Shared library web UI + JSON API (/shared/)
    ├── agents.py                    # Chord (GitHub Copilot agent) management
    ├── chords.py                    # Chord execution routes
    ├── chord_executor.py            # Project spawning from approved chords
    ├── oauth_server.py              # OAuth 2.1 AS with DCR (for MCP)
    ├── mcp_server.py                # MCP protocol handler (43 tools, JSON-RPC 2.0)
    ├── stripe_billing.py            # Stripe subscription + webhook handling
    ├── crypto.py                    # Fernet encryption for stored secrets
    ├── dashboard.py                 # Dashboard routes (system status, recent notes)
    ├── dropbox.py                   # Transcript upload endpoint
    ├── import_api.py                # Markdown ZIP import (upload, classify, preview, confirm)
    ├── markdown_importer.py         # Markdown import classifier (ZIP processing)
    ├── motif_api.py                 # Motif (voice/text capture) API with job queue
    ├── motif_processor.py           # Background motif processing worker
    ├── memory_api.py                # Memory API — RAG endpoints for pipeline integration
    ├── recovery.py                  # Library recovery CLI (validate, normalize, fix, rebuild)
    ├── worker.py                    # Background worker thread (in-process)
    ├── worker_main.py               # Standalone worker process entry point (Fly.io)
    ├── rag/
    │   ├── database.py              # DB init, migrations, connection management
    │   ├── usage.py                 # Token usage tracking + credit cap enforcement
    │   ├── chat_service.py          # LLM provider abstraction (Anthropic/OpenAI/Gemini)
    │   ├── chat_session_manager.py  # In-memory session cache + periodic flush
    │   ├── embedding_provider.py    # Abstract embedding provider interface + factory
    │   ├── embedding_service.py     # Embedding generation + similarity search
    │   ├── gemini_provider.py       # Gemini text-embedding-004 (768-dim, default)
    │   ├── openai_provider.py       # OpenAI text-embedding-ada-002 (1536-dim)
    │   ├── ollama_provider.py       # Ollama local embeddings
    │   ├── whisper_service.py       # Gemini Flash multimodal transcription
    │   ├── github_service.py        # GitHub API abstraction for library sync
    │   ├── library_sync.py          # GitHub <-> local DB sync
    │   └── context_builder.py       # RAG context assembly for chat
    ├── templates/                   # Jinja2 templates (Catppuccin Mocha dark theme)
    │   ├── base.html, landing.html, marketing_base.html
    │   ├── dashboard.html, dashboard_graph3d.html
    │   ├── library.html, library_entry.html, library_search.html
    │   ├── library_category.html, library_tasks.html, library_projects.html
    │   ├── library_daily.html, library_monthly.html, library_yearly.html
    │   ├── library_graph.html, library_graph3d.html, knowledge_graph_notes.html
    │   ├── chat.html, motif.html, dropbox.html
    │   ├── shared_libraries.html, shared_library_detail.html, shared_library_profile.html
    │   ├── agents.html, chords.html, profile.html, billing.html
    │   ├── import.html, import_preview.html, setup.html
    │   ├── docs_mcp.html                         # MCP documentation page
    │   ├── pricing.html, features.html, faq.html  # Marketing pages
    │   ├── about.html, contact.html               # Company pages
    │   ├── privacy.html, terms.html, security.html # Legal pages
    │   ├── mcp_first_pkm.html, memory_layer_for_ai.html  # SEO content pages
    │   ├── persistent_memory_for_ai.html, pkb_for_ai.html
    │   ├── voice_notes_to_kb.html
    │   ├── published_note.html, report_note.html
    │   ├── admin/                   # Admin console templates
    │   ├── chord/                   # Chord project templates
    │   └── note/                    # Note-related templates
    └── static/
        ├── css/                     # Stylesheets (Catppuccin Mocha theme)
        └── img/                     # Images and icons
```

## Security Notes

- **Sessions**: HttpOnly, SameSite=Lax, Secure (in production)
- **Rate limiting**: 200/day, 50/hour default; admin login: 10/min, 20/hour; MCP: per-user keying via JWT
- **API key storage**: Fernet-encrypted per user, derived from master key
- **OAuth 2.1**: PKCE required, state parameter CSRF protection, redirect URI exact match
- **Stripe webhooks**: Signature verified with `stripe.WebhookSignature.verify_header`
- **SQL**: All queries use parameterized statements (no f-string interpolation in WHERE clauses)
- **Trial expiry**: Enforced at the `before_request` level in multi-tenant mode; beta users exempt
- **WAL checkpointing**: Explicit `PRAGMA wal_checkpoint(RESTART)` after MCP writes for cross-worker visibility
- **Shared library access**: Role-based gating — collaborators cannot write directly, must use draft/merge workflow

## Development Notes

- Use `uv` exclusively — never `pip` or `venv` directly
- SQLite WAL mode is enabled for all databases (concurrent read/write)
- The app runs as a single process on Fly.io with 2 gunicorn workers; background sync uses daemon threads
- `DATA_DIR` (default `/data` on Fly.io) holds SQLite files and persistent keys
- `LEGATO_MODE=single-tenant` bypasses all SaaS gating — useful for local dev
- Recovery CLI: `legate-recovery validate`, `legate-recovery normalize --dry-run`, `legate-recovery full_recovery`
- Embedding provider auto-detection: Gemini (default) > OpenAI > Ollama (local)
