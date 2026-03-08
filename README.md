# Legate Studio

A personal knowledge platform with AI-powered capture, search, and chat. Your second brain — backed by GitHub, served by Flask, secured by design.

## Stack

- **Runtime**: Python 3.12+, [uv](https://github.com/astral-sh/uv) for dependency management
- **Web**: Flask (application factory pattern, blueprints, Jinja2 templates)
- **Database**: SQLite in WAL mode (`legato.db`, `agents.db`, `chat.db`)
- **Auth**: GitHub App OAuth (multi-tenant) or GitHub OAuth App (single-tenant)
- **Billing**: Stripe (subscriptions + one-time top-up credits)
- **Encryption**: Fernet per-user encryption for stored API keys (via `crypto.py`)
- **Rate limiting**: Flask-Limiter (in-memory by default, Redis optional)
- **MCP**: OAuth 2.1 Authorization Server with Dynamic Client Registration
- **Deployment**: Fly.io (persistent `/data` volume for SQLite + secrets)
- **Error tracking**: Sentry (optional, set `SENTRY_DSN`)

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

Top-ups: $2.99 per purchase → +$2.69 in token credits (same 10% platform margin).

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

### Optional

```bash
SENTRY_DSN=https://...@sentry.io/...   # Error tracking
SENTRY_ENVIRONMENT=production
DATA_DIR=/data                          # Persistent storage path (Fly.io: /data)
```

## GitHub App Setup (multi-tenant)

1. Go to [GitHub Developer Settings → GitHub Apps](https://github.com/settings/apps)
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

Legate Studio exposes an MCP (Model Context Protocol) server at `/mcp`. It implements OAuth 2.1 with Dynamic Client Registration so Claude Desktop and other MCP clients can authenticate without manual token management.

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

### Available MCP tools

- `search_library` — Hybrid semantic + keyword search across your notes
- `get_note` — Retrieve a specific note by ID, path, or title
- `create_note` — Create a new note with category, content, tags
- `update_note` — Update an existing note (diff-based or full replace)
- `list_categories` — List all note categories
- `get_library_stats` — Library statistics (counts, recent activity)

## Architecture

```
src/
├── main.py                    # Entry point (gunicorn or direct)
└── legate_studio/
    ├── core.py                # App factory, decorators (login_required, paid_required, beta_gate)
    ├── auth.py                # GitHub App + OAuth auth, session management
    ├── admin.py               # Admin console (user management, feature flags)
    ├── chat.py                # AI chat with library context
    ├── library.py             # Knowledge entry CRUD + markdown rendering
    ├── categories.py          # Category management
    ├── assets.py              # File attachment handling
    ├── agents.py              # Chord (GitHub Copilot agent) management
    ├── chords.py              # Chord execution routes
    ├── oauth_server.py        # OAuth 2.1 AS with DCR (for MCP)
    ├── mcp_server.py          # MCP protocol handler
    ├── stripe_billing.py      # Stripe subscription + webhook handling
    ├── crypto.py              # Fernet encryption for stored secrets
    ├── dashboard.py           # Dashboard routes
    ├── dropbox.py             # Transcript upload endpoint
    ├── import_api.py          # Markdown ZIP import
    ├── motif_api.py           # Motif (voice/text capture) API
    ├── motif_processor.py     # Background motif processing
    ├── worker.py              # Background worker thread
    ├── rag/
    │   ├── database.py        # DB init, migrations, connection management
    │   ├── usage.py           # Token usage tracking + credit cap enforcement
    │   ├── chat_service.py    # LLM provider abstraction (Anthropic/OpenAI/Gemini)
    │   ├── chat_session_manager.py  # In-memory session cache + periodic flush
    │   ├── embedding_service.py     # Embedding generation
    │   ├── library_sync.py    # GitHub → local DB sync
    │   └── context_builder.py # RAG context assembly for chat
    ├── templates/             # Jinja2 templates (Catppuccin Mocha dark theme)
    └── static/                # CSS, JS, images
```

## Security Notes

- **Sessions**: HttpOnly, SameSite=Lax, Secure (in production)
- **Rate limiting**: 200/day, 50/hour default; admin login: 10/min, 20/hour
- **API key storage**: Fernet-encrypted per user, derived from master key
- **OAuth 2.1**: PKCE required, state parameter CSRF protection, redirect URI exact match
- **Stripe webhooks**: Signature verified with `stripe.WebhookSignature.verify_header`
- **SQL**: All queries use parameterized statements (no f-string interpolation in WHERE clauses)
- **Trial expiry**: Enforced at the `before_request` level in multi-tenant mode; beta users exempt

## Development Notes

- Use `uv` exclusively — never `pip` or `venv` directly
- SQLite WAL mode is enabled for all databases (concurrent read/write)
- The app runs as a single process on Fly.io; background sync uses daemon threads
- `DATA_DIR` (default `/data` on Fly.io) holds SQLite files and persistent keys
- `LEGATO_MODE=single-tenant` bypasses all SaaS gating — useful for local dev
