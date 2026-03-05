# Legate Studio

Dashboard and Transcript Dropbox for the Legate Studio system.

## Features

- **Dashboard**: Real-time view of Legate Studio system status, recent transcript jobs, and knowledge artifacts
- **Transcript Dropbox**: Secure upload endpoint for transcripts (mobile-friendly)
- **GitHub OAuth**: Secure authentication with user allowlist
- **Fly.io Deployment**: Automated deployment with GitHub Actions

## Setup

### 1. Create GitHub OAuth App

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click "New OAuth App"
3. Fill in:
   - Application name: `Legate Studio`
   - Homepage URL: `https://legate.studio`
   - Authorization callback URL: `https://legate.studio/auth/github/callback`
4. Note the Client ID and generate a Client Secret

### 2. Configure Environment

```bash
# Run the interactive setup
python utils/flask_keygen.py
```

Or manually create `.env`:

```bash
cp .env.example .env
# Edit .env with your values
```

Required variables:
- `GITHUB_CLIENT_ID` - OAuth App Client ID
- `GITHUB_CLIENT_SECRET` - OAuth App Client Secret
- `GITHUB_ALLOWED_USERS` - Comma-separated usernames (e.g., `bobbyhiddn`)
- `SYSTEM_PAT` - GitHub PAT with repo scope for API access

### 3. Deploy to Fly.io

```bash
# First time: create the app
fly launch

# Deploy with secrets
./utils/fly_deploy.sh
```

Or set secrets manually:
```bash
fly secrets set FLASK_SECRET_KEY="..." GITHUB_CLIENT_ID="..." ...
fly deploy
```

### 4. Configure GitHub Actions (Optional)

Add these secrets to your GitHub repo for automatic deployments:
- `FLY_API_TOKEN` - Fly.io API token

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r src/requirements.txt

# Run locally
cd src
FLASK_ENV=development python main.py
```

Or with Docker:
```bash
docker-compose up --build
```

## Architecture

```
src/
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
└── legate_studio/
    ├── core.py          # Flask app factory
    ├── auth.py          # GitHub OAuth
    ├── dashboard.py     # Dashboard routes
    ├── dropbox.py       # Transcript upload
    ├── templates/       # Jinja2 templates
    └── static/          # CSS, images
```

## Security

- GitHub OAuth with state parameter (CSRF protection)
- User allowlist enforcement
- Secure session cookies (httponly, secure, samesite)
- Rate limiting on all endpoints
- File type validation for uploads

## API Endpoints

- `GET /health` - Health check (no auth)
- `GET /auth/login` - Login page
- `GET /auth/github` - Initiate OAuth
- `GET /auth/github/callback` - OAuth callback
- `GET /auth/logout` - Logout
- `GET /dashboard` - Dashboard (auth required)
- `GET /dashboard/api/status` - Dashboard data JSON
- `GET /dropbox` - Upload form (auth required)
- `POST /dropbox/upload` - Upload transcript (form)
- `POST /dropbox/api/upload` - Upload transcript (JSON API)
