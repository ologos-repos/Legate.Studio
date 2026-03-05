"""
Chord Executor - Spawns projects from approved chords.

This replaces Legato.Conduct by handling project creation directly in Pit.
Uses embedded templates and GitHub API to create repos, push files, and
create issues for Copilot.

Supports both single-tenant and multi-tenant modes with user installation tokens.
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Template directory (embedded in Pit)
TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class ChordSpec:
    """Specification for spawning a chord project."""

    name: str
    project_type: str  # 'chord' or 'note'
    title: str
    description: str
    domain_tags: list[str]
    key_phrases: list[str]
    source_entry_id: Optional[str] = None
    tasker_body: Optional[str] = None
    source_note_content: Optional[str] = None  # Full markdown content of source note
    source_note_title: Optional[str] = None    # Note title for filename

    def get_repo_name(self, org: str) -> str:
        """Get the full repository name."""
        suffix = "Chord" if self.project_type == "chord" else "Note"
        return f"{org}/{self.name}.{suffix}"


class ChordExecutor:
    """
    Executes chord spawning - creates repos, pushes templates, creates issues.

    For single-tenant: Uses SYSTEM_PAT from environment
    For multi-tenant: Uses installation token from user context
    """

    def __init__(self, token: str, org: str):
        """
        Initialize the executor.

        Args:
            token: GitHub token (PAT or installation token)
            org: GitHub organization or username for created repos
        """
        self.token = token
        self.org = org
        self.api_base = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        logger.info(f"ChordExecutor initialized: org={org}, token_len={len(token) if token else 0}, token_prefix={token[:10] if token and len(token) > 10 else 'N/A'}...")

    def spawn(self, spec: ChordSpec, assign_copilot: bool = True) -> dict:
        """
        Spawn a project from a chord specification.

        Args:
            spec: The chord specification
            assign_copilot: Whether to assign the initial issue to Copilot

        Returns:
            Dict with repo_url, issue_url, success status
        """
        repo_name = spec.get_repo_name(self.org)
        repo_short = repo_name.split("/")[1]

        logger.info(f"Spawning chord: {repo_name}")

        try:
            # Step 1: Create repository (with legato-chord topic)
            repo_url, repo_id = self._create_repo(repo_short, spec.title)
            logger.info(f"Created repo: {repo_url} (id: {repo_id})")

            # Step 2: Push template files
            self._push_templates(repo_name, spec)
            logger.info(f"Pushed templates to {repo_name}")

            # Step 2.5: Push source note if available (for context)
            self._push_source_note(repo_name, spec)

            # Step 3: Create initial issue
            issue_url, issue_number = self._create_issue(repo_name, spec)
            logger.info(f"Created issue: {issue_url}")

            # Step 4: Assign to Copilot (optional)
            assigned = False
            if assign_copilot:
                assigned = self._assign_to_copilot(repo_name, issue_number)
                if assigned:
                    logger.info(f"Assigned issue to Copilot")
                else:
                    logger.warning(f"Could not assign to Copilot (may not be enabled)")

            return {
                "success": True,
                "repo_name": repo_name,
                "repo_id": repo_id,
                "repo_url": repo_url,
                "issue_url": issue_url,
                "issue_number": issue_number,
                "assigned_copilot": assigned,
            }

        except Exception as e:
            logger.error(f"Failed to spawn chord {spec.name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "repo_name": repo_name,
            }

    def _create_repo(self, name: str, description: str) -> tuple[str, int]:
        """Create a new GitHub repository.

        Returns:
            Tuple of (html_url, repo_id)
        """
        # Check if repo already exists
        check_url = f"{self.api_base}/repos/{self.org}/{name}"
        check_resp = requests.get(check_url, headers=self.headers, timeout=10)

        if check_resp.status_code == 200:
            logger.info(f"Repo {self.org}/{name} already exists")
            repo_data = check_resp.json()
            return repo_data["html_url"], repo_data["id"]

        # Try to create as org repo first, fall back to user repo
        create_url = f"{self.api_base}/orgs/{self.org}/repos"
        payload = {
            "name": name,
            "description": f"Legate Studio project: {description}",
            "private": False,
            "auto_init": True,  # Creates initial commit with README
        }

        logger.info(f"Creating repo at {create_url} for org={self.org}")
        resp = requests.post(create_url, headers=self.headers, json=payload, timeout=30)

        if resp.status_code == 404:
            # Not an org, try as user
            create_url = f"{self.api_base}/user/repos"
            logger.info(f"Org not found, trying user repo at {create_url}")
            resp = requests.post(create_url, headers=self.headers, json=payload, timeout=30)

        if resp.status_code not in (200, 201):
            # Log token info for debugging (prefix only, not full token)
            token_info = f"len={len(self.token)}, prefix={self.token[:7]}..." if self.token else "None"
            logger.error(f"Repo creation failed: status={resp.status_code}, token_info={token_info}")
            raise RuntimeError(f"Failed to create repo: {resp.status_code} - {resp.text}")

        repo_data = resp.json()
        repo_id = repo_data["id"]
        repo_url = repo_data["html_url"]

        # Add legato-chord topic to the repo
        self._add_topic(f"{self.org}/{name}", "legato-chord")

        return repo_url, repo_id

    def _add_topic(self, repo_name: str, topic: str):
        """Add a topic to a repository."""
        # Get current topics
        url = f"{self.api_base}/repos/{repo_name}/topics"
        resp = requests.get(url, headers=self.headers, timeout=10)

        current_topics = []
        if resp.ok:
            current_topics = resp.json().get("names", [])

        if topic not in current_topics:
            current_topics.append(topic)

            # Update topics
            resp = requests.put(
                url,
                headers=self.headers,
                json={"names": current_topics},
                timeout=10
            )

            if resp.ok:
                logger.info(f"Added topic '{topic}' to {repo_name}")
            else:
                logger.warning(f"Failed to add topic to {repo_name}: {resp.status_code}")

    def _push_templates(self, repo_name: str, spec: ChordSpec):
        """Push template files to the repository."""
        template_dir = TEMPLATES_DIR / spec.project_type

        if not template_dir.exists():
            raise RuntimeError(f"Template not found: {spec.project_type}")

        # Collect all template files
        for file_path in template_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(template_dir)
                content = file_path.read_text(encoding="utf-8")

                # Substitute template variables
                content = self._substitute_variables(content, spec)

                # Push to repo
                self._create_or_update_file(
                    repo_name,
                    str(rel_path),
                    content,
                    f"Add {rel_path}"
                )

    def _substitute_variables(self, content: str, spec: ChordSpec) -> str:
        """Replace {{ variable }} placeholders in template content."""
        replacements = {
            "project_name": spec.name,
            "project_description": spec.description,
            "domain_tags": ", ".join(spec.domain_tags) if spec.domain_tags else "N/A",
            "key_phrases": ", ".join(spec.key_phrases) if spec.key_phrases else "N/A",
            "source_entry_id": spec.source_entry_id or "N/A",
            "created_date": datetime.utcnow().isoformat() + "Z",
            "org": self.org,
        }

        for key, value in replacements.items():
            # Match {{ key }} with optional whitespace
            pattern = r"\{\{\s*" + re.escape(key) + r"\s*\}\}"
            content = re.sub(pattern, value, content)

        return content

    def _create_or_update_file(self, repo_name: str, path: str, content: str, message: str):
        """Create or update a file in the repository."""
        import base64

        url = f"{self.api_base}/repos/{repo_name}/contents/{path}"

        # Check if file exists (to get SHA for update)
        existing_sha = None
        check_resp = requests.get(url, headers=self.headers, timeout=10)
        if check_resp.status_code == 200:
            existing_sha = check_resp.json().get("sha")

        payload = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        }

        if existing_sha:
            payload["sha"] = existing_sha

        resp = requests.put(url, headers=self.headers, json=payload, timeout=30)

        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Failed to create file {path}: {resp.status_code} - {resp.text}")

    def _create_issue(self, repo_name: str, spec: ChordSpec) -> tuple[str, int]:
        """Create the initial issue for the project."""
        url = f"{self.api_base}/repos/{repo_name}/issues"

        # Generate tasker body if not provided
        body = spec.tasker_body
        if not body:
            body = self._generate_tasker_body(spec)

        payload = {
            "title": "Initial Implementation",
            "body": body,
            "labels": ["copilot"],
        }

        resp = requests.post(url, headers=self.headers, json=payload, timeout=30)

        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Failed to create issue: {resp.status_code} - {resp.text}")

        data = resp.json()
        return data["html_url"], data["number"]

    def _generate_tasker_body(self, spec: ChordSpec) -> str:
        """Generate a tasker body for the initial issue."""
        criteria = spec.key_phrases[:5] if spec.key_phrases else ["Implement core functionality"]
        criteria_list = "\n".join(f"- [ ] {c}" for c in criteria)

        return f"""## Tasker: {spec.title}

### Context

This project was spawned from a Legate Studio knowledge entry.

**Source**: {spec.source_entry_id or 'N/A'}

### Objective

{spec.description}

### Acceptance Criteria

{criteria_list}

### Constraints

- Follow patterns in `copilot-instructions.md`
- Reference `SIGNAL.md` for project intent
- Write tests for new functionality
- Keep PRs focused and reviewable

### References

- [SIGNAL.md](./SIGNAL.md) - Project intent
- [docs/architecture.md](./docs/architecture.md) - System design

---
*Generated by Legate Studio*
"""

    def _assign_to_copilot(self, repo_name: str, issue_number: int) -> bool:
        """Assign an issue to GitHub Copilot coding agent."""
        owner, repo = repo_name.split("/")

        try:
            # Get issue node ID via GraphQL
            query = """
            query($owner: String!, $repo: String!, $number: Int!) {
                repository(owner: $owner, name: $repo) {
                    issue(number: $number) {
                        id
                    }
                }
            }
            """

            resp = requests.post(
                f"{self.api_base}/graphql",
                headers=self.headers,
                json={
                    "query": query,
                    "variables": {"owner": owner, "repo": repo, "number": issue_number}
                },
                timeout=30
            )

            if resp.status_code != 200:
                logger.warning(f"GraphQL query failed: {resp.status_code}")
                return False

            data = resp.json()
            issue_id = data.get("data", {}).get("repository", {}).get("issue", {}).get("id")

            if not issue_id:
                logger.warning("Could not get issue ID")
                return False

            # Find Copilot in suggested actors
            actors_query = """
            query($owner: String!, $repo: String!) {
                repository(owner: $owner, name: $repo) {
                    suggestedActors(capabilities: [CAN_BE_ASSIGNED], first: 100) {
                        nodes {
                            login
                            ... on Bot { id }
                            ... on User { id }
                        }
                    }
                }
            }
            """

            resp = requests.post(
                f"{self.api_base}/graphql",
                headers=self.headers,
                json={
                    "query": actors_query,
                    "variables": {"owner": owner, "repo": repo}
                },
                timeout=30
            )

            if resp.status_code != 200:
                return False

            data = resp.json()
            nodes = data.get("data", {}).get("repository", {}).get("suggestedActors", {}).get("nodes", [])

            copilot_id = None
            for node in nodes:
                if node.get("login") == "copilot-swe-agent":
                    copilot_id = node.get("id")
                    break

            if not copilot_id:
                logger.info("Copilot not available as assignee")
                return False

            # Assign to Copilot
            assign_mutation = """
            mutation($issueId: ID!, $actorIds: [ID!]!) {
                replaceActorsForAssignable(input: {assignableId: $issueId, actorIds: $actorIds}) {
                    assignable {
                        ... on Issue {
                            assignees(first: 5) {
                                nodes { login }
                            }
                        }
                    }
                }
            }
            """

            resp = requests.post(
                f"{self.api_base}/graphql",
                headers=self.headers,
                json={
                    "query": assign_mutation,
                    "variables": {"issueId": issue_id, "actorIds": [copilot_id]}
                },
                timeout=30
            )

            if resp.status_code == 200:
                data = resp.json()
                assignees = data.get("data", {}).get("replaceActorsForAssignable", {}).get("assignable", {}).get("assignees", {}).get("nodes", [])
                return any(a.get("login") == "copilot-swe-agent" for a in assignees)

            return False

        except Exception as e:
            logger.warning(f"Failed to assign to Copilot: {e}")
            return False

    def _push_source_note(self, repo_name: str, spec: ChordSpec):
        """Push the source note to notes/ directory if available.

        This provides the spawned chord with direct context from the
        Library entry that triggered it.

        Args:
            repo_name: Full repository name (org/repo)
            spec: ChordSpec containing source note content and title
        """
        if not spec.source_note_content or not spec.source_note_title:
            return

        # Sanitize filename - remove special chars, replace spaces with dashes
        safe_title = re.sub(r'[^\w\s-]', '', spec.source_note_title)
        safe_title = re.sub(r'\s+', '-', safe_title).strip('-')
        if not safe_title:
            safe_title = "source-note"

        filename = f"notes/{safe_title}.md"

        try:
            self._create_or_update_file(
                repo_name,
                filename,
                spec.source_note_content,
                f"Add source note: {spec.source_note_title}"
            )
            logger.info(f"Pushed source note to {repo_name}/{filename}")
        except Exception as e:
            logger.warning(f"Failed to push source note to {repo_name}: {e}")


def get_executor(user_id: Optional[str] = None) -> ChordExecutor:
    """
    Get a ChordExecutor for the current context.

    For single-tenant: Uses SYSTEM_PAT and configured org
    For multi-tenant: Uses user's installation token and their org

    Args:
        user_id: Optional user ID for multi-tenant mode

    Returns:
        Configured ChordExecutor instance
    """
    from flask import current_app

    mode = current_app.config.get("LEGATO_MODE", "single-tenant")

    if mode == "multi-tenant" and user_id:
        # Multi-tenant: Get user's OAuth token for repo creation
        # OAuth tokens have public_repo scope; installation tokens can't create repos
        from .auth import _get_user_oauth_token
        from .rag.database import init_db

        db = init_db()  # Shared DB for user lookups

        # Get user's org from their configured repos
        row = db.execute(
            "SELECT repo_full_name FROM user_repos WHERE user_id = ? LIMIT 1",
            (user_id,)
        ).fetchone()

        if row:
            org = row["repo_full_name"].split("/")[0]
        else:
            # Fall back to their GitHub login
            user_row = db.execute(
                "SELECT github_login FROM users WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            org = user_row["github_login"] if user_row else "unknown"

        # Use OAuth token for repo creation (has public_repo scope)
        token = _get_user_oauth_token(user_id)

        if not token:
            raise RuntimeError("No OAuth token available for user - please re-authenticate")

        # Validate token before creating executor
        import requests
        test_resp = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            timeout=10
        )
        if test_resp.status_code != 200:
            logger.error(f"Token validation failed: {test_resp.status_code} - {test_resp.text[:200]}")
            raise RuntimeError(f"OAuth token is invalid (status {test_resp.status_code}). Please re-authenticate.")
        else:
            user_data = test_resp.json()
            logger.info(f"Token validated for GitHub user: {user_data.get('login')}")

        return ChordExecutor(token, org)

    else:
        # Single-tenant: Use SYSTEM_PAT
        token = os.environ.get("SYSTEM_PAT")
        org = current_app.config.get("LEGATO_ORG", "bobbyhiddn")

        if not token:
            raise RuntimeError("SYSTEM_PAT not configured")

        return ChordExecutor(token, org)


def ensure_library_exists(token: str, org: str) -> dict:
    """
    Ensure a Legate.Library repository exists for the user.

    Creates the repo if it doesn't exist, with initial structure.

    Args:
        token: GitHub token (PAT or installation token)
        org: GitHub organization or username

    Returns:
        Dict with repo_url and created status
    """
    import base64

    api_base = "https://api.github.com"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    repo_name = f"{org}/Legate.Library"

    # Check if repo exists
    check_url = f"{api_base}/repos/{repo_name}"
    check_resp = requests.get(check_url, headers=headers, timeout=10)

    if check_resp.status_code == 200:
        logger.info(f"Library repo {repo_name} already exists")
        return {
            "success": True,
            "repo_url": check_resp.json()["html_url"],
            "created": False,
        }

    # Create the repository
    logger.info(f"Creating Library repo: {repo_name}")

    # Try org first, fall back to user
    create_url = f"{api_base}/orgs/{org}/repos"
    payload = {
        "name": "Legate.Library",
        "description": "Legate Studio Knowledge Store - Personal knowledge artifacts",
        "private": False,
        "auto_init": False,
    }

    resp = requests.post(create_url, headers=headers, json=payload, timeout=30)

    if resp.status_code == 404:
        create_url = f"{api_base}/user/repos"
        resp = requests.post(create_url, headers=headers, json=payload, timeout=30)

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create Library repo: {resp.status_code} - {resp.text}")

    repo_url = resp.json()["html_url"]

    # Create initial README
    readme_content = f"""# Legate.Library

> Personal Knowledge Store - Managed by [Legate Studio](https://github.com/{org}/Legato.Pit)

This repository stores your knowledge artifacts created through Legate Studio.

## Structure

```
├── epiphany/       # Major insights and breakthrough ideas
├── concept/        # Technical concepts and definitions
├── reflection/     # Personal thoughts and observations
├── glimmer/        # Quick ideas, seeds for future exploration
├── reminder/       # Action items and follow-ups
├── worklog/        # Session summaries and progress notes
├── tech-thought/   # Technical musings
├── research-topic/ # Research areas
└── index.json      # Quick lookup index
```

## Usage

This repository is managed by Legate Studio. Knowledge artifacts are created automatically
when you interact with Claude through the Legate Studio MCP tools.

**Do not manually edit files** - changes may be overwritten during sync.

---
*Managed by Legate Studio*
"""

    readme_url = f"{api_base}/repos/{repo_name}/contents/README.md"
    requests.put(
        readme_url,
        headers=headers,
        json={
            "message": "Initialize Legate.Library",
            "content": base64.b64encode(readme_content.encode()).decode(),
        },
        timeout=30,
    )

    # Create category directories with .gitkeep
    categories = [
        "epiphany", "concept", "reflection", "glimmer", "reminder",
        "worklog", "tech-thought", "research-topic", "theology",
        "writing", "agent-thought", "article-idea"
    ]

    for category in categories:
        gitkeep_url = f"{api_base}/repos/{repo_name}/contents/{category}/.gitkeep"
        requests.put(
            gitkeep_url,
            headers=headers,
            json={
                "message": f"Create {category} directory",
                "content": base64.b64encode(f"# {category.replace('-', ' ').title()}\n".encode()).decode(),
            },
            timeout=30,
        )

    # Create empty index.json
    index_url = f"{api_base}/repos/{repo_name}/contents/index.json"
    requests.put(
        index_url,
        headers=headers,
        json={
            "message": "Initialize index",
            "content": base64.b64encode(b"{}").decode(),
        },
        timeout=30,
    )

    logger.info(f"Created and initialized Library repo: {repo_name}")

    return {
        "success": True,
        "repo_url": repo_url,
        "created": True,
    }


def spawn_chord(
    name: str,
    project_type: str,
    title: str,
    description: str,
    domain_tags: list[str] = None,
    key_phrases: list[str] = None,
    source_entry_id: str = None,
    tasker_body: str = None,
    user_id: str = None,
    assign_copilot: bool = True,
    source_note_content: str = None,
    source_note_title: str = None,
) -> dict:
    """
    Convenience function to spawn a chord.

    Args:
        name: Project name (will become repo name)
        project_type: 'chord' or 'note'
        title: Human-readable title
        description: Project description
        domain_tags: List of domain tags
        key_phrases: List of key phrases
        source_entry_id: ID of the source knowledge entry
        tasker_body: Custom tasker body (auto-generated if not provided)
        user_id: User ID for multi-tenant mode
        assign_copilot: Whether to assign to Copilot
        source_note_content: Full markdown content of the source Library note
        source_note_title: Title of the source note (used for filename)

    Returns:
        Result dict with success status, URLs, etc.
    """
    spec = ChordSpec(
        name=name,
        project_type=project_type,
        title=title,
        description=description,
        domain_tags=domain_tags or [],
        key_phrases=key_phrases or [],
        source_entry_id=source_entry_id,
        tasker_body=tasker_body,
        source_note_content=source_note_content,
        source_note_title=source_note_title,
    )

    executor = get_executor(user_id)
    result = executor.spawn(spec, assign_copilot=assign_copilot)

    # OAuth token is used for all chord operations - no need to add to GitHub App installation
    # The user owns the repo, so OAuth gives us full access for management

    return result
