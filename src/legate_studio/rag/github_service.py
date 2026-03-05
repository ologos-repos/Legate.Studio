"""
GitHub Service

Handles GitHub API operations for committing files.
"""

import os
import base64
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def get_file_sha(
    repo: str,
    path: str,
    token: str,
    branch: str = "main",
) -> str:
    """Get current SHA of a file from GitHub.

    Args:
        repo: Repository in "owner/repo" format
        path: File path within repo
        token: GitHub PAT
        branch: Branch name

    Returns:
        SHA string of the file

    Raises:
        requests.RequestException on API errors
    """
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["sha"]


def commit_file(
    repo: str,
    path: str,
    content: str,
    message: str,
    token: str,
    branch: str = "main",
) -> dict:
    """Commit a file to GitHub.

    Args:
        repo: Repository in "owner/repo" format
        path: File path within repo
        content: New file content (plain text)
        message: Commit message
        token: GitHub PAT
        branch: Branch name

    Returns:
        Dict with commit info from GitHub API

    Raises:
        requests.RequestException on API errors
    """
    # Get current SHA
    sha = get_file_sha(repo, path, token, branch)

    # Encode content to base64
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    # Commit via Contents API
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    response = requests.put(
        url,
        json={
            "message": message,
            "content": encoded,
            "sha": sha,
            "branch": branch,
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=15,
    )

    # Log details on error before raising
    if not response.ok:
        try:
            error_body = response.json()
            logger.error(f"GitHub API error committing {path}: {response.status_code} - {error_body}")
        except Exception:
            logger.error(f"GitHub API error committing {path}: {response.status_code} - {response.text[:500]}")
        response.raise_for_status()

    result = response.json()
    logger.info(f"Committed {path} to {repo}: {result['commit']['sha'][:7]}")
    return result


def get_file_content(
    repo: str,
    path: str,
    token: str,
    branch: str = "main",
) -> Optional[str]:
    """Get file content from GitHub.

    Args:
        repo: Repository in "owner/repo" format
        path: File path within repo
        token: GitHub PAT
        branch: Branch name

    Returns:
        File content as string, or None if not found
    """
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=10,
    )

    if response.status_code == 404:
        return None

    response.raise_for_status()
    data = response.json()

    # Decode base64 content
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content


def delete_file(
    repo: str,
    path: str,
    message: str,
    token: str,
    branch: str = "main",
) -> dict:
    """Delete a file from GitHub.

    Args:
        repo: Repository in "owner/repo" format
        path: File path within repo
        message: Commit message
        token: GitHub PAT
        branch: Branch name

    Returns:
        Dict with commit info from GitHub API

    Raises:
        requests.RequestException on API errors
    """
    # Get current SHA
    sha = get_file_sha(repo, path, token, branch)

    # Delete via Contents API
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    response = requests.delete(
        url,
        json={
            "message": message,
            "sha": sha,
            "branch": branch,
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=15,
    )
    response.raise_for_status()

    result = response.json()
    logger.info(f"Deleted {path} from {repo}: {result['commit']['sha'][:7]}")
    return result


def list_folder(
    repo: str,
    path: str,
    token: str,
    branch: str = "main",
) -> list[dict]:
    """List contents of a folder on GitHub.

    Args:
        repo: Repository in "owner/repo" format
        path: Folder path within repo
        token: GitHub PAT
        branch: Branch name

    Returns:
        List of file/folder info dicts with 'name', 'path', 'type', 'sha'

    Raises:
        requests.RequestException on API errors
    """
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=10,
    )

    if response.status_code == 404:
        return []

    response.raise_for_status()
    data = response.json()

    # Contents API returns list for folders, dict for files
    if isinstance(data, dict):
        return [data]
    return data


def move_file(
    repo: str,
    old_path: str,
    new_path: str,
    message: str,
    token: str,
    branch: str = "main",
) -> dict:
    """Move a file from one path to another on GitHub.

    This is done by getting content, creating at new location, deleting old.
    If a file already exists at the destination, it will be overwritten.

    Args:
        repo: Repository in "owner/repo" format
        old_path: Current file path
        new_path: New file path
        message: Commit message
        token: GitHub PAT
        branch: Branch name

    Returns:
        Dict with 'created' and 'deleted' commit info

    Raises:
        requests.RequestException on API errors
    """
    # Get current content
    content = get_file_content(repo, old_path, token, branch)
    if content is None:
        raise ValueError(f"File not found: {old_path}")

    # Check if destination already exists (collision case)
    if file_exists(repo, new_path, token, branch):
        # Use commit_file to update existing file (requires SHA)
        logger.info(f"Destination {new_path} exists, updating instead of creating")
        created = commit_file(repo, new_path, content, message, token, branch)
    else:
        # Create at new location
        created = create_file(repo, new_path, content, message, token, branch)

    # Delete from old location
    deleted = delete_file(repo, old_path, f"Remove {old_path} (moved)", token, branch)

    logger.info(f"Moved {old_path} -> {new_path} in {repo}")
    return {'created': created, 'deleted': deleted}


def create_file(
    repo: str,
    path: str,
    content: str,
    message: str,
    token: str,
    branch: str = "main",
) -> dict:
    """Create a new file on GitHub (fails if exists).

    Args:
        repo: Repository in "owner/repo" format
        path: File path within repo
        content: File content (plain text)
        message: Commit message
        token: GitHub PAT
        branch: Branch name

    Returns:
        Dict with commit info from GitHub API

    Raises:
        requests.RequestException on API errors
    """
    # Encode content to base64
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    # Create via Contents API (no sha = create new)
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    response = requests.put(
        url,
        json={
            "message": message,
            "content": encoded,
            "branch": branch,
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=15,
    )

    # Log details on error before raising
    if not response.ok:
        try:
            error_body = response.json()
            logger.error(f"GitHub API error creating {path}: {response.status_code} - {error_body}")
        except Exception:
            logger.error(f"GitHub API error creating {path}: {response.status_code} - {response.text[:500]}")
        response.raise_for_status()

    result = response.json()
    logger.info(f"Created {path} in {repo}: {result['commit']['sha'][:7]}")
    return result


def create_binary_file(
    repo: str,
    path: str,
    content: bytes,
    message: str,
    token: str,
    branch: str = "main",
) -> dict:
    """Create a new binary file on GitHub (fails if exists).

    Args:
        repo: Repository in "owner/repo" format
        path: File path within repo
        content: File content as bytes (binary data)
        message: Commit message
        token: GitHub PAT
        branch: Branch name

    Returns:
        Dict with commit info from GitHub API

    Raises:
        requests.RequestException on API errors
    """
    # Encode binary content to base64
    encoded = base64.b64encode(content).decode("utf-8")

    # Create via Contents API (no sha = create new)
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    response = requests.put(
        url,
        json={
            "message": message,
            "content": encoded,
            "branch": branch,
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=30,  # Longer timeout for binary files
    )
    response.raise_for_status()

    result = response.json()
    logger.info(f"Created binary file {path} in {repo}: {result['commit']['sha'][:7]}")
    return result


def get_binary_file(
    repo: str,
    path: str,
    token: str,
    branch: str = "main",
) -> Optional[bytes]:
    """Get binary file content from GitHub.

    Args:
        repo: Repository in "owner/repo" format
        path: File path within repo
        token: GitHub PAT
        branch: Branch name

    Returns:
        File content as bytes, or None if not found
    """
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=30,  # Longer timeout for binary files
    )

    if response.status_code == 404:
        return None

    response.raise_for_status()
    data = response.json()

    # Decode base64 content to bytes
    content = base64.b64decode(data["content"])
    return content


def update_binary_file(
    repo: str,
    path: str,
    content: bytes,
    message: str,
    token: str,
    branch: str = "main",
) -> dict:
    """Update an existing binary file on GitHub.

    Args:
        repo: Repository in "owner/repo" format
        path: File path within repo
        content: New file content as bytes
        message: Commit message
        token: GitHub PAT
        branch: Branch name

    Returns:
        Dict with commit info from GitHub API

    Raises:
        requests.RequestException on API errors
    """
    # Get current SHA
    sha = get_file_sha(repo, path, token, branch)

    # Encode binary content to base64
    encoded = base64.b64encode(content).decode("utf-8")

    # Commit via Contents API
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    response = requests.put(
        url,
        json={
            "message": message,
            "content": encoded,
            "sha": sha,
            "branch": branch,
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=30,
    )
    response.raise_for_status()

    result = response.json()
    logger.info(f"Updated binary file {path} in {repo}: {result['commit']['sha'][:7]}")
    return result


def file_exists(
    repo: str,
    path: str,
    token: str,
    branch: str = "main",
) -> bool:
    """Check if a file exists on GitHub.

    Args:
        repo: Repository in "owner/repo" format
        path: File path within repo
        token: GitHub PAT
        branch: Branch name

    Returns:
        True if file exists, False otherwise
    """
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=10,
    )

    return response.status_code == 200
