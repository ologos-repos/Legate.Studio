"""
Library Recovery and Normalization Module

Provides tools for recovering from data corruption without accessing user content.
Designed for blind recovery in multi-tenant environments.

Operations:
- validate: Check library integrity, report issues
- normalize_ids: Fix ID format inconsistencies
- fix_frontmatter: Remove duplicate frontmatter blocks
- merge_categories: Consolidate duplicate category folders
- rebuild_hashes: Recompute content_hash for all entries
- full_recovery: Run all recovery operations

Usage:
    python -m legate_studio.recovery validate
    python -m legate_studio.recovery normalize --dry-run
    python -m legate_studio.recovery full_recovery
"""

import os
import re
import json
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)


def get_default_branch(repo: str, headers: dict) -> str:
    """Get the default branch for a repository.

    Args:
        repo: Full repo name (owner/repo)
        headers: GitHub API headers with auth

    Returns:
        Default branch name (e.g., 'main' or 'master')
    """
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{repo}",
            headers=headers,
            timeout=10
        )
        if resp.ok:
            return resp.json().get('default_branch', 'main')
    except Exception as e:
        logger.warning(f"Failed to get default branch for {repo}: {e}")
    return 'main'  # fallback


# ============ Data Classes ============

@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    issue_type: str
    file_path: str
    details: str
    severity: str = "warning"  # warning, error, critical


@dataclass
class ValidationReport:
    """Report of all validation issues found."""
    tenant_id: str
    timestamp: str
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def add_issue(self, issue_type: str, file_path: str, details: str, severity: str = "warning"):
        self.issues.append(ValidationIssue(issue_type, file_path, details, severity))

    def to_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp,
            "issue_count": len(self.issues),
            "issues_by_type": self._group_by_type(),
            "issues_by_severity": self._group_by_severity(),
            "stats": self.stats,
            "issues": [
                {"type": i.issue_type, "path": i.file_path, "details": i.details, "severity": i.severity}
                for i in self.issues
            ]
        }

    def _group_by_type(self) -> dict:
        groups = {}
        for issue in self.issues:
            groups[issue.issue_type] = groups.get(issue.issue_type, 0) + 1
        return groups

    def _group_by_severity(self) -> dict:
        groups = {}
        for issue in self.issues:
            groups[issue.severity] = groups.get(issue.severity, 0) + 1
        return groups


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    operation: str
    success: bool
    files_processed: int = 0
    files_modified: int = 0
    errors: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


# ============ Helper Functions ============

def compute_content_hash(content: str) -> str:
    """Compute stable hash of content."""
    normalized = content.strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def generate_slug(title: str) -> str:
    """Generate URL-safe slug from title."""
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower())[:50].strip('-')
    return slug or 'untitled'


def generate_canonical_id(category: str, title: str, tenant_id: str = None) -> str:
    """Generate canonical entry ID.

    Format: [tenant.]library.{category}.{slug}
    """
    slug = generate_slug(title)
    if tenant_id:
        return f"{tenant_id}.library.{category}.{slug}"
    return f"library.{category}.{slug}"


def parse_all_frontmatter(content: str) -> Tuple[List[Dict], str]:
    """Parse ALL frontmatter blocks from content (handles corruption).

    Returns:
        Tuple of (list of frontmatter dicts, body content after all frontmatter)
    """
    frontmatters = []
    remaining = content

    while remaining.startswith('---'):
        parts = remaining.split('---', 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            fm_dict = {}
            for line in fm_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    value = value.strip().strip('"\'')
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    fm_dict[key.strip()] = value
            frontmatters.append(fm_dict)
            remaining = parts[2].strip()

            # Check if there's another frontmatter block
            if not remaining.startswith('---'):
                break
        else:
            break

    return frontmatters, remaining


def merge_frontmatter(frontmatters: List[Dict]) -> Dict:
    """Merge multiple frontmatter blocks, preferring first block's values."""
    if not frontmatters:
        return {}

    # Start with first (intended) frontmatter
    merged = dict(frontmatters[0])

    # Add any fields from other blocks that aren't in first
    for fm in frontmatters[1:]:
        for key, value in fm.items():
            if key not in merged:
                merged[key] = value

    return merged


def normalize_category(category: str) -> str:
    """Normalize category name to singular form."""
    # Map common typos and plurals to canonical form
    category_map = {
        # Typos
        'epiphanys': 'epiphany',
        'epiphanies': 'epiphany',
        'theologys': 'theology',
        'theologies': 'theology',
        'tech-thoughtss': 'tech-thought',
        'tech-thoughts': 'tech-thought',
        'research-topicss': 'research-topic',
        'research-topics': 'research-topic',
        # Plurals to singular
        'concepts': 'concept',
        'reflections': 'reflection',
        'glimmers': 'glimmer',
        'reminders': 'reminder',
        'worklogs': 'worklog',
    }
    return category_map.get(category.lower(), category.lower())


# ============ Validation ============

def validate_library(
    repo: str = "bobbyhiddn/Legate.Library",
    token: Optional[str] = None,
    tenant_id: str = "default"
) -> ValidationReport:
    """Validate library integrity without reading content details.

    Checks:
    - ID format consistency
    - Double frontmatter
    - Category/folder mismatches
    - Missing content_hash
    - Orphan entries (no source transcript)
    """
    token = token or os.environ.get('SYSTEM_PAT')
    report = ValidationReport(
        tenant_id=tenant_id,
        timestamp=datetime.utcnow().isoformat() + 'Z'
    )

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
    }

    try:
        # Get repository tree
        branch = get_default_branch(repo, headers)
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        response = requests.get(tree_url, headers=headers, timeout=30)
        response.raise_for_status()
        tree_data = response.json()

        # Filter for markdown files
        md_files = [
            item for item in tree_data.get('tree', [])
            if item['type'] == 'blob' and item['path'].endswith('.md')
            and not item['path'].startswith('.')
            and item['path'] != 'README.md'
        ]

        report.stats['total_files'] = len(md_files)
        report.stats['categories_found'] = set()

        for item in md_files:
            path = item['path']

            # Extract category from path
            parts = Path(path).parts
            if parts:
                folder_category = parts[0]
                report.stats['categories_found'].add(folder_category)

                # Check for typo categories
                normalized = normalize_category(folder_category)
                if normalized != folder_category.lower():
                    report.add_issue(
                        "category_typo",
                        path,
                        f"Folder '{folder_category}' should be '{normalized}'",
                        "warning"
                    )

            # Fetch file content to check frontmatter
            try:
                content_url = f"https://api.github.com/repos/{repo}/contents/{path}"
                content_response = requests.get(content_url, headers=headers, timeout=30)
                content_response.raise_for_status()

                import base64
                file_data = content_response.json()
                content = base64.b64decode(file_data['content']).decode('utf-8')

                # Check for double frontmatter
                frontmatters, body = parse_all_frontmatter(content)

                if len(frontmatters) > 1:
                    report.add_issue(
                        "double_frontmatter",
                        path,
                        f"Found {len(frontmatters)} frontmatter blocks",
                        "critical"
                    )

                if frontmatters:
                    fm = frontmatters[0]

                    # Check ID format
                    entry_id = fm.get('id', '')
                    if entry_id:
                        if entry_id.startswith('kb-'):
                            report.add_issue(
                                "legacy_id_format",
                                path,
                                f"Uses legacy kb-hash format: {entry_id}",
                                "warning"
                            )
                        elif not entry_id.startswith('library.') and not '.' in entry_id:
                            report.add_issue(
                                "invalid_id_format",
                                path,
                                f"Invalid ID format: {entry_id}",
                                "error"
                            )
                    else:
                        report.add_issue(
                            "missing_id",
                            path,
                            "No ID in frontmatter",
                            "warning"
                        )

                    # Check for content_hash
                    if not fm.get('content_hash'):
                        report.add_issue(
                            "missing_content_hash",
                            path,
                            "No content_hash in frontmatter",
                            "warning"
                        )

                    # Check category matches folder
                    fm_category = fm.get('category', '')
                    if fm_category and parts:
                        norm_fm = normalize_category(fm_category)
                        norm_folder = normalize_category(parts[0])
                        if norm_fm != norm_folder:
                            report.add_issue(
                                "category_mismatch",
                                path,
                                f"Frontmatter category '{fm_category}' vs folder '{parts[0]}'",
                                "warning"
                            )
                else:
                    report.add_issue(
                        "no_frontmatter",
                        path,
                        "File has no frontmatter",
                        "error"
                    )

            except Exception as e:
                report.add_issue(
                    "fetch_error",
                    path,
                    str(e),
                    "error"
                )

        report.stats['categories_found'] = list(report.stats['categories_found'])

    except Exception as e:
        report.add_issue("api_error", "", str(e), "critical")

    return report


# ============ Recovery Operations ============

def fix_double_frontmatter(
    repo: str = "bobbyhiddn/Legate.Library",
    token: Optional[str] = None,
    dry_run: bool = True
) -> RecoveryResult:
    """Fix files with double frontmatter blocks.

    Strategy: Keep first frontmatter block, merge unique fields from others,
    rewrite file with single clean frontmatter.
    """
    token = token or os.environ.get('SYSTEM_PAT')
    result = RecoveryResult(operation="fix_double_frontmatter", success=True)

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
    }

    try:
        # Get all markdown files
        branch = get_default_branch(repo, headers)
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        response = requests.get(tree_url, headers=headers, timeout=30)
        response.raise_for_status()

        md_files = [
            item for item in response.json().get('tree', [])
            if item['type'] == 'blob' and item['path'].endswith('.md')
            and not item['path'].startswith('.') and item['path'] != 'README.md'
        ]

        result.files_processed = len(md_files)

        for item in md_files:
            path = item['path']

            try:
                # Fetch content
                content_url = f"https://api.github.com/repos/{repo}/contents/{path}"
                content_response = requests.get(content_url, headers=headers, timeout=30)
                content_response.raise_for_status()

                import base64
                file_data = content_response.json()
                content = base64.b64decode(file_data['content']).decode('utf-8')
                sha = file_data['sha']

                # Check for double frontmatter
                frontmatters, body = parse_all_frontmatter(content)

                if len(frontmatters) > 1:
                    # Merge frontmatter
                    merged = merge_frontmatter(frontmatters)

                    # Ensure content_hash
                    if 'content_hash' not in merged:
                        merged['content_hash'] = compute_content_hash(body)

                    # Build clean frontmatter
                    fm_lines = ['---']
                    # Preserve order: id, title, category first
                    priority_keys = ['id', 'title', 'category', 'created', 'content_hash', 'source_transcript', 'source']
                    for key in priority_keys:
                        if key in merged:
                            value = merged[key]
                            if isinstance(value, str) and key == 'title':
                                fm_lines.append(f'{key}: "{value}"')
                            elif isinstance(value, list):
                                fm_lines.append(f'{key}: {json.dumps(value)}')
                            else:
                                fm_lines.append(f'{key}: {value}')
                    # Add remaining keys
                    for key, value in merged.items():
                        if key not in priority_keys:
                            if isinstance(value, str) and ' ' in value:
                                fm_lines.append(f'{key}: "{value}"')
                            elif isinstance(value, list):
                                fm_lines.append(f'{key}: {json.dumps(value)}')
                            else:
                                fm_lines.append(f'{key}: {value}')
                    fm_lines.append('---')
                    fm_lines.append('')

                    new_content = '\n'.join(fm_lines) + body

                    if not dry_run:
                        # Commit fix
                        update_response = requests.put(
                            content_url,
                            headers=headers,
                            json={
                                'message': f'[recovery] Fix double frontmatter: {path}',
                                'content': base64.b64encode(new_content.encode()).decode(),
                                'sha': sha
                            },
                            timeout=30
                        )
                        update_response.raise_for_status()

                    result.files_modified += 1
                    logger.info(f"{'[DRY RUN] Would fix' if dry_run else 'Fixed'}: {path}")

            except Exception as e:
                result.errors.append(f"{path}: {str(e)}")

    except Exception as e:
        result.success = False
        result.errors.append(str(e))

    return result


def normalize_ids(
    repo: str = "bobbyhiddn/Legate.Library",
    token: Optional[str] = None,
    dry_run: bool = True,
    tenant_id: str = None
) -> RecoveryResult:
    """Normalize all entry IDs to canonical format.

    Format: [tenant.]library.{category}.{slug}
    """
    token = token or os.environ.get('SYSTEM_PAT')
    result = RecoveryResult(operation="normalize_ids", success=True)
    result.details['id_changes'] = []

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
    }

    try:
        branch = get_default_branch(repo, headers)
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        response = requests.get(tree_url, headers=headers, timeout=30)
        response.raise_for_status()

        md_files = [
            item for item in response.json().get('tree', [])
            if item['type'] == 'blob' and item['path'].endswith('.md')
            and not item['path'].startswith('.') and item['path'] != 'README.md'
        ]

        result.files_processed = len(md_files)

        for item in md_files:
            path = item['path']

            try:
                content_url = f"https://api.github.com/repos/{repo}/contents/{path}"
                content_response = requests.get(content_url, headers=headers, timeout=30)
                content_response.raise_for_status()

                import base64
                file_data = content_response.json()
                content = base64.b64decode(file_data['content']).decode('utf-8')
                sha = file_data['sha']

                frontmatters, body = parse_all_frontmatter(content)

                if frontmatters:
                    fm = merge_frontmatter(frontmatters) if len(frontmatters) > 1 else frontmatters[0]
                    old_id = fm.get('id', '')
                    title = fm.get('title', '')
                    category = normalize_category(fm.get('category', ''))

                    # Generate canonical ID
                    new_id = generate_canonical_id(category, title, tenant_id)

                    # Check if ID needs updating
                    needs_update = False
                    if not old_id:
                        needs_update = True
                    elif old_id.startswith('kb-'):
                        needs_update = True
                    elif old_id != new_id:
                        # Check if it's just a format difference
                        old_parts = old_id.split('.')
                        new_parts = new_id.split('.')
                        if old_parts[-1] != new_parts[-1]:  # Different slug
                            needs_update = False  # Keep existing slug if intentional
                        else:
                            needs_update = True

                    if needs_update and old_id != new_id:
                        result.details['id_changes'].append({
                            'path': path,
                            'old_id': old_id,
                            'new_id': new_id
                        })

                        fm['id'] = new_id

                        # Ensure content_hash
                        if 'content_hash' not in fm:
                            fm['content_hash'] = compute_content_hash(body)

                        # Rebuild frontmatter
                        fm_lines = ['---']
                        priority_keys = ['id', 'title', 'category', 'created', 'content_hash']
                        for key in priority_keys:
                            if key in fm:
                                value = fm[key]
                                if isinstance(value, str) and key == 'title':
                                    fm_lines.append(f'{key}: "{value}"')
                                else:
                                    fm_lines.append(f'{key}: {value}')
                        for key, value in fm.items():
                            if key not in priority_keys:
                                if isinstance(value, str) and ' ' in value:
                                    fm_lines.append(f'{key}: "{value}"')
                                elif isinstance(value, list):
                                    fm_lines.append(f'{key}: {json.dumps(value)}')
                                else:
                                    fm_lines.append(f'{key}: {value}')
                        fm_lines.append('---')
                        fm_lines.append('')

                        new_content = '\n'.join(fm_lines) + body

                        if not dry_run:
                            update_response = requests.put(
                                content_url,
                                headers=headers,
                                json={
                                    'message': f'[recovery] Normalize ID: {old_id} -> {new_id}',
                                    'content': base64.b64encode(new_content.encode()).decode(),
                                    'sha': sha
                                },
                                timeout=30
                            )
                            update_response.raise_for_status()

                        result.files_modified += 1
                        logger.info(f"{'[DRY RUN] Would normalize' if dry_run else 'Normalized'}: {old_id} -> {new_id}")

            except Exception as e:
                result.errors.append(f"{path}: {str(e)}")

    except Exception as e:
        result.success = False
        result.errors.append(str(e))

    return result


def rebuild_content_hashes(
    repo: str = "bobbyhiddn/Legate.Library",
    token: Optional[str] = None,
    dry_run: bool = True
) -> RecoveryResult:
    """Recompute content_hash for all entries."""
    token = token or os.environ.get('SYSTEM_PAT')
    result = RecoveryResult(operation="rebuild_content_hashes", success=True)

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
    }

    try:
        branch = get_default_branch(repo, headers)
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        response = requests.get(tree_url, headers=headers, timeout=30)
        response.raise_for_status()

        md_files = [
            item for item in response.json().get('tree', [])
            if item['type'] == 'blob' and item['path'].endswith('.md')
            and not item['path'].startswith('.') and item['path'] != 'README.md'
        ]

        result.files_processed = len(md_files)

        for item in md_files:
            path = item['path']

            try:
                content_url = f"https://api.github.com/repos/{repo}/contents/{path}"
                content_response = requests.get(content_url, headers=headers, timeout=30)
                content_response.raise_for_status()

                import base64
                file_data = content_response.json()
                content = base64.b64decode(file_data['content']).decode('utf-8')
                sha = file_data['sha']

                frontmatters, body = parse_all_frontmatter(content)

                if frontmatters:
                    fm = merge_frontmatter(frontmatters) if len(frontmatters) > 1 else frontmatters[0]

                    # Compute hash
                    new_hash = compute_content_hash(body)
                    old_hash = fm.get('content_hash', '')

                    if old_hash != new_hash:
                        fm['content_hash'] = new_hash

                        # Rebuild frontmatter
                        fm_lines = ['---']
                        for key, value in fm.items():
                            if isinstance(value, str) and (key == 'title' or ' ' in value):
                                fm_lines.append(f'{key}: "{value}"')
                            elif isinstance(value, list):
                                fm_lines.append(f'{key}: {json.dumps(value)}')
                            else:
                                fm_lines.append(f'{key}: {value}')
                        fm_lines.append('---')
                        fm_lines.append('')

                        new_content = '\n'.join(fm_lines) + body

                        if not dry_run:
                            update_response = requests.put(
                                content_url,
                                headers=headers,
                                json={
                                    'message': f'[recovery] Add/update content_hash: {path}',
                                    'content': base64.b64encode(new_content.encode()).decode(),
                                    'sha': sha
                                },
                                timeout=30
                            )
                            update_response.raise_for_status()

                        result.files_modified += 1
                        logger.info(f"{'[DRY RUN] Would update' if dry_run else 'Updated'} hash: {path}")

            except Exception as e:
                result.errors.append(f"{path}: {str(e)}")

    except Exception as e:
        result.success = False
        result.errors.append(str(e))

    return result


def sync_category_descriptions(
    repo: str = "bobbyhiddn/Legate.Library",
    token: Optional[str] = None,
    dry_run: bool = True
) -> RecoveryResult:
    """Sync category descriptions from Pit database to Library.

    Creates description.md files in each category folder that doesn't have one.
    This ensures categories can be reconstructed from Library alone.
    """
    token = token or os.environ.get('SYSTEM_PAT')
    result = RecoveryResult(operation="sync_category_descriptions", success=True)
    result.details['categories_synced'] = []

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
    }

    try:
        from .rag.database import get_user_legato_db, get_user_categories

        db = get_user_legato_db()
        categories = get_user_categories(db, 'default')

        # Get existing files in repo
        branch = get_default_branch(repo, headers)
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        response = requests.get(tree_url, headers=headers, timeout=30)
        response.raise_for_status()

        existing_files = {item['path'] for item in response.json().get('tree', [])}

        for cat in categories:
            folder_name = cat['folder_name']
            desc_path = f"{folder_name}/description.md"

            # Skip if description.md already exists
            if desc_path in existing_files:
                logger.debug(f"Category description exists: {desc_path}")
                continue

            # Check if folder exists (has any files)
            folder_exists = any(f.startswith(f"{folder_name}/") for f in existing_files)
            if not folder_exists:
                logger.debug(f"Category folder doesn't exist in Library: {folder_name}")
                continue

            # Build description.md content
            content = f"""---
name: {cat['name']}
display_name: {cat['display_name']}
description: {cat['description']}
color: "{cat['color']}"
sort_order: {cat['sort_order']}
---

# {cat['display_name']}

{cat['description']}
"""
            result.details['categories_synced'].append(cat['name'])

            if not dry_run:
                import base64
                create_response = requests.put(
                    f"https://api.github.com/repos/{repo}/contents/{desc_path}",
                    headers=headers,
                    json={
                        'message': f'[recovery] Add category description: {cat["name"]}',
                        'content': base64.b64encode(content.encode()).decode()
                    },
                    timeout=30
                )
                create_response.raise_for_status()

            result.files_modified += 1
            logger.info(f"{'[DRY RUN] Would create' if dry_run else 'Created'}: {desc_path}")

    except Exception as e:
        result.success = False
        result.errors.append(str(e))

    return result


def rebuild_database_from_library(
    repo: str = "bobbyhiddn/Legate.Library",
    token: Optional[str] = None,
) -> RecoveryResult:
    """Rebuild the Pit database by re-syncing from Library.

    This is the nuclear option - wipes the knowledge_entries table
    and re-syncs everything from GitHub.
    """
    result = RecoveryResult(operation="rebuild_database", success=True)

    try:
        from .rag.database import get_user_legato_db
        from .rag.library_sync import LibrarySync

        db = get_user_legato_db()

        # Clear existing entries
        db.execute("DELETE FROM knowledge_entries")
        db.execute("DELETE FROM embeddings WHERE entry_type = 'knowledge'")
        db.commit()

        logger.info("Cleared knowledge_entries and embeddings tables")

        # Re-sync from GitHub
        sync = LibrarySync(db)
        stats = sync.sync_from_github(repo=repo, token=token)

        result.files_processed = stats.get('files_found', 0)
        result.files_modified = stats.get('entries_created', 0)
        result.details = stats

        logger.info(f"Re-synced {result.files_modified} entries from Library")

    except Exception as e:
        result.success = False
        result.errors.append(str(e))

    return result


def full_recovery(
    repo: str = "bobbyhiddn/Legate.Library",
    token: Optional[str] = None,
    dry_run: bool = True,
    tenant_id: str = None
) -> Dict[str, RecoveryResult]:
    """Run full recovery pipeline.

    Order:
    1. Validate (always runs, even in dry_run)
    2. Fix double frontmatter
    3. Normalize IDs
    4. Rebuild content hashes
    5. Sync category descriptions
    6. Rebuild database (if not dry_run)
    """
    results = {}

    logger.info(f"Starting full recovery {'(DRY RUN)' if dry_run else ''}")

    # 1. Validate
    logger.info("Step 1: Validating library...")
    report = validate_library(repo, token, tenant_id or "default")
    results['validation'] = RecoveryResult(
        operation="validate",
        success=True,
        details=report.to_dict()
    )
    logger.info(f"Found {len(report.issues)} issues")

    # 2. Fix double frontmatter
    logger.info("Step 2: Fixing double frontmatter...")
    results['fix_frontmatter'] = fix_double_frontmatter(repo, token, dry_run)
    logger.info(f"Modified {results['fix_frontmatter'].files_modified} files")

    # 3. Normalize IDs
    logger.info("Step 3: Normalizing IDs...")
    results['normalize_ids'] = normalize_ids(repo, token, dry_run, tenant_id)
    logger.info(f"Modified {results['normalize_ids'].files_modified} files")

    # 4. Rebuild content hashes
    logger.info("Step 4: Rebuilding content hashes...")
    results['rebuild_hashes'] = rebuild_content_hashes(repo, token, dry_run)
    logger.info(f"Modified {results['rebuild_hashes'].files_modified} files")

    # 5. Sync category descriptions to Library
    logger.info("Step 5: Syncing category descriptions...")
    results['sync_categories'] = sync_category_descriptions(repo, token, dry_run)
    logger.info(f"Synced {results['sync_categories'].files_modified} category descriptions")

    # 6. Rebuild database (only if not dry_run)
    if not dry_run:
        logger.info("Step 6: Rebuilding database...")
        results['rebuild_database'] = rebuild_database_from_library(repo, token)
        logger.info(f"Synced {results['rebuild_database'].files_modified} entries")
    else:
        logger.info("Step 6: Skipping database rebuild (dry run)")

    return results


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(description="Legate Studio Library Recovery Tool")
    parser.add_argument('operation', choices=[
        'validate', 'fix_frontmatter', 'normalize_ids',
        'rebuild_hashes', 'sync_categories', 'rebuild_database', 'full_recovery'
    ])
    parser.add_argument('--repo', default='bobbyhiddn/Legate.Library')
    parser.add_argument('--token', help='GitHub PAT (or set SYSTEM_PAT env var)')
    parser.add_argument('--tenant', help='Tenant ID for multi-tenant')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    token = args.token or os.environ.get('SYSTEM_PAT')
    if not token:
        logger.error("No GitHub token provided. Set SYSTEM_PAT or use --token")
        return 1

    if args.operation == 'validate':
        report = validate_library(args.repo, token, args.tenant or "default")
        print(json.dumps(report.to_dict(), indent=2))

    elif args.operation == 'fix_frontmatter':
        result = fix_double_frontmatter(args.repo, token, args.dry_run)
        print(json.dumps({
            'operation': result.operation,
            'success': result.success,
            'files_processed': result.files_processed,
            'files_modified': result.files_modified,
            'errors': result.errors
        }, indent=2))

    elif args.operation == 'normalize_ids':
        result = normalize_ids(args.repo, token, args.dry_run, args.tenant)
        print(json.dumps({
            'operation': result.operation,
            'success': result.success,
            'files_processed': result.files_processed,
            'files_modified': result.files_modified,
            'id_changes': result.details.get('id_changes', []),
            'errors': result.errors
        }, indent=2))

    elif args.operation == 'rebuild_hashes':
        result = rebuild_content_hashes(args.repo, token, args.dry_run)
        print(json.dumps({
            'operation': result.operation,
            'success': result.success,
            'files_processed': result.files_processed,
            'files_modified': result.files_modified,
            'errors': result.errors
        }, indent=2))

    elif args.operation == 'sync_categories':
        result = sync_category_descriptions(args.repo, token, args.dry_run)
        print(json.dumps({
            'operation': result.operation,
            'success': result.success,
            'files_modified': result.files_modified,
            'categories_synced': result.details.get('categories_synced', []),
            'errors': result.errors
        }, indent=2))

    elif args.operation == 'rebuild_database':
        if args.dry_run:
            logger.warning("rebuild_database cannot be run in dry-run mode")
            return 1
        result = rebuild_database_from_library(args.repo, token)
        print(json.dumps({
            'operation': result.operation,
            'success': result.success,
            'files_processed': result.files_processed,
            'files_modified': result.files_modified,
            'details': result.details,
            'errors': result.errors
        }, indent=2))

    elif args.operation == 'full_recovery':
        results = full_recovery(args.repo, token, args.dry_run, args.tenant)
        summary = {
            op: {
                'success': r.success,
                'files_modified': r.files_modified,
                'errors': len(r.errors)
            }
            for op, r in results.items()
        }
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == '__main__':
    exit(main())
