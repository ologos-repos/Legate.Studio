# Library Recovery Guide

This document describes how to recover from data corruption in Legato.Library without accessing user content (blind recovery).

## Overview

The recovery system is designed for multi-tenant environments where support staff should not need to view user content to fix issues. All operations work on structural metadata only.

## Common Issues and Fixes

| Issue | Symptom | Recovery Operation |
|-------|---------|-------------------|
| Double frontmatter | Files have duplicate `---` blocks | `fix_frontmatter` |
| ID mismatch | MCP shows different ID than search | `normalize_ids` |
| Missing content_hash | Integrity checks fail | `rebuild_hashes` |
| Database out of sync | Notes missing in Pit | `rebuild_database` |
| Category folder typos | `epiphanys`, `tech-thoughtss` | `normalize_ids` + file move |

## ID Format

All entry IDs should follow the canonical format:

```
library.{category}.{slug}
```

For multi-tenant:
```
{tenant}.library.{category}.{slug}
```

Examples:
- `library.concept.legato-ingress-methods`
- `library.epiphany.oracle-machines-insight`
- `acme.library.reflection.morning-thoughts`

## Recovery CLI

### Validate Library

Check for issues without making changes:

```bash
cd Legato.Pit
python -m legate_studio.recovery validate --repo bobbyhiddn/Legato.Library
```

Output includes:
- `double_frontmatter`: Files with multiple `---` blocks
- `legacy_id_format`: IDs using old `kb-xxxxx` format
- `missing_content_hash`: Files without integrity hash
- `category_mismatch`: Frontmatter category doesn't match folder
- `category_typo`: Folder name has typos (e.g., `epiphanys`)

### Fix Double Frontmatter

Preview:
```bash
python -m legate_studio.recovery fix_frontmatter --dry-run
```

Apply:
```bash
python -m legate_studio.recovery fix_frontmatter
```

This merges multiple frontmatter blocks, keeping the first block's values and adding any unique fields from subsequent blocks.

### Normalize IDs

Preview:
```bash
python -m legate_studio.recovery normalize_ids --dry-run
```

Apply:
```bash
python -m legate_studio.recovery normalize_ids
```

Converts legacy IDs to canonical format:
- `kb-a1b2c3d4` → `library.concept.my-note-title`
- `library.concepts.my-note` → `library.concept.my-note` (singular)

### Rebuild Content Hashes

Preview:
```bash
python -m legate_studio.recovery rebuild_hashes --dry-run
```

Apply:
```bash
python -m legate_studio.recovery rebuild_hashes
```

Computes SHA256 hash of content body and stores in frontmatter for integrity verification.

### Rebuild Database

**Warning**: This clears and rebuilds the knowledge_entries table.

```bash
python -m legate_studio.recovery rebuild_database
```

Use this when the Pit database is out of sync with Library.

### Full Recovery

Run all recovery operations in order:

Preview:
```bash
python -m legate_studio.recovery full_recovery --dry-run
```

Apply:
```bash
python -m legate_studio.recovery full_recovery
```

Order of operations:
1. Validate (report issues)
2. Fix double frontmatter
3. Normalize IDs
4. Rebuild content hashes
5. Rebuild database (if not dry-run)

## Multi-Tenant Recovery

For tenant-specific recovery:

```bash
python -m legate_studio.recovery validate --tenant acme-corp
python -m legate_studio.recovery full_recovery --tenant acme-corp --dry-run
```

The tenant ID is prefixed to all generated IDs: `acme-corp.library.concept.my-note`

## Manual Recovery Steps

### Merge Duplicate Category Folders

If you have both `tech-thoughts` and `tech-thoughtss` folders:

1. Run validation to identify affected files:
   ```bash
   python -m legate_studio.recovery validate | grep category_typo
   ```

2. Manually move files in GitHub or locally:
   ```bash
   git mv tech-thoughtss/* tech-thought/
   rmdir tech-thoughtss
   ```

3. Run ID normalization:
   ```bash
   python -m legate_studio.recovery normalize_ids
   ```

4. Rebuild database:
   ```bash
   python -m legate_studio.recovery rebuild_database
   ```

### Recover Specific File

If a single file is corrupted:

1. Check file in Library repo (git history preserved)
2. Fix frontmatter manually or use recovery script
3. Sync Pit:
   ```bash
   # Via API
   curl -X POST https://pit.legato.dev/api/sync
   ```

## Content Hash Verification

The `content_hash` field in frontmatter is a SHA256 hash (first 16 chars) of the markdown body content. Use it to:

1. **Detect tampering**: Recompute hash and compare
2. **Deduplicate**: Same content produces same hash
3. **Track changes**: Hash changes when content changes

Verify a file's hash:
```python
import hashlib
with open('note.md') as f:
    content = f.read()
    # Extract body after frontmatter
    body = content.split('---', 2)[2].strip()
    computed = hashlib.sha256(body.encode()).hexdigest()[:16]
    print(f"Computed: {computed}")
```

## Environment Variables

- `SYSTEM_PAT`: GitHub Personal Access Token with repo access
- `LEGATO_ORG`: GitHub organization (default: `bobbyhiddn`)

## Recovery Guarantees

1. **No data loss**: Git history preserves all versions
2. **Idempotent**: Running recovery multiple times is safe
3. **Blind operation**: No content access required for support
4. **Audit trail**: All changes committed with `[recovery]` prefix
5. **Rollback**: Git revert if recovery causes issues

## Troubleshooting

### "GitHub token required"
Set `SYSTEM_PAT` environment variable or use `--token` flag.

### "File not found" errors
The file may have been deleted or moved. Check git history.

### ID conflicts after normalization
Two files may have the same slug. The recovery script logs conflicts - resolve manually by renaming one file.

### Database rebuild fails
Check database permissions and that Fly.io volume is mounted at `/data`.
