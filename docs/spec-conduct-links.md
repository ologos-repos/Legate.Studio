# Conduct Link Propagation Specification

**Version:** 1.0
**Created:** 2026-01-12
**Status:** Proposed
**Depends On:** spec-note-links.md

## Overview

This specification defines how Legato.Conduct should handle note links during transcript processing, knowledge extraction, and project spawning.

## Required Changes to Conduct

### 1. Knowledge Extraction (`knowledge.py`)

#### Current Behavior
- Extracts knowledge entries from transcripts
- Commits markdown files with frontmatter to Library
- No link awareness

#### New Behavior
- Detect references to existing notes in content
- Include `links` array in frontmatter
- Update `_meta/links.json` on Library

```python
# New function signature
def extract_knowledge_entry(
    thread: dict,
    existing_entries: list[dict],  # For reference detection
    pit_api_url: str = None        # For correlation API
) -> dict:
    """
    Extract knowledge entry with link detection.

    Returns dict with:
    - frontmatter: dict (includes 'links' array)
    - content: str
    - detected_links: list[dict]
    """
```

#### Reference Detection Algorithm

```python
def detect_references(content: str, existing_entries: list[dict]) -> list[dict]:
    """
    Detect references to existing notes in content.

    1. Exact title matches (case-insensitive)
    2. Entry ID mentions (kb-XXXXXXXX pattern)
    3. Semantic similarity above threshold (via Pit API)
    """
    links = []

    # Pattern 1: Entry ID mentions
    entry_id_pattern = r'\b(kb-[a-f0-9]{8})\b'
    for match in re.finditer(entry_id_pattern, content):
        entry_id = match.group(1)
        if entry_exists(entry_id, existing_entries):
            links.append({
                'target': entry_id,
                'type': 'references',
                'auto_detected': True
            })

    # Pattern 2: Title mentions
    for entry in existing_entries:
        if entry['title'].lower() in content.lower():
            links.append({
                'target': entry['entry_id'],
                'type': 'references',
                'auto_detected': True,
                'match_type': 'title'
            })

    return deduplicate_links(links)
```

### 2. Links Index Management

#### New File: `_meta/links.json`

Conduct should maintain this file in the Library repository:

```python
def update_links_index(
    repo: str,
    new_links: list[dict],
    token: str
) -> None:
    """
    Update the _meta/links.json file in the Library.

    1. Fetch current links.json (or create empty)
    2. Merge new links (deduplicate by source+target+type)
    3. Update timestamps
    4. Commit changes
    """
    current = fetch_links_index(repo, token)
    merged = merge_links(current['links'], new_links)

    new_index = {
        'version': 1,
        'updated_at': datetime.utcnow().isoformat() + 'Z',
        'links': merged
    }

    commit_file(
        repo=repo,
        path='_meta/links.json',
        content=json.dumps(new_index, indent=2),
        message='Update links index',
        token=token
    )
```

### 3. Project Spawning (`projects.py`)

#### Current Behavior
- Creates Lab repository from template
- Copies SIGNAL.md and creates issue
- No link awareness

#### New Behavior
- Include linked notes in TASKER body
- Add `depends_on` links as project prerequisites
- Create backlinks from spawned project to source notes

```python
def spawn_project(
    signal: dict,
    source_notes: list[dict],
    project_type: str = 'note'
) -> dict:
    """
    Spawn a Lab project with full link context.

    New fields in signal:
    - linked_notes: list of entry_ids with link types
    - dependencies: list of entry_ids from depends_on links
    """

    # Collect all linked notes for context
    linked_context = []
    dependencies = []

    for note in source_notes:
        # Get links for this note from Pit
        note_links = get_note_links(note['entry_id'])

        for link in note_links:
            if link['type'] == 'depends_on':
                dependencies.append(link['target'])
            linked_context.append({
                'entry_id': link['target'],
                'type': link['type'],
                'title': link['title']
            })

    # Include in SIGNAL.md
    signal['linked_notes'] = linked_context
    signal['dependencies'] = dependencies

    # Generate enhanced TASKER body
    tasker_body = generate_tasker_with_links(signal, source_notes, linked_context)

    return create_repository(signal, tasker_body, project_type)
```

#### Enhanced TASKER Body Format

```markdown
## Tasker: {title}

### Primary Notes
{list of source notes}

### Linked Context
The following notes are linked and may provide relevant context:

| Note | Relationship | Why |
|------|--------------|-----|
| [kb-abc123] Async Patterns | implements | This note implements the pattern |
| [kb-def456] Error Handling | references | Referenced in primary note |

### Dependencies
These notes should be reviewed/completed before this project:
- [kb-ghi789] Core Architecture (depends_on)

### Implementation Notes
{content}

---
*Queued via Conduct | {n} linked notes | {m} dependencies*
```

### 4. Transcript Processing Workflow

#### Updated Flow

```yaml
# .github/workflows/process-transcript.yml

jobs:
  process:
    steps:
      - name: Fetch existing entries
        run: |
          # Fetch entry list from Pit for reference detection
          curl -H "Authorization: Bearer $PIT_TOKEN" \
            "$PIT_URL/library/api/entries" > entries.json

      - name: Process transcript
        run: |
          python -m legato_conduct.classifier \
            --transcript "${{ inputs.transcript }}" \
            --existing-entries entries.json \
            --detect-links

      - name: Commit with links
        run: |
          python -m legato_conduct.knowledge \
            --entries classified.json \
            --update-links-index
```

### 5. API Integration

#### New Pit API Endpoints for Conduct

Conduct will call these Pit endpoints:

```
GET /library/api/entries
  → Returns list of all entry_ids, titles, categories

GET /library/api/links/{entry_id}
  → Returns all links for a specific entry

POST /library/api/links/sync
  → Syncs links discovered during processing
  Body: { "links": [...] }

GET /memory/correlate
  → Existing semantic similarity endpoint
  → Used for auto-link detection
```

### 6. Configuration

#### Environment Variables

```bash
# Enable link detection during processing
CONDUCT_DETECT_LINKS=true

# Semantic similarity threshold for auto-linking
CONDUCT_LINK_SIMILARITY_THRESHOLD=0.85

# Pit API URL for correlation checks
PIT_API_URL=https://legate.studio

# Include links in spawned projects
CONDUCT_INCLUDE_PROJECT_LINKS=true
```

## Migration Path

1. **Phase 1:** Add link detection to knowledge extraction (no breaking changes)
2. **Phase 2:** Create `_meta/links.json` on first detected link
3. **Phase 3:** Update project spawning to include link context
4. **Phase 4:** Add bidirectional sync between Pit and Library links

## Testing

### Unit Tests

```python
def test_detect_entry_id_references():
    content = "This builds on kb-abc12345 and references kb-def67890."
    existing = [
        {'entry_id': 'kb-abc12345', 'title': 'Note A'},
        {'entry_id': 'kb-def67890', 'title': 'Note B'},
    ]
    links = detect_references(content, existing)
    assert len(links) == 2
    assert all(l['type'] == 'references' for l in links)

def test_detect_title_references():
    content = "As discussed in Async Patterns, we should..."
    existing = [
        {'entry_id': 'kb-abc12345', 'title': 'Async Patterns'},
    ]
    links = detect_references(content, existing)
    assert len(links) == 1
    assert links[0]['target'] == 'kb-abc12345'
```

### Integration Tests

1. Process transcript with known references → verify links created
2. Spawn project from linked notes → verify TASKER includes context
3. Sync links from Library → verify Pit database updated

## Rollback Plan

If issues arise:
1. Set `CONDUCT_DETECT_LINKS=false` to disable link detection
2. Links already created remain in Pit database
3. `_meta/links.json` can be deleted without breaking anything
4. Spawned projects with link context continue to work (just extra info)
