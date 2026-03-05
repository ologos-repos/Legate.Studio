"""
Memory API - RAG Endpoints for Pipeline Integration

Provides REST API for:
- Correlation checking (before extraction)
- Entry registration (after extraction)
- Similarity search
"""

import os
import logging
from functools import wraps

from flask import Blueprint, request, jsonify, current_app, g

logger = logging.getLogger(__name__)

memory_api_bp = Blueprint('memory_api', __name__, url_prefix='/memory/api')


def require_api_token(f):
    """Decorator to require Bearer token authentication.

    NOTE: This API is designed for machine-to-machine auth (AI agents -> Pit).
    In multi-tenant mode, this API needs redesign to include user context
    in requests (e.g., user_id header or per-user API keys).

    Currently uses SYSTEM_PAT for single-tenant backwards compatibility.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')

        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401

        token = auth_header[7:]  # Remove 'Bearer ' prefix

        # Multi-tenant mode: Check for user-specific token first
        # TODO: Implement per-user API tokens for multi-tenant
        mode = current_app.config.get('LEGATO_MODE', 'single-tenant')
        if mode == 'multi-tenant':
            # For now, reject requests in multi-tenant mode without proper auth
            # In future: validate user-specific tokens
            return jsonify({'error': 'Memory API not available in multi-tenant mode (needs redesign)'}), 403

        # Single-tenant: Use SYSTEM_PAT
        expected_token = current_app.config.get('SYSTEM_PAT')
        if not expected_token or token != expected_token:
            return jsonify({'error': 'Invalid token'}), 403

        return f(*args, **kwargs)
    return decorated


def get_db():
    """Get legato database connection for current user."""
    from .rag.database import get_user_legato_db
    return get_user_legato_db()


def get_embedding_service():
    """Get or create the embedding service."""
    if 'embedding_service' not in g:
        from .rag.embedding_service import EmbeddingService
        from .rag.openai_provider import OpenAIEmbeddingProvider

        # Initialize database for current user
        db_conn = get_db()

        # Create provider (prefer OpenAI for API consistency)
        try:
            provider = OpenAIEmbeddingProvider()
        except ValueError:
            # Fall back to Ollama if OpenAI not configured
            from .rag.ollama_provider import OllamaEmbeddingProvider
            provider = OllamaEmbeddingProvider()

        g.embedding_service = EmbeddingService(provider, db_conn)

    return g.embedding_service


@memory_api_bp.route('/health', methods=['GET'])
def health():
    """Health check for the memory API."""
    return jsonify({
        'status': 'healthy',
        'service': 'memory_api',
    })


@memory_api_bp.route('/correlate', methods=['POST'])
@require_api_token
def correlate():
    """Check if similar content already exists (with chord-aware routing).

    Request body:
    {
        "title": "Entry title",
        "content": "Entry content",
        "key_phrases": ["optional", "phrases"],  # Optional
        "needs_chord": false  # Whether this thread wants implementation
    }

    Response:
    {
        "action": "CREATE" | "APPEND" | "QUEUE" | "SKIP",
        "score": 0.85,
        "matches": [
            {
                "entry_id": "kb-001",
                "title": "...",
                "similarity": 0.85,
                "chord_status": "active",
                "chord_repo": "org/repo-name"
            }
        ],
        "recommendation": {
            "entry_id": "kb-001",
            "reason": "High similarity (0.85) with active chord - queue agent task"
        }
    }

    Action meanings:
    - CREATE: No similar entries, create new note
    - APPEND: Very similar entry exists, append content to it
    - QUEUE: Similar entry exists with active chord, queue agent task instead of new chord
    - SKIP: Exact/near-exact duplicate, skip processing
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    title = data.get('title', '')
    content = data.get('content', '')
    key_phrases = data.get('key_phrases', [])
    needs_chord = data.get('needs_chord', False)

    if not title and not content:
        return jsonify({'error': 'title or content required'}), 400

    # Include key phrases in content for better matching
    query_text = f"{title}\n\n{content}"
    if key_phrases:
        query_text = f"{query_text}\n\nKey phrases: {', '.join(key_phrases)}"

    try:
        service = get_embedding_service()
        db = get_db()

        # Find similar entries
        similar = service.find_similar(
            query_text=query_text,
            entry_type='knowledge',
            limit=5,
            threshold=0.5,  # Lower threshold to catch more potential matches
        )

        # Enrich matches with chord status
        matches = []
        for s in similar:
            row = db.execute(
                """
                SELECT entry_id, title, category, chord_status, chord_repo, needs_chord
                FROM knowledge_entries
                WHERE entry_id = ?
                """,
                (s['entry_id'],)
            ).fetchone()

            if row:
                matches.append({
                    'entry_id': row['entry_id'],
                    'title': row['title'],
                    'category': row['category'],
                    'similarity': round(s['similarity'], 3),
                    'chord_status': row['chord_status'],
                    'chord_repo': row['chord_repo'],
                    'needs_chord': bool(row['needs_chord']),
                })

        # Determine action based on similarity and chord status
        action = 'CREATE'
        recommendation = None
        best_score = matches[0]['similarity'] if matches else 0.0

        if matches:
            best_match = matches[0]

            if best_score >= 0.95:
                # Near-exact duplicate
                action = 'SKIP'
                recommendation = {
                    'entry_id': best_match['entry_id'],
                    'reason': f'Near-exact duplicate (similarity={best_score:.2f})'
                }
            elif best_score >= 0.80:
                # High similarity
                if needs_chord and best_match.get('chord_status') == 'active':
                    # This wants a chord, but similar entry already has one
                    action = 'QUEUE'
                    recommendation = {
                        'entry_id': best_match['entry_id'],
                        'chord_repo': best_match['chord_repo'],
                        'reason': f'Similar note has active chord at {best_match["chord_repo"]} - queue agent task instead'
                    }
                elif needs_chord and best_match.get('chord_status') == 'pending':
                    # Chord already pending
                    action = 'SKIP'
                    recommendation = {
                        'entry_id': best_match['entry_id'],
                        'reason': f'Similar note already has pending chord request'
                    }
                else:
                    # Similar content, should append
                    action = 'APPEND'
                    recommendation = {
                        'entry_id': best_match['entry_id'],
                        'reason': f'High similarity (similarity={best_score:.2f}) - append to existing note'
                    }
            elif best_score >= 0.65:
                # Moderate similarity - suggest but create
                action = 'CREATE'
                recommendation = {
                    'entry_id': best_match['entry_id'],
                    'reason': f'Related note exists (similarity={best_score:.2f}) but distinct enough to create new'
                }

        logger.info(f"Correlation check: {title[:50]}... -> {action} (score={best_score:.2f})")

        return jsonify({
            'action': action,
            'score': round(best_score, 3),
            'matches': matches,
            'recommendation': recommendation,
        })

    except Exception as e:
        logger.error(f"Correlation failed: {e}")
        return jsonify({'error': str(e)}), 500


@memory_api_bp.route('/append', methods=['POST'])
@require_api_token
def append_to_entry():
    """Append content to an existing knowledge entry.

    Used when the correlation check returns APPEND action.

    Request body:
    {
        "entry_id": "kb-abc123",
        "content": "New content to append",
        "source_transcript": "transcript-123"
    }

    Response:
    {
        "success": true,
        "entry_id": "kb-abc123",
        "action": "appended"
    }
    """
    from .rag.github_service import commit_file
    import os

    data = request.get_json()

    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    entry_id = data.get('entry_id')
    new_content = data.get('content', '')
    source_transcript = data.get('source_transcript', 'unknown')

    if not entry_id:
        return jsonify({'error': 'entry_id required'}), 400

    if not new_content:
        return jsonify({'error': 'content required'}), 400

    try:
        db = get_db()

        # Get existing entry
        entry = db.execute(
            "SELECT * FROM knowledge_entries WHERE entry_id = ?",
            (entry_id,)
        ).fetchone()

        if not entry:
            return jsonify({'error': f'Entry {entry_id} not found'}), 404

        entry_dict = dict(entry)
        file_path = entry_dict.get('file_path')

        # Append content with separator
        from datetime import datetime
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
        appended = f"\n\n---\n\n## Appended ({timestamp})\n\n*Source: {source_transcript}*\n\n{new_content}"

        new_full_content = entry_dict['content'] + appended

        # Update database
        db.execute(
            """
            UPDATE knowledge_entries
            SET content = ?, updated_at = CURRENT_TIMESTAMP
            WHERE entry_id = ?
            """,
            (new_full_content, entry_id)
        )
        db.commit()

        # Commit to GitHub if file_path exists
        if file_path:
            token = os.environ.get('SYSTEM_PAT')
            if token:
                try:
                    from .core import get_user_library_repo
                    commit_file(
                        repo=get_user_library_repo(),
                        path=file_path,
                        content=new_full_content,
                        message=f'Append content from {source_transcript}',
                        token=token
                    )
                    logger.info(f"Appended content to {entry_id} and committed to GitHub")
                except Exception as e:
                    logger.warning(f"Failed to commit append to GitHub: {e}")

        logger.info(f"Appended content to entry {entry_id}")

        return jsonify({
            'success': True,
            'entry_id': entry_id,
            'action': 'appended',
        })

    except Exception as e:
        logger.error(f"Append failed: {e}")
        return jsonify({'error': str(e)}), 500


@memory_api_bp.route('/queue-task', methods=['POST'])
@require_api_token
def queue_agent_task():
    """Queue an agent task for an existing chord.

    Used when correlation check returns QUEUE action - the note wants a chord
    but a similar note already has an active chord. Instead of creating a
    duplicate chord, we queue a task issue on the existing chord's repo.

    Request body:
    {
        "chord_repo": "org/repo-name",
        "title": "Add feature X",
        "description": "Detailed description of the task",
        "source_entry_id": "kb-abc123",
        "source_transcript": "transcript-123"
    }

    Response:
    {
        "success": true,
        "issue_url": "https://github.com/org/repo/issues/42"
    }
    """
    import requests as http_requests
    import os

    data = request.get_json()

    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    chord_repo = data.get('chord_repo')
    title = data.get('title')
    description = data.get('description', '')
    source_entry_id = data.get('source_entry_id')
    source_transcript = data.get('source_transcript')

    if not chord_repo or not title:
        return jsonify({'error': 'chord_repo and title required'}), 400

    try:
        token = os.environ.get('SYSTEM_PAT')
        if not token:
            return jsonify({'error': 'SYSTEM_PAT not configured'}), 500

        # Create issue on the chord repo
        issue_body = f"""## Task Request

{description}

---

**Source:** `{source_entry_id or 'unknown'}`
**Transcript:** `{source_transcript or 'unknown'}`

*Auto-generated by LEGATO Listen - similar request routed to existing chord*
"""

        response = http_requests.post(
            f'https://api.github.com/repos/{chord_repo}/issues',
            headers={
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.github+json',
            },
            json={
                'title': f'[LEGATO] {title}',
                'body': issue_body,
                'labels': ['legato-task'],
            },
            timeout=15,
        )

        if response.status_code == 201:
            issue_data = response.json()
            logger.info(f"Created issue on {chord_repo}: {issue_data['html_url']}")

            return jsonify({
                'success': True,
                'issue_url': issue_data['html_url'],
                'issue_number': issue_data['number'],
            })
        else:
            return jsonify({
                'error': f'Failed to create issue: {response.status_code}',
                'detail': response.text,
            }), response.status_code

    except Exception as e:
        logger.error(f"Queue task failed: {e}")
        return jsonify({'error': str(e)}), 500


@memory_api_bp.route('/register', methods=['POST'])
@require_api_token
def register():
    """Register a new knowledge entry.

    Request body:
    {
        "entry_id": "kb-001",
        "title": "Entry title",
        "category": "concepts",
        "content": "Entry content",
        "source_thread": "thread-001",  # Optional
        "source_transcript": "transcript.txt"  # Optional
    }

    Response:
    {
        "success": true,
        "id": 1,
        "embedding_generated": true
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    required = ['entry_id', 'title', 'content']
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({'error': f'Missing required fields: {missing}'}), 400

    try:
        service = get_embedding_service()
        conn = service.conn

        # Insert entry
        cursor = conn.execute(
            """
            INSERT INTO knowledge_entries
            (entry_id, title, category, content, source_thread, source_transcript)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id) DO UPDATE SET
                title = excluded.title,
                category = excluded.category,
                content = excluded.content,
                source_thread = excluded.source_thread,
                source_transcript = excluded.source_transcript,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            (
                data['entry_id'],
                data['title'],
                data.get('category', 'general'),
                data['content'],
                data.get('source_thread'),
                data.get('source_transcript'),
            ),
        )
        row = cursor.fetchone()
        entry_db_id = row[0]
        conn.commit()

        # Generate embedding
        text = f"Title: {data['title']}\n\nContent: {data['content']}"
        embedding = service.generate_and_store(entry_db_id, 'knowledge', text)

        logger.info(f"Registered entry: {data['entry_id']}")

        return jsonify({
            'success': True,
            'id': entry_db_id,
            'embedding_generated': embedding is not None,
        })

    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return jsonify({'error': str(e)}), 500


@memory_api_bp.route('/search', methods=['GET'])
@require_api_token
def search():
    """Search for similar entries.

    Query params:
    - q: Search query (required)
    - limit: Max results (default 10)
    - threshold: Min similarity (default 0.4)
    - type: Entry type (default 'knowledge')

    Response:
    {
        "results": [
            {
                "entry_id": "kb-001",
                "title": "...",
                "category": "concepts",
                "similarity": 0.85,
                "snippet": "First 200 chars..."
            }
        ]
    }
    """
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'q parameter required'}), 400

    limit = int(request.args.get('limit', 10))
    threshold = float(request.args.get('threshold', 0.4))
    entry_type = request.args.get('type', 'knowledge')

    try:
        service = get_embedding_service()
        results = service.find_similar(
            query_text=query,
            entry_type=entry_type,
            limit=limit,
            threshold=threshold,
        )

        # Format results
        formatted = [
            {
                'entry_id': r['entry_id'],
                'title': r['title'],
                'category': r.get('category'),
                'similarity': round(r['similarity'], 3),
                'snippet': (r.get('content', '')[:200] + '...') if r.get('content') else None,
            }
            for r in results
        ]

        return jsonify({'results': formatted})

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({'error': str(e)}), 500


@memory_api_bp.route('/stats', methods=['GET'])
@require_api_token
def stats():
    """Get knowledge base statistics.

    Response:
    {
        "knowledge_entries": 42,
        "project_entries": 5,
        "embeddings": 42,
        "provider": "openai:text-embedding-3-small"
    }
    """
    try:
        from .rag.context_builder import ContextBuilder

        service = get_embedding_service()
        builder = ContextBuilder(service)

        return jsonify(builder.get_stats())

    except Exception as e:
        logger.error(f"Stats failed: {e}")
        return jsonify({'error': str(e)}), 500


@memory_api_bp.route('/sync', methods=['POST'])
@require_api_token
def trigger_sync():
    """Trigger a sync from Library repository.

    Request body (optional):
    {
        "clear": true  // Clear all entries before sync (fixes duplicates)
    }
    """
    from .rag.library_sync import LibrarySync
    from .rag.embedding_service import EmbeddingService
    from .rag.openai_provider import OpenAIEmbeddingProvider

    data = request.get_json() or {}
    clear_first = data.get('clear', False)

    try:
        db = get_db()

        # Clear existing entries if requested
        if clear_first:
            db.execute("DELETE FROM embeddings WHERE entry_type = 'knowledge'")
            db.execute("DELETE FROM knowledge_entries")
            db.commit()
            logger.info("Cleared knowledge entries before sync")

        # Create embedding service if possible
        embedding_service = None
        if os.environ.get('OPENAI_API_KEY'):
            try:
                provider = OpenAIEmbeddingProvider()
                embedding_service = EmbeddingService(provider, db)
            except Exception:
                pass

        sync = LibrarySync(db, embedding_service)
        token = os.environ.get('SYSTEM_PAT')
        from .core import get_user_library_repo
        stats = sync.sync_from_github(get_user_library_repo(), token=token)

        return jsonify({
            'status': 'success',
            'stats': stats,
        })
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        return jsonify({'error': str(e)}), 500


@memory_api_bp.route('/pipeline/status', methods=['POST'])
@require_api_token
def pipeline_status():
    """Update pipeline status.

    Called by Conduct workflows to report progress.

    Request body:
    {
        "run_id": "12345",
        "stage": "parse" | "pre-classify" | "classify" | "process-knowledge" | "process-projects" | "complete",
        "status": "started" | "success" | "failed",
        "details": {
            "thread_count": 5,
            "knowledge_count": 3,
            ...
        }
    }

    For "pre-classify" stage, details should include:
    {
        "categories": [{"name": "epiphany", "description": "..."}],
        "motifs": [{"id": "thread-1", "preview": "First 200 chars...", "title": "..."}]
    }

    Response:
    {
        "success": true,
        "message": "Status updated"
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    required = ['run_id', 'stage', 'status']
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({'error': f'Missing required fields: {missing}'}), 400

    try:
        import json

        db = get_db()

        # Store pipeline status
        db.execute(
            """
            INSERT INTO pipeline_runs (run_id, stage, status, details, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id, stage) DO UPDATE SET
                status = excluded.status,
                details = excluded.details,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                data['run_id'],
                data['stage'],
                data['status'],
                json.dumps(data.get('details', {})),
            ),
        )
        db.commit()

        logger.info(f"Pipeline {data['run_id']} stage {data['stage']}: {data['status']}")

        # Enhanced logging for pre-classify stage to aid debugging
        if data['stage'] == 'pre-classify':
            details = data.get('details', {})
            categories = details.get('categories', [])
            motifs = details.get('motifs', [])

            logger.info(f"PRE-CLASSIFY DEBUG for run {data['run_id']}:")
            logger.info(f"  Categories available ({len(categories)}): {[c.get('name') for c in categories]}")
            for motif in motifs[:5]:  # Log first 5 motifs
                preview = motif.get('preview', '')[:100]
                logger.info(f"  Motif '{motif.get('id', 'unknown')}': {preview}...")
            if len(motifs) > 5:
                logger.info(f"  ... and {len(motifs) - 5} more motifs")

        return jsonify({
            'success': True,
            'message': 'Status updated',
        })

    except Exception as e:
        logger.error(f"Pipeline status update failed: {e}")
        return jsonify({'error': str(e)}), 500


@memory_api_bp.route('/pipeline/pre-classify/<run_id>', methods=['GET'])
@require_api_token
def get_pre_classify_details(run_id: str):
    """Get pre-classify debug details for a pipeline run.

    This endpoint helps debug classification issues by showing:
    - What categories were available to the classifier
    - Preview of each motif being classified

    Response:
    {
        "run_id": "12345",
        "found": true,
        "categories": [{"name": "epiphany", "description": "..."}],
        "motifs": [{"id": "thread-1", "preview": "...", "title": "..."}],
        "updated_at": "2024-01-15T10:30:00"
    }
    """
    try:
        import json

        db = get_db()

        row = db.execute(
            """
            SELECT details, updated_at
            FROM pipeline_runs
            WHERE run_id = ? AND stage = 'pre-classify'
            """,
            (run_id,),
        ).fetchone()

        if not row:
            return jsonify({
                'run_id': run_id,
                'found': False,
                'message': 'No pre-classify stage found for this run. Conduct may need to be updated to report this stage.',
            })

        details = json.loads(row['details']) if row['details'] else {}

        return jsonify({
            'run_id': run_id,
            'found': True,
            'categories': details.get('categories', []),
            'motifs': details.get('motifs', []),
            'updated_at': row['updated_at'],
        })

    except Exception as e:
        logger.error(f"Get pre-classify details failed: {e}")
        return jsonify({'error': str(e)}), 500


@memory_api_bp.route('/pipeline/status/<run_id>', methods=['GET'])
@require_api_token
def get_pipeline_status(run_id: str):
    """Get pipeline status for a run.

    Response:
    {
        "run_id": "12345",
        "stages": [
            {"stage": "parse", "status": "success", "details": {...}, "updated_at": "..."},
            {"stage": "classify", "status": "running", ...}
        ]
    }
    """
    try:
        import json

        db = get_db()

        rows = db.execute(
            """
            SELECT stage, status, details, updated_at
            FROM pipeline_runs
            WHERE run_id = ?
            ORDER BY updated_at
            """,
            (run_id,),
        ).fetchall()

        stages = [
            {
                'stage': r['stage'],
                'status': r['status'],
                'details': json.loads(r['details']) if r['details'] else {},
                'updated_at': r['updated_at'],
            }
            for r in rows
        ]

        return jsonify({
            'run_id': run_id,
            'stages': stages,
        })

    except Exception as e:
        logger.error(f"Get pipeline status failed: {e}")
        return jsonify({'error': str(e)}), 500
