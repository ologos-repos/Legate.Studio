"""
Chat Blueprint - RAG-Enabled Conversational Interface

Provides web UI and API for chatting with the knowledge base.
Uses ChatSessionManager for in-memory buffering with periodic flush.
"""

import os
import json
import logging
import secrets
from datetime import datetime

from flask import (
    Blueprint, render_template, request, jsonify,
    session, current_app, g
)

from .core import login_required, library_required, paid_required
from .rag.chat_session_manager import get_chat_manager

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__, url_prefix='/chat')


def get_services():
    """Get or create RAG services."""
    if 'chat_services' not in g:
        from .rag.database import get_user_legato_db, init_chat_db
        from .rag.embedding_service import EmbeddingService
        from .rag.openai_provider import OpenAIEmbeddingProvider
        from .rag.context_builder import ContextBuilder
        from .rag.chat_service import ChatService, ChatProvider

        # Initialize databases
        # User-scoped legato db for embeddings/knowledge
        legato_db = get_user_legato_db()

        # chat.db for sessions/messages (shared for now)
        if 'chat_db_conn' not in g:
            g.chat_db_conn = init_chat_db()

        # Create embedding provider
        try:
            provider = OpenAIEmbeddingProvider()
        except ValueError:
            from .rag.ollama_provider import OllamaEmbeddingProvider
            provider = OllamaEmbeddingProvider()

        # Embedding service uses user's legato db
        embedding_service = EmbeddingService(provider, legato_db)
        context_builder = ContextBuilder(embedding_service)

        # Create chat service
        chat_provider = ChatProvider.CLAUDE
        if os.environ.get('CHAT_PROVIDER', 'claude').lower() == 'openai':
            chat_provider = ChatProvider.OPENAI

        chat_service = ChatService(
            provider=chat_provider,
            model=os.environ.get('CHAT_MODEL'),
        )

        g.chat_services = {
            'embedding': embedding_service,
            'context': context_builder,
            'chat': chat_service,
            'legato_db': legato_db,           # For RAG/embeddings (from get_user_legato_db)
            'chat_db': g.chat_db_conn,        # For sessions/messages
        }

    return g.chat_services


def get_or_create_session(db_conn) -> str:
    """Get current chat session or create a new one.

    Uses ChatSessionManager - session header is written to DB immediately,
    messages are buffered in memory.
    """
    session_id = session.get('chat_session_id')
    user = session.get('user', {})
    manager = get_chat_manager()

    if not session_id:
        session_id = secrets.token_urlsafe(16)
        session['chat_session_id'] = session_id

    # Ensure session exists in manager (creates in DB if new)
    manager.get_or_create_session(session_id, user.get('username'), db_conn)

    return session_id


def get_chat_history(db_conn, session_id: str, limit: int = 20):
    """Get recent chat history for a session.

    Combines messages from DB and in-memory buffer.
    """
    manager = get_chat_manager()
    messages = manager.get_messages(session_id, limit, db_conn)

    return [
        {
            'role': m['role'],
            'content': m['content'],
            'context': m.get('context_used'),
            'timestamp': None,  # Not tracked in buffer
        }
        for m in messages
    ]


def save_message(db_conn, session_id: str, role: str, content: str, context=None, model=None):
    """Save a chat message (buffered in memory, flushed periodically)."""
    manager = get_chat_manager()
    manager.add_message(session_id, role, content, context, model, db_conn)


@chat_bp.route('/')
@library_required
@paid_required
def index():
    """Chat interface page."""
    services = get_services()
    stats = services['context'].get_stats()

    return render_template(
        'chat.html',
        stats=stats,
        provider=services['chat'].provider.value,
        model=services['chat'].model,
    )


@chat_bp.route('/api/send', methods=['POST'])
@login_required
@paid_required
def send_message():
    """Send a message and get a response.

    Request body:
    {
        "message": "User's question",
        "include_context": true,  # Optional, default true
        "provider": "claude",     # Optional: claude or openai
        "model": "claude-sonnet-4-20250514"  # Optional: specific model
    }

    Response:
    {
        "response": "Assistant's response",
        "context": [{"entry_id": "...", "title": "...", "similarity": 0.85}],
        "model": "claude-sonnet-4-20250514"
    }
    """
    data = request.get_json()

    if not data or not data.get('message'):
        return jsonify({'error': 'message required'}), 400

    message = data['message']
    include_context = data.get('include_context', True)
    requested_provider = data.get('provider')
    requested_model = data.get('model')

    try:
        services = get_services()

        # If provider/model requested, create a new chat service
        chat_service = services['chat']
        if requested_provider or requested_model:
            from .rag.chat_service import ChatService, ChatProvider
            provider = ChatProvider.CLAUDE if requested_provider == 'claude' else ChatProvider.OPENAI if requested_provider == 'openai' else chat_service.provider
            chat_service = ChatService(provider=provider, model=requested_model)

        session_id = get_or_create_session(services['chat_db'])

        # Get conversation history
        history = get_chat_history(services['chat_db'], session_id, limit=10)
        history_messages = [
            {'role': h['role'], 'content': h['content']}
            for h in history
        ]

        # Build prompt with RAG context
        messages = services['context'].build_messages(
            query=message,
            history=history_messages,
        )

        # Get context entries for response
        prompt_data = services['context'].build_prompt(message)
        context_entries = prompt_data.get('context_entries', [])

        # Save user message (buffered)
        save_message(services['chat_db'], session_id, 'user', message)

        # Get LLM response
        response = chat_service.chat(messages)

        # Save assistant message with context (buffered)
        save_message(
            services['chat_db'],
            session_id,
            'assistant',
            response,
            context=context_entries,
            model=chat_service.model,
        )

        # Flush turn to disk - ensures complete turns are never lost
        manager = get_chat_manager()
        manager.flush_session(session_id, services['chat_db'])

        # Auto-title the session if no title yet (use first 50 chars of first message)
        session_row = services['chat_db'].execute(
            "SELECT title FROM chat_sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if session_row and not session_row['title']:
            title = message[:50] + ('...' if len(message) > 50 else '')
            services['chat_db'].execute(
                "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (title, session_id)
            )
            services['chat_db'].commit()

        return jsonify({
            'response': response,
            'context': context_entries if include_context else [],
            'model': chat_service.model,
            'provider': chat_service.provider.value,
        })

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/api/history', methods=['GET'])
@login_required
@paid_required
def get_history():
    """Get chat history for current session.

    Response:
    {
        "messages": [
            {
                "role": "user" | "assistant",
                "content": "...",
                "context": [...],
                "timestamp": "..."
            }
        ]
    }
    """
    try:
        services = get_services()
        session_id = session.get('chat_session_id')

        if not session_id:
            return jsonify({'messages': []})

        history = get_chat_history(services['chat_db'], session_id, limit=50)
        return jsonify({'messages': history})

    except Exception as e:
        logger.error(f"Get history failed: {e}")
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/api/sessions', methods=['GET'])
@login_required
@paid_required
def list_sessions():
    """List all chat sessions for the current user."""
    try:
        services = get_services()
        user = session.get('user', {})

        rows = services['chat_db'].execute(
            """
            SELECT session_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT 20
            """,
            (user.get('username'),),
        ).fetchall()

        return jsonify({
            'sessions': [
                {
                    'session_id': r['session_id'],
                    'title': r['title'],
                    'created_at': r['created_at'],
                    'updated_at': r['updated_at'],
                }
                for r in rows
            ]
        })

    except Exception as e:
        logger.error(f"List sessions failed: {e}")
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/api/sessions/new', methods=['POST'])
@login_required
@paid_required
def new_session():
    """Start a new chat session."""
    try:
        services = get_services()
        manager = get_chat_manager()

        # Flush current session before clearing
        old_session_id = session.get('chat_session_id')
        if old_session_id:
            manager.end_session(old_session_id, services['chat_db'])

        # Clear from flask session
        session.pop('chat_session_id', None)

        # Create new one
        session_id = get_or_create_session(services['chat_db'])

        return jsonify({
            'session_id': session_id,
            'message': 'New session created',
        })

    except Exception as e:
        logger.error(f"New session failed: {e}")
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/api/sessions/<session_id>/load', methods=['POST'])
@login_required
@paid_required
def load_session(session_id):
    """Switch to/load a specific chat session.

    Sets the current session in the Flask session and returns the session info.
    """
    try:
        services = get_services()
        user = session.get('user', {})

        # Verify ownership
        row = services['chat_db'].execute(
            "SELECT user_id, title FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        if not row or row['user_id'] != user.get('username'):
            return jsonify({'error': 'Session not found'}), 404

        # Set as current session
        session['chat_session_id'] = session_id

        return jsonify({
            'success': True,
            'session_id': session_id,
            'title': row['title'],
        })

    except Exception as e:
        logger.error(f"Load session failed: {e}")
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/api/sessions/<session_id>', methods=['DELETE'])
@login_required
@paid_required
def delete_session(session_id):
    """Delete a chat session."""
    try:
        services = get_services()
        manager = get_chat_manager()
        user = session.get('user', {})

        # Verify ownership
        row = services['chat_db'].execute(
            "SELECT user_id FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        if not row or row['user_id'] != user.get('username'):
            return jsonify({'error': 'Session not found'}), 404

        # Remove from memory cache (don't flush - we're deleting anyway)
        manager._sessions.pop(session_id, None)

        # Delete messages and session from DB
        services['chat_db'].execute(
            "DELETE FROM chat_messages WHERE session_id = ?",
            (session_id,),
        )
        services['chat_db'].execute(
            "DELETE FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        )
        services['chat_db'].commit()

        # Clear from flask session if current
        if session.get('chat_session_id') == session_id:
            session.pop('chat_session_id', None)

        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Delete session failed: {e}")
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/api/config', methods=['GET'])
@login_required
@paid_required
def get_config():
    """Get current chat configuration."""
    from .rag.chat_service import ChatService, ChatProvider

    services = get_services()

    return jsonify({
        'provider': services['chat'].provider.value,
        'model': services['chat'].model,
        'available_models': {
            'claude': ChatService.get_available_models(ChatProvider.CLAUDE),
            'openai': ChatService.get_available_models(ChatProvider.OPENAI),
        },
    })


@chat_bp.route('/api/stats', methods=['GET'])
@login_required
@paid_required
def get_stats():
    """Get RAG system statistics for debugging.

    Response:
    {
        "knowledge_entries": 42,
        "embeddings": 42,
        "provider": "text-embedding-3-small",
        "has_openai_key": true,
        "has_anthropic_key": true
    }
    """
    import os

    try:
        services = get_services()
        stats = services['context'].get_stats()

        # Add key availability info
        stats['has_openai_key'] = bool(os.environ.get('OPENAI_API_KEY'))
        stats['has_anthropic_key'] = bool(os.environ.get('ANTHROPIC_API_KEY'))
        stats['has_system_pat'] = bool(os.environ.get('SYSTEM_PAT'))

        return jsonify(stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/api/models', methods=['GET'])
@login_required
@paid_required
def get_models():
    """Get available models for a provider.

    Query params:
    - provider: 'claude' or 'openai' (default: both)

    Response:
    {
        "claude": [{"id": "...", "name": "..."}],
        "openai": [{"id": "...", "name": "..."}]
    }
    """
    from .rag.chat_service import ChatService, ChatProvider

    provider = request.args.get('provider')

    try:
        if provider == 'claude':
            return jsonify({
                'claude': ChatService.get_available_models(ChatProvider.CLAUDE),
            })
        elif provider == 'openai':
            return jsonify({
                'openai': ChatService.get_available_models(ChatProvider.OPENAI),
            })
        else:
            return jsonify({
                'claude': ChatService.get_available_models(ChatProvider.CLAUDE),
                'openai': ChatService.get_available_models(ChatProvider.OPENAI),
            })
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return jsonify({'error': str(e)}), 500
