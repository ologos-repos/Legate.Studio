"""
Chat Session Manager

Provides in-memory chat session caching with periodic flush to SQLite.
Solves SQLite locking issues in multi-machine deployments by:
- Buffering messages in memory during active conversations
- Batching writes to disk on flush triggers
- Writing session headers immediately for cross-machine visibility
"""

import json
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Singleton instance
_manager: Optional['ChatSessionManager'] = None
_manager_lock = threading.Lock()


@dataclass
class CachedSession:
    """An in-memory cached chat session."""
    session_id: str
    user_id: str
    title: Optional[str] = None
    messages: List[dict] = field(default_factory=list)  # Buffered messages not yet in DB
    last_activity: float = field(default_factory=time.time)
    created_in_db: bool = False  # Whether session row exists in DB

    def touch(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()


class ChatSessionManager:
    """
    Manages chat sessions with in-memory caching and periodic flush.

    Design:
    - Session header written to DB immediately on create
    - Messages buffered in memory during conversation
    - Flush to DB on: threshold reached, inactivity timeout, explicit end, shutdown
    """

    FLUSH_THRESHOLD = 10  # Flush after N buffered messages
    INACTIVITY_TIMEOUT = 300  # 5 minutes
    CLEANUP_INTERVAL = 30  # Check for inactive sessions every 30s

    def __init__(self):
        self._sessions: Dict[str, CachedSession] = {}
        self._lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = False

    # ============ Session Lifecycle ============

    def get_or_create_session(
        self,
        session_id: str,
        user_id: str,
        db_conn,
    ) -> CachedSession:
        """
        Get existing session or create new one.

        Session row is written to DB immediately to ensure cross-machine visibility.
        """
        with self._lock:
            # Check memory cache first
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session

        # Not in memory - check DB
        row = db_conn.execute(
            "SELECT session_id, user_id, title FROM chat_sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        with self._lock:
            if row:
                # Load from DB into memory
                session = CachedSession(
                    session_id=row['session_id'],
                    user_id=row['user_id'],
                    title=row['title'],
                    created_in_db=True,
                )
                self._sessions[session_id] = session
                logger.debug(f"Loaded session from DB: {session_id}")
            else:
                # Create new session - write to DB immediately
                db_conn.execute(
                    """
                    INSERT INTO chat_sessions (session_id, user_id)
                    VALUES (?, ?)
                    """,
                    (session_id, user_id)
                )
                db_conn.commit()

                session = CachedSession(
                    session_id=session_id,
                    user_id=user_id,
                    created_in_db=True,
                )
                self._sessions[session_id] = session
                logger.info(f"Created new session: {session_id}")

            return session

    def get_session(self, session_id: str, db_conn) -> Optional[CachedSession]:
        """Get existing session without creating."""
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session

        # Check DB
        row = db_conn.execute(
            "SELECT session_id, user_id, title FROM chat_sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        if row:
            with self._lock:
                session = CachedSession(
                    session_id=row['session_id'],
                    user_id=row['user_id'],
                    title=row['title'],
                    created_in_db=True,
                )
                self._sessions[session_id] = session
                return session

        return None

    def end_session(self, session_id: str, db_conn) -> None:
        """End a session - flush and remove from memory."""
        self.flush_session(session_id, db_conn)
        with self._lock:
            self._sessions.pop(session_id, None)
        logger.debug(f"Ended session: {session_id}")

    # ============ Message Handling ============

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        context: Optional[List[dict]],
        model: Optional[str],
        db_conn,
    ) -> None:
        """
        Add a message to the session buffer.

        Triggers flush if buffer reaches threshold.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"Adding message to unknown session: {session_id}")
                return

            session.messages.append({
                'role': role,
                'content': content,
                'context_used': json.dumps(context) if context else None,
                'model_used': model,
                'created_at': time.time(),
            })
            session.touch()

            buffer_count = len(session.messages)

        # Check flush threshold outside lock
        if buffer_count >= self.FLUSH_THRESHOLD:
            self.flush_session(session_id, db_conn)

    def get_messages(
        self,
        session_id: str,
        limit: int,
        db_conn,
    ) -> List[dict]:
        """
        Get messages for a session, combining DB and buffered messages.

        Returns messages in chronological order.
        """
        # Get messages from DB - use id for ordering (more reliable than timestamp)
        rows = db_conn.execute(
            """
            SELECT role, content, context_used, model_used, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit)
        ).fetchall()

        db_messages = []
        for row in reversed(rows):  # Reverse to chronological order
            msg = {
                'role': row['role'],
                'content': row['content'],
                'model_used': row['model_used'],
            }
            if row['context_used']:
                try:
                    msg['context_used'] = json.loads(row['context_used'])
                except json.JSONDecodeError:
                    msg['context_used'] = None
            db_messages.append(msg)

        # Add buffered messages
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                buffered = [
                    {
                        'role': m['role'],
                        'content': m['content'],
                        'model_used': m.get('model_used'),
                        'context_used': json.loads(m['context_used']) if m.get('context_used') else None,
                    }
                    for m in session.messages
                ]
            else:
                buffered = []

        # Combine and limit
        all_messages = db_messages + buffered
        return all_messages[-limit:] if len(all_messages) > limit else all_messages

    # ============ Flush Operations ============

    def flush_session(self, session_id: str, db_conn) -> int:
        """
        Flush buffered messages for a session to DB.

        Returns count of messages flushed.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.messages:
                return 0

            messages_to_flush = session.messages.copy()
            session.messages.clear()

        # Batch insert outside lock
        try:
            for msg in messages_to_flush:
                db_conn.execute(
                    """
                    INSERT INTO chat_messages
                    (session_id, role, content, context_used, model_used)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        msg['role'],
                        msg['content'],
                        msg.get('context_used'),
                        msg.get('model_used'),
                    )
                )
            db_conn.commit()
            logger.debug(f"Flushed {len(messages_to_flush)} messages for session {session_id}")
            return len(messages_to_flush)

        except Exception as e:
            logger.error(f"Failed to flush session {session_id}: {e}")
            # Restore messages on failure
            with self._lock:
                if session_id in self._sessions:
                    self._sessions[session_id].messages = messages_to_flush + self._sessions[session_id].messages
            raise

    def flush_all(self, db_conn) -> int:
        """Flush all sessions. Called on shutdown."""
        with self._lock:
            session_ids = list(self._sessions.keys())

        total_flushed = 0
        for session_id in session_ids:
            try:
                total_flushed += self.flush_session(session_id, db_conn)
            except Exception as e:
                logger.error(f"Error flushing session {session_id}: {e}")

        logger.info(f"Flushed {total_flushed} messages from {len(session_ids)} sessions")
        return total_flushed

    def flush_inactive(self, db_conn) -> int:
        """
        Flush and evict inactive sessions.

        Called periodically by cleanup thread.
        Returns count of sessions evicted.
        """
        now = time.time()
        inactive_ids = []

        with self._lock:
            for session_id, session in self._sessions.items():
                if now - session.last_activity > self.INACTIVITY_TIMEOUT:
                    inactive_ids.append(session_id)

        evicted = 0
        for session_id in inactive_ids:
            try:
                self.flush_session(session_id, db_conn)
                with self._lock:
                    self._sessions.pop(session_id, None)
                evicted += 1
                logger.debug(f"Evicted inactive session: {session_id}")
            except Exception as e:
                logger.error(f"Error evicting session {session_id}: {e}")

        if evicted:
            logger.info(f"Evicted {evicted} inactive sessions")

        return evicted

    # ============ Background Cleanup ============

    def start_cleanup_thread(self, app) -> None:
        """Start background thread for cleaning up inactive sessions."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return

        self._shutdown = False

        def cleanup_loop():
            while not self._shutdown:
                time.sleep(self.CLEANUP_INTERVAL)
                if self._shutdown:
                    break
                try:
                    with app.app_context():
                        from .database import init_chat_db
                        db = init_chat_db()
                        self.flush_inactive(db)
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("Started chat session cleanup thread")

    def stop_cleanup_thread(self) -> None:
        """Stop the cleanup thread."""
        self._shutdown = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
            logger.info("Stopped chat session cleanup thread")

    # ============ Stats ============

    def get_stats(self) -> dict:
        """Get current manager stats."""
        with self._lock:
            total_buffered = sum(len(s.messages) for s in self._sessions.values())
            return {
                'active_sessions': len(self._sessions),
                'buffered_messages': total_buffered,
            }


# ============ Module-level Functions ============

def init_chat_manager(app) -> ChatSessionManager:
    """Initialize the global chat session manager."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ChatSessionManager()
            _manager.start_cleanup_thread(app)
            logger.info("Initialized ChatSessionManager")
        return _manager


def get_chat_manager() -> ChatSessionManager:
    """Get the global chat session manager."""
    global _manager
    if _manager is None:
        raise RuntimeError("ChatSessionManager not initialized. Call init_chat_manager first.")
    return _manager


def shutdown_chat_manager(db_conn) -> None:
    """Shutdown the chat manager, flushing all sessions."""
    global _manager
    if _manager:
        _manager.stop_cleanup_thread()
        _manager.flush_all(db_conn)
        logger.info("ChatSessionManager shutdown complete")
