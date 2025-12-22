"""
Multi-turn Conversation Support - Phase 3.5
Enables follow-up questions with conversation history

Features:
- Conversation session management
- Context preservation across turns
- Reference resolution (it, this, that)
- Session timeout and cleanup
"""
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import threading
import re

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ConversationSession:
    """A conversation session with history"""
    session_id: str
    user_id: str
    product_context: Optional[str] = None  # Current product being discussed
    part_number: Optional[str] = None
    language: str = "en"
    turns: List[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def add_turn(self, role: str, content: str, metadata: Dict = None):
        """Add a turn to the conversation"""
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.turns.append(turn)
        self.last_activity = datetime.now()
    
    def get_history(self, max_turns: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        recent = self.turns[-max_turns:] if len(self.turns) > max_turns else self.turns
        return [t.to_dict() for t in recent]
    
    def get_context_summary(self) -> str:
        """Generate a summary of conversation context"""
        if not self.turns:
            return ""
        
        context_parts = []
        
        if self.product_context:
            context_parts.append(f"Current product: {self.product_context}")
        
        if self.part_number:
            context_parts.append(f"Part number: {self.part_number}")
        
        # Summarize recent topics
        recent_user_messages = [
            t.content for t in self.turns[-6:] 
            if t.role == "user"
        ]
        
        if recent_user_messages:
            context_parts.append(f"Recent discussion: {'; '.join(recent_user_messages[-3:])}")
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "product_context": self.product_context,
            "part_number": self.part_number,
            "language": self.language,
            "turn_count": len(self.turns),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }


class ConversationManager:
    """
    Manages multi-turn conversation sessions
    
    Features:
    - Session creation and retrieval
    - Automatic session cleanup
    - Context preservation
    - Reference resolution
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, session_timeout_minutes: int = 30, max_sessions: int = 1000):
        """
        Initialize conversation manager
        
        Args:
            session_timeout_minutes: Session timeout in minutes
            max_sessions: Maximum sessions to keep in memory
        """
        if self._initialized:
            return
            
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_sessions = max_sessions
        self.sessions: OrderedDict[str, ConversationSession] = OrderedDict()
        self._session_lock = threading.RLock()
        self._initialized = True
        
        logger.info(f"Conversation Manager initialized (timeout: {session_timeout_minutes}min)")
    
    def create_session(
        self,
        user_id: str,
        product_context: str = None,
        part_number: str = None,
        language: str = "en"
    ) -> ConversationSession:
        """Create a new conversation session"""
        with self._session_lock:
            session_id = str(uuid.uuid4())[:8]
            
            session = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                product_context=product_context,
                part_number=part_number,
                language=language
            )
            
            self.sessions[session_id] = session
            self._cleanup_old_sessions()
            
            logger.info(f"Created conversation session: {session_id} for user: {user_id}")
            return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get an existing session"""
        with self._session_lock:
            session = self.sessions.get(session_id)
            
            if session:
                # Check if expired
                if datetime.now() - session.last_activity > self.session_timeout:
                    del self.sessions[session_id]
                    logger.info(f"Session expired: {session_id}")
                    return None
                
                # Move to end (most recently used)
                self.sessions.move_to_end(session_id)
                return session
            
            return None
    
    def get_or_create_session(
        self,
        session_id: Optional[str],
        user_id: str,
        product_context: str = None,
        part_number: str = None,
        language: str = "en"
    ) -> ConversationSession:
        """Get existing session or create new one"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                # Update context if provided
                if product_context:
                    session.product_context = product_context
                if part_number:
                    session.part_number = part_number
                return session
        
        return self.create_session(user_id, product_context, part_number, language)
    
    def add_user_message(
        self,
        session_id: str,
        content: str,
        metadata: Dict = None
    ) -> bool:
        """Add a user message to session"""
        session = self.get_session(session_id)
        if session:
            session.add_turn("user", content, metadata)
            return True
        return False
    
    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        metadata: Dict = None
    ) -> bool:
        """Add an assistant response to session"""
        session = self.get_session(session_id)
        if session:
            session.add_turn("assistant", content, metadata)
            return True
        return False
    
    def resolve_references(
        self,
        query: str,
        session: ConversationSession
    ) -> str:
        """
        Resolve references like 'it', 'this', 'that' using conversation context
        
        Args:
            query: User query with potential references
            session: Conversation session with history
            
        Returns:
            Query with resolved references
        """
        if not session.turns:
            return query
        
        # Patterns to detect references
        reference_patterns = [
            (r'\b(it|this tool|this product|this)\b', 'product'),
            (r'\b(that error|this error|the error|that issue|this issue)\b', 'error'),
            (r'\b(same problem|same issue|this problem)\b', 'problem'),
        ]
        
        resolved_query = query
        
        for pattern, ref_type in reference_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                if ref_type == 'product' and session.product_context:
                    resolved_query = re.sub(
                        pattern,
                        session.product_context,
                        resolved_query,
                        flags=re.IGNORECASE
                    )
                elif ref_type in ('error', 'problem'):
                    # Look for recent error mention in history
                    for turn in reversed(session.turns):
                        if turn.role == "user":
                            # Extract potential error/problem description
                            error_match = re.search(
                                r'(error|problem|issue|fault)[:\s]+([^.]+)',
                                turn.content,
                                re.IGNORECASE
                            )
                            if error_match:
                                resolved_query = re.sub(
                                    pattern,
                                    error_match.group(0),
                                    resolved_query,
                                    flags=re.IGNORECASE
                                )
                                break
        
        if resolved_query != query:
            logger.debug(f"Reference resolved: '{query}' → '{resolved_query}'")
        
        return resolved_query
    
    def build_conversation_prompt(
        self,
        session: ConversationSession,
        current_query: str,
        max_history_turns: int = 6
    ) -> str:
        """
        Build a prompt that includes conversation history
        
        Args:
            session: Conversation session
            current_query: Current user query
            max_history_turns: Maximum history turns to include
            
        Returns:
            Prompt with conversation context
        """
        parts = []
        
        # Add context summary
        context = session.get_context_summary()
        if context:
            parts.append(f"[Conversation Context]\n{context}\n")
        
        # Add recent history
        history = session.get_history(max_history_turns)
        if history:
            parts.append("[Previous Conversation]")
            for turn in history[:-1]:  # Exclude current if already added
                role_label = "User" if turn["role"] == "user" else "Assistant"
                parts.append(f"{role_label}: {turn['content'][:500]}")
            parts.append("")
        
        # Add current query
        parts.append(f"[Current Question]\n{current_query}")
        
        return "\n".join(parts)
    
    def _cleanup_old_sessions(self):
        """Remove expired sessions and enforce max limit"""
        now = datetime.now()
        
        # Remove expired
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_activity > self.session_timeout
        ]
        
        for sid in expired:
            del self.sessions[sid]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        
        # Enforce max limit (remove oldest)
        while len(self.sessions) > self.max_sessions:
            oldest_id = next(iter(self.sessions))
            del self.sessions[oldest_id]
            logger.debug(f"Removed oldest session: {oldest_id}")
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session"""
        with self._session_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
            return False
    
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all sessions for a user"""
        with self._session_lock:
            return [
                session.to_dict()
                for session in self.sessions.values()
                if session.user_id == user_id
            ]
    
    def get_stats(self) -> Dict:
        """Get conversation manager statistics"""
        with self._session_lock:
            total = len(self.sessions)
            if total == 0:
                return {
                    "total_sessions": 0,
                    "avg_turns": 0,
                    "active_users": 0
                }
            
            total_turns = sum(len(s.turns) for s in self.sessions.values())
            unique_users = len(set(s.user_id for s in self.sessions.values()))
            
            return {
                "total_sessions": total,
                "avg_turns": total_turns / total,
                "active_users": unique_users,
                "max_sessions": self.max_sessions,
                "session_timeout_minutes": self.session_timeout.total_seconds() / 60
            }


# Global instance getter
_manager_instance: Optional[ConversationManager] = None

def get_conversation_manager() -> ConversationManager:
    """Get or create the global conversation manager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ConversationManager()
    return _manager_instance
