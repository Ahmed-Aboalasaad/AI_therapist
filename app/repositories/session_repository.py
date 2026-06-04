from typing import Dict
from langchain_core.chat_history import InMemoryChatMessageHistory

class SessionRepository:
    """
    In-memory session repository implementation using LangChain's InMemoryChatMessageHistory.
    """
    def __init__(self):
        self._store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """
        Retrieves or initializes the chat history for a session ID.
        """
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]
        
    def add_user_message(self, session_id: str, message: str) -> None:
        """
        Adds a user message to the session history.
        """
        history = self.get_session_history(session_id)
        history.add_user_message(message)

    def add_ai_message(self, session_id: str, message: str) -> None:
        """
        Adds an AI message to the session history.
        """
        history = self.get_session_history(session_id)
        history.add_ai_message(message)
        
    def clear_session(self, session_id: str) -> None:
        """
        Clears the session history.
        """
        if session_id in self._store:
            self._store[session_id].clear()
