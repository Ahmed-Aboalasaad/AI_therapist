from app.repositories.session_repository import SessionRepository
from app.utils.helpers import format_chat_history

class SessionService:
    """
    Service layer wrapping around SessionRepository to provide high-level dialogue management.
    """
    def __init__(self, session_repository: SessionRepository):
        self.repository = session_repository

    def get_formatted_history(self, session_id: str) -> str:
        """
        Retrieves conversational history and formats it as a string.
        """
        history = self.repository.get_session_history(session_id)
        chat_history_dicts = []
        for msg in history.messages:
            role = "user" if msg.type == "human" else "assistant"
            chat_history_dicts.append({"role": role, "content": msg.content})
        return format_chat_history(chat_history_dicts)

    def add_user_message(self, session_id: str, message: str):
        self.repository.add_user_message(session_id, message)

    def add_ai_message(self, session_id: str, message: str):
        self.repository.add_ai_message(session_id, message)
        
    def get_raw_history(self, session_id: str):
        return self.repository.get_session_history(session_id)
