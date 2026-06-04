from typing import Dict, Any, List, Optional

def format_chat_history(chat_history: Optional[List[Dict[str, str]]]) -> str:
    """
    Formats dialogue history list of dicts into a single clean string.
    """
    if not chat_history:
        return ""
    formatted = []
    for msg in chat_history:
        role = msg.get("role", msg.get("sender", "user")).lower()
        content = msg.get("content", msg.get("text", ""))
        if not content:
            continue
        if role in ["user", "patient", "human"]:
            formatted.append(f"Patient: {content}")
        else:
            formatted.append(f"AI Counselor: {content}")
    return "\n".join(formatted) + "\n"
