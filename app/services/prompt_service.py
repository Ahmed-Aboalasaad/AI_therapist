from typing import Optional
from app.rag.prompts import MENTAL_HEALTH_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT

class PromptService:
    """
    Service responsible ONLY for selecting prompt templates and formatting them.
    Returns a fully constructed prompt string.
    """
    def select_prompt_template(self, intent: str) -> str:
        """
        Selects the appropriate prompt template based on intent.
        """
        if intent == "asking_mental_health_question":
            return MENTAL_HEALTH_SYSTEM_PROMPT
        return GENERAL_SYSTEM_PROMPT

    def build_prompt(
        self,
        intent: str,
        query: str,
        emotion: str,
        language: str,
        chat_history: str,
        context: Optional[str] = None
    ) -> str:
        """
        Constructs and returns the fully formatted prompt string.
        """
        template = self.select_prompt_template(intent)
        
        if intent == "asking_mental_health_question":
            formatted = template.format(
                emotion=emotion,
                language=language,
                context=context or "No clinical context is required for this interaction.",
                chat_history=chat_history,
                prompt=query
            )
        else:
            formatted = template.format(
                intent=intent,
                language=language,
                emotion=emotion,
                chat_history=chat_history,
                prompt=query
            )
            
        return formatted
