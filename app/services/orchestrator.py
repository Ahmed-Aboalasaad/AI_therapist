import logging
from typing import Dict, Any, List, Optional

from modules.translation.translator import GroqTranslator
from modules.emotion_classification.emotion_classifier import EmotionClassifier
from modules.intent_classification.intent_classifier import classify_user_intent
from modules.rag.chains import get_mental_health_chain, retrieve_context
from modules.language_detection.language_detector import LanguageDetector
from langchain_core.chat_history import InMemoryChatMessageHistory

LANG_MAP = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Chinese"
}

logger = logging.getLogger(__name__)

def format_chat_history(chat_history: Optional[List[Dict[str, str]]]) -> str:
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


class QueryHandler:

    def __init__(self):
        logger.info("Initializing QueryHandler...")

        self.translator = GroqTranslator()
        self.emotion_classifier = EmotionClassifier()
        self.language_detector = LanguageDetector()
        self.session_store = {}

    def process_query(self, query: str, session_id: str = "default-session") -> Dict[str, Any]:

        if not query.strip():
            return {
                "response": "Please enter a message.",
                "language": "English",
                "intent": "out_of_scope",
                "emotion": "neutral",
                "emotion_confidence": 1.0,
                "english_query": "",
                "english_response": "Please enter a message.",
                "references": []
            }

        # --------------------------------------------------
        # Step 1.1: Detect Language
        # --------------------------------------------------
        logger.info(f"Detecting language")
        try:
            lang = self.language_detector.predict([query])
            detected_code = lang[0]
        except Exception as e:
            logger.error(f"Couldn't detect the language: {str(e)}")
            detected_code = "en"

        try:
            language = LANG_MAP[detected_code.lower()]
        except KeyError:
            logger.warning(f"Language code '{detected_code}' not found in LANG_MAP. Defaulting to English.")
            language = "English"

        # --------------------------------------------------
        # Step 1.2: Translate to English (using translator)
        # --------------------------------------------------
        if language != "English":
            logger.info(f"Translating from {language}: '{query}'")
            english_query = self.translator.translate_to_english(query, language)
            logger.info(f"To English: {english_query}")
        else:
            english_query = query

        # --------------------------------------------------
        # Step 2: Emotion Classification
        # --------------------------------------------------
        emotion_res = self.emotion_classifier.predict(english_query)
        emotion = emotion_res["emotion"]
        emotion_confidence = emotion_res["confidence"]
        logger.info(f"Emotion: {emotion} (confident: {emotion_confidence:.2f}%)")

        # --------------------------------------------------
        # Step 3: Intent Classification
        # --------------------------------------------------
        intent = classify_user_intent(english_query).intent
        logger.info(f"Intent: {intent}")

        # --------------------------------------------------
        # Step 4: Retrieve RAG Context (Mental Health Questions only)
        # --------------------------------------------------
        references = []
        context = ""
        if intent == "asking_mental_health_question":
            logger.info("Running RAG context retrieval")
            context, docs = retrieve_context(english_query)
            references = [
                {
                    "snippet": doc.page_content,
                    "answer_source": doc.metadata.get("answers", "")
                }
                for doc in docs
            ]

        # --------------------------------------------------
        # Step 5: Format Dialogue Chat History
        # --------------------------------------------------
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        history = self.session_store[session_id]

        chat_history_dicts = []
        for msg in history.messages:
            role = "user" if msg.type == "human" else "assistant"
            chat_history_dicts.append({"role": role, "content": msg.content})

        formatted_history = format_chat_history(chat_history_dicts)

        # --------------------------------------------------
        # Step 6: Invoke LLM Chain
        # --------------------------------------------------
        input_data = {
            "prompt": query,
            "emotion": emotion,
            "language": language,
            "chat_history": formatted_history,
            "intent": intent
        }
        if intent == "asking_mental_health_question":
            input_data["context"] = context

        chain = get_mental_health_chain(intent)
        response = chain.invoke(input_data)

        # Save turn to LangChain session history
        history.add_user_message(query)
        history.add_ai_message(response)

        return {
            "response": response,
            "language": language,
            "intent": intent,
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
            "english_query": english_query,
            "english_response": response,
            "references": references
        }