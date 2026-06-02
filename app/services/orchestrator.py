import logging
from typing import Dict, Any

from modules.translation.translator import GroqTranslator
from modules.intent_classifier.intent_classifier import classify_user_intent
from modules.emotion_classifier.classifier import EmotionClassifier
from modules.rag.chains import get_mental_health_chain
from modules.language_detection.language_detector import LanguageDetector

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


STATIC_RESPONSES = {
    "greeting":
        "Hello! I'm here to support you. How are you feeling today?",

    "goodbye":
        "Take care of yourself. I'm always here if you need support in the future.",

    "gratitude":
        "You're very welcome. I'm glad I could support you today.",

    "out_of_scope":
        "I'm designed to help with mental health and emotional well-being. Feel free to share any thoughts, feelings, stress, or concerns you'd like to discuss."
}


class FlowOrchestrator:

    def __init__(self):
        logger.info("Initializing FlowOrchestrator...")

        self.translator = GroqTranslator()
        self.emotion_classifier = EmotionClassifier()
        self.language_detector = LanguageDetector()

        # Single RAG chain (Retriever + Prompt + LLM)
        self.mental_health_chain = get_mental_health_chain()

    def process_query(self, user_query: str) -> Dict[str, Any]:

        if not user_query.strip():
            return {
                "response": "Please enter a message.",
                "detected_language": "English",
                "intent": "out_of_scope",
                "emotion": "neutral",
                "confidence": 1.0
            }

        # --------------------------------------------------
        # Step 1: Detect Language (using local SVM detector)
        # --------------------------------------------------
        logger.info(
            f"Detecting language for: '{user_query}'"
        )
        pred_langs = self.language_detector.predict([user_query])
        detected_code = pred_langs[0] if len(pred_langs) > 0 else "en"
        source_lang = LANG_MAP.get(detected_code.lower(), detected_code)

        # --------------------------------------------------
        # Step 1b: Translate to English (using translator only)
        # --------------------------------------------------
        logger.info(
            f"Translating {source_lang} to English: '{user_query}'"
        )
        english_query = self.translator.translate_to_english(user_query, source_lang)

        logger.info(
            f"Language: {source_lang} | Query: {english_query}"
        )

        # --------------------------------------------------
        # Step 2: Emotion Classification
        # --------------------------------------------------
        emotion_res = self.emotion_classifier.predict(
            english_query
        )
        #print(emotion_res)
        emotion = emotion_res["emotion"]
        emotion_confidence = emotion_res["confidence"]

        # --------------------------------------------------
        # Step 3: Intent Classification
        # --------------------------------------------------
        intent_res = classify_user_intent(
            english_query
        )
        #print(intent_res)
        intent = intent_res.intent

        logger.info(
            f"Intent: {intent} | Emotion: {emotion}"
        )

        # --------------------------------------------------
        # Step 4: Route Request
        # --------------------------------------------------
        if intent == "asking_mental_health_question":

            logger.info(
                "Running Mental Health RAG Chain"
            )

            english_response = (
                self.mental_health_chain.invoke(
                    english_query
                )
            )

        else:

            english_response = STATIC_RESPONSES.get(
                intent,
                STATIC_RESPONSES["out_of_scope"]
            )

        # --------------------------------------------------
        # Step 5: Translate Back
        # --------------------------------------------------
        final_response = english_response

        if source_lang.lower() != "english":

            logger.info(
                f"Translating response back to {source_lang}"
            )

            final_response = (
                self.translator.translate_back(
                    english_response,
                    source_lang
                )
            )

        # --------------------------------------------------
        # Return Result
        # --------------------------------------------------
        return {
            "response": final_response,
            "detected_language": source_lang,
            "intent": intent,
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
            "english_query": english_query,
            "english_response": english_response
        }