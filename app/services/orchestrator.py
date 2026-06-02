import logging
from typing import Dict, Any

from . import *
from modules.translation.translator import GroqTranslator
from modules.intent_classification.intent_classifier import classify_user_intent
from modules.emotion_classification.classifier import EmotionClassifier
from modules.rag.chains import get_RAG_chain
from modules.language_detection.language_detector import LanguageDetector


logger = logging.getLogger(__name__)


class QueryHandler:
    def __init__(self):
        logger.info("Initializing QueryHandler...")

        self.translator = GroqTranslator()
        self.emotion_classifier = EmotionClassifier()
        self.language_detector = LanguageDetector()

        # Single RAG chain (Retriever + Prompt + LLM)
        self.RAG_chain = get_RAG_chain()

    def process_query(self, query: str) -> Dict[str, Any]:
        if not query.strip():
            return {
                "response": "Please enter a message.",
                "language": "English",
                "intent": "out_of_scope",
                "emotion": "neutral",
                "confidence": 1.0
            }

        # --------------------------------------------------
        # Step 1.1: Detect Language
        # --------------------------------------------------
        logger.info(
            f"Detecting language"
        )
        try:
            lang = self.language_detector.predict([query])
            detected_code = lang[0]
        except Exception as e:
            logger.error(f"Couldn't detect the language: {str(e)}")

        try:
            language = LANG_MAP[detected_code.lower()]
        except KeyError:
            logger.warning(f"Language code '{detected_code}' not found in LANG_MAP. Defaulting to English.")
            language = "English"


        # --------------------------------------------------
        # Step 1.2: Translate to English (using translator only)
        # --------------------------------------------------
        if language != "English":
            logger.info(f"Translating from {language}: '{query}'")
            query = self.translator.translate_to_english(query, language)
            logger.info(f"To English: {query}")


        # --------------------------------------------------
        # Step 2: Emotion Classification
        # --------------------------------------------------
        emotion_res = self.emotion_classifier.predict(query)
        emotion = emotion_res["emotion"]
        emotion_confidence = emotion_res["confidence"]
        logger.log(f"Emotion: {emotion} (confident: {emotion_confidence:.2f})%")


        # --------------------------------------------------
        # Step 3: Intent Classification
        # --------------------------------------------------
        intent = classify_user_intent(query).intent
        logger.info(f"Intent: {intent}")


        # --------------------------------------------------
        # Step 4: Route Request
        # --------------------------------------------------
        if intent == "asking_mental_health_question":
            logger.info("Running Mental Health RAG Chain")
            response = self.RAG_chain.invoke(query) # Always in English
        else:
            try:
                response = STATIC_RESPONSES[intent]
            except KeyError as e:
                logger.error(f"Intent '{intent}' not found in STATIC_RESPONSES. Defaulting to 'out_of_scope' response.")
                response = STATIC_RESPONSES["out_of_scope"]


        # --------------------------------------------------
        # Step 5: Translate Back
        # --------------------------------------------------
        final_response = response
        if language != "English":
            logger.info(f"Translating response back to {language}")
            final_response = self.translator.translate_back(response, language)


        return {
            "response": final_response,
            "language": language,
            "intent": intent,
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
            "english_query": query,
            "english_response": response
        }