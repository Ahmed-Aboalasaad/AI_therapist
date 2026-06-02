import logging
from typing import Dict, Any

from modules.translation.translator import GroqTranslator
from modules.intent_classifier.intent_classifier import classify_user_intent
from modules.emotion_classifier.classifier import EmotionClassifier
from modules.rag.chains import get_mental_health_chain

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
        # Step 1: Detect Language + Translate to English
        # --------------------------------------------------
        logger.info(
            f"Detecting language and translating: '{user_query}'"
        )

        translation_res = self.translator.detect_and_translate(
            user_query
        )

        source_lang = translation_res.source_language
        english_query = translation_res.translated_text

        logger.info(
            f"Language: {source_lang} | Query: {english_query}"
        )

        # --------------------------------------------------
        # Step 2: Emotion Classification
        # --------------------------------------------------
        emotion_res = self.emotion_classifier.predict(
            english_query
        )

        emotion = emotion_res["emotion"]
        emotion_confidence = emotion_res["confidence"]

        # --------------------------------------------------
        # Step 3: Intent Classification
        # --------------------------------------------------
        intent_res = classify_user_intent(
            english_query
        )
        # print(intent_res)
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