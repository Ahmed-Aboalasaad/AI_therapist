import logging
from typing import Dict, Any, List, Optional

from app.core.constants import LANG_MAP
from app.services.language_service import LanguageDetector
from app.services.translation_service import TranslationService
from app.services.emotion_service import EmotionService
from app.services.intent_service import IntentService
from app.services.session_service import SessionService
from app.services.prompt_service import PromptService
from app.services.inference_service import InferenceService
from app.repositories.session_repository import SessionRepository
from app.repositories.qdrant_repository import QdrantRepository
from app.rag.retriever import retrieve_context
from app.rag.embeddings import load_embedding_model

logger = logging.getLogger(__name__)

class ChatService:
    """
    Workflow coordinator service. Coordinates other services to process user queries.
    Preloads all model weights (SVM, Emotion, and Embeddings) on initialization to avoid lazy load delay.
    """
    def __init__(self):
        logger.info("Initializing ChatService components...")
        
        # 1. Load Language SVM Detector
        logger.info("Loading Language Detector model...")
        self.language_service = LanguageDetector()
        
        # 2. Load Translation Service
        self.translation_service = TranslationService()
        
        # 3. Load HuggingFace Emotion classification model
        logger.info("Loading Emotion classifier model...")
        self.emotion_service = EmotionService()
        
        # 4. Load Intent classification service
        self.intent_service = IntentService()
        
        # 5. Initialize session history repository and service
        self.session_repository = SessionRepository()
        self.session_service = SessionService(self.session_repository)
        
        # 6. Initialize and connect Qdrant client
        logger.info("Connecting Qdrant client...")
        self.qdrant_repository = QdrantRepository()
        _ = self.qdrant_repository.client # Accessing property triggers connection establishment
        
        # 7. Preload local Sentence-Transformer Embedding model
        logger.info("Loading Embedding model...")
        load_embedding_model()
        
        # 8. Load Prompt and LLM inference services
        self.prompt_service = PromptService()
        self.inference_service = InferenceService()
        
        logger.info("All services and models successfully preloaded!")

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

        # 1. Language Detection
        logger.info("Detecting language")
        try:
            lang = self.language_service.predict([query])
            detected_code = lang[0]
        except Exception as e:
            logger.error(f"Couldn't detect the language: {str(e)}")
            detected_code = "en"

        try:
            language = LANG_MAP[detected_code.lower()]
        except KeyError:
            logger.warning(f"Language code '{detected_code}' not found in LANG_MAP. Defaulting to English.")
            language = "English"

        # 2. Translation to English
        if language != "English":
            logger.info(f"Translating from {language}: '{query}'")
            english_query = self.translation_service.translate_to_english(query, language)
            logger.info(f"To English: {english_query}")
        else:
            english_query = query

        # 3. Emotion Classification
        emotion_res = self.emotion_service.predict(english_query)
        emotion = emotion_res["emotion"]
        emotion_confidence = emotion_res["confidence"]
        logger.info(f"Emotion: {emotion} (confident: {emotion_confidence:.2f}%)")

        # 4. Intent Classification
        intent_res = self.intent_service.classify_user_intent(english_query)
        intent = intent_res.intent
        logger.info(f"Intent: {intent}")

        # 5. Retrieve RAG Context (Mental Health Questions only)
        references = []
        context = ""
        if intent == "asking_mental_health_question":
            logger.info("Running RAG context retrieval")
            context, docs = retrieve_context(english_query, self.qdrant_repository)
            references = [
                {
                    "snippet": doc.page_content,
                    "answer_source": doc.metadata.get("answers", "")
                }
                for doc in docs
            ]

        # 6. Dialogue Chat History Assembly
        formatted_history = self.session_service.get_formatted_history(session_id)

        # 7. Construct Full Prompt (delegated to PromptService)
        constructed_prompt = self.prompt_service.build_prompt(
            intent=intent,
            query=query,
            emotion=emotion,
            language=language,
            chat_history=formatted_history,
            context=context if intent == "asking_mental_health_question" else None
        )

        # 8. LLM Invocation & Response Generation (delegated to InferenceService)
        response = self.inference_service.generate_response(constructed_prompt)

        # 9. Save turn to Session Service history
        self.session_service.add_user_message(session_id, query)
        self.session_service.add_ai_message(session_id, response)

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

    @property
    def session_store(self):
        """
        Exposes the session store dictionary for backward compatibility.
        """
        return self.session_repository._store

# Alias for backward compatibility
QueryHandler = ChatService
