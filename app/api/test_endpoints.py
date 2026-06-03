from fastapi import APIRouter, HTTPException
from app.schemas.testing import (
    TranslationTestRequest, TranslationTestResponse,
    TranslateBackTestRequest, TranslateBackTestResponse,
    IntentTestRequest, IntentTestResponse,
    EmotionTestRequest, EmotionTestResponse,
    RagTestRequest, RagTestResponse,
    LanguageDetectionTestRequest, LanguageDetectionTestResponse,
    LlmTestRequest, LlmTestResponse
)
from modules.translation.translator import GroqTranslator
from modules.intent_classification.intent_classifier import classify_user_intent
from modules.emotion_classification.emotion_classifier import EmotionClassifier
from modules.rag.database import get_retriever
from modules.rag.chains import get_mental_health_chain
from modules.language_detection.language_detector import LanguageDetector

router = APIRouter(
    prefix="/api/test",
    tags=["Testing Services"]
)

# Instantiate the service objects
try:
    translator = GroqTranslator()
except Exception as e:
    print(f"Error loading GroqTranslator: {e}")
    translator = None

try:
    emotion_classifier = EmotionClassifier()
except Exception as e:
    print(f"Error loading EmotionClassifier: {e}")
    emotion_classifier = None

try:
    retriever = get_retriever()
except Exception as e:
    print(f"Error loading Qdrant retriever: {e}")
    retriever = None

try:
    language_detector = LanguageDetector()
except Exception as e:
    print(f"Error loading LanguageDetector: {e}")
    language_detector = None

@router.post("/translate", response_model=TranslationTestResponse)
async def test_translate(request: TranslationTestRequest):
    if not translator:
        raise HTTPException(status_code=500, detail="Translator is not initialized. Check GROQ_API_KEY.")
    try:
        res = translator.detect_and_translate(request.text)
        return TranslationTestResponse(
            source_language=res.source_language,
            translated_text=res.translated_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@router.post("/translate-back", response_model=TranslateBackTestResponse)
async def test_translate_back(request: TranslateBackTestRequest):
    if not translator:
        raise HTTPException(status_code=500, detail="Translator is not initialized. Check GROQ_API_KEY.")
    try:
        translated = translator.translate_back(request.text, request.target_language)
        return TranslateBackTestResponse(
            translated_text=translated
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation back failed: {str(e)}")

@router.post("/intent", response_model=IntentTestResponse)
async def test_intent(request: IntentTestRequest):
    try:
        res = classify_user_intent(request.text)
        # Handle cases where confidence is not present in instructor output
        confidence = getattr(res, "confidence", 1.0)
        return IntentTestResponse(
            intent=res.intent,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intent classification failed: {str(e)}")

@router.post("/emotion", response_model=EmotionTestResponse)
async def test_emotion(request: EmotionTestRequest):
    if not emotion_classifier:
        raise HTTPException(status_code=500, detail="Emotion classifier is not initialized.")
    try:
        res = emotion_classifier.predict(request.text)
        return EmotionTestResponse(
            emotion=res["emotion"],
            confidence=res["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emotion classification failed: {str(e)}")

@router.post("/rag", response_model=RagTestResponse)
async def test_rag(request: RagTestRequest):
    if not retriever:
        raise HTTPException(status_code=500, detail="RAG retriever is not initialized. Check QDRANT_URL.")
    try:
        docs = retriever.invoke(request.text)
        retrieved_contexts = []
        for doc in docs:
            retrieved_contexts.append({
                "snippet": doc.page_content,
                "answer_source": doc.metadata.get("answers", "")
            })
        return RagTestResponse(
            query=request.text,
            retrieved_contexts=retrieved_contexts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {str(e)}")

@router.post("/language-detection", response_model=LanguageDetectionTestResponse)
async def test_language_detection(request: LanguageDetectionTestRequest):
    if not language_detector:
        raise HTTPException(status_code=500, detail="Language detector is not initialized.")
    try:
        res = language_detector.predict([request.text])
        detected_lang = res[0] if len(res) > 0 else "unknown"
        return LanguageDetectionTestResponse(
            language=str(detected_lang)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")

@router.post("/llm", response_model=LlmTestResponse)
async def test_llm(request: LlmTestRequest):
    try:
        from app.services.orchestrator import format_chat_history
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from app.api.chat import query_handler
        
        if query_handler and request.session_id in query_handler.session_store:
            history = query_handler.session_store[request.session_id]
        else:
            history = InMemoryChatMessageHistory()

        chat_history_dicts = []
        for msg in history.messages:
            role = "user" if msg.type == "human" else "assistant"
            chat_history_dicts.append({"role": role, "content": msg.content})

        formatted_history = format_chat_history(chat_history_dicts)

        chain = get_mental_health_chain(request.intent)
        input_data = {
            "prompt": request.prompt,
            "emotion": request.emotion,
            "language": request.language or "English",
            "chat_history": formatted_history
        }
        if request.intent == "asking_mental_health_question":
            input_data["context"] = request.context or "No clinical context is required for this interaction."

        res = chain.invoke(input_data)
        return LlmTestResponse(response=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM chain invocation failed: {str(e)}")
