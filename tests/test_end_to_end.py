import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dummy environment variables before importing modules
os.environ["GROQ_API_KEY"] = "dummy_groq_key"
os.environ["QDRANT_URL"] = ":memory:" # Using in-memory Qdrant for tests
os.environ["COLLECTION_NAME"] = "test-collection"
os.environ["MODEL_LOCAL_PATH"] = "/home/sharaf/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
os.environ["HUGGINGFACE_HUB_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["VECTOR_SIZE"] = "384"
os.environ["CLEANED_DATA_PATH"] = "./data/cleaned_mental_health_data.csv"

# Mock HuggingFaceEmbeddings to avoid downloading/loading sentence-transformers during tests
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

mock_embeddings = MagicMock(spec=HuggingFaceEmbeddings)
mock_embeddings.embed_documents.return_value = [[0.1] * 384]
mock_embeddings.embed_query.return_value = [0.1] * 384

# Patch the load_embedding_model helper to return our mock
with patch('modules.rag.helpers.data_helpers.load_embedding_model', return_value=mock_embeddings):
    from app.services.orchestrator import FlowOrchestrator
    from modules.translation.translator import TranslationResponse
    from modules.intent_classifier.intent_classifier import IntentResponse

def test_orchestrator_flow():
    # 1. Translator detect_and_translate mock response
    mock_trans_response = TranslationResponse(
        source_language="Arabic",
        translated_text="I feel very sad and anxious."
    )
    
    # 2. Intent classifier mock response
    mock_intent_response = IntentResponse(
        intent="asking_mental_health_question",
        confidence=0.95
    )
    
    # 3. ChatGroq response mock
    mock_llm_response = "I am so sorry you are feeling sad and anxious. Here are some coping strategies..."
    
    # 4. Translator translate_back mock response
    mock_translate_back_response = "أنا آسف جدا لأنك تشعر بالحزن والقلق. إليك بعض الاستراتيجيات..."

    with patch('app.services.orchestrator.GroqTranslator.detect_and_translate') as mock_detect_and_translate, \
         patch('app.services.orchestrator.GroqTranslator.translate_back') as mock_translate_back, \
         patch('modules.intent_classifier.intent_classifier.client.chat.completions.create') as mock_intent_client_create, \
         patch('app.services.orchestrator.ChatGroq') as mock_chat_groq:
         
         # Configure mocks
         mock_detect_and_translate.return_value = mock_trans_response
         mock_translate_back.return_value = mock_translate_back_response
         
         # intent_classifier.client.chat.completions.create is called once
         mock_intent_client_create.return_value = mock_intent_response
         
         # ChatGroq will return mock_llm_response on invoke
         mock_llm_instance = MagicMock()
         mock_llm_instance.invoke.return_value = mock_llm_response
         mock_chat_groq.return_value = mock_llm_instance
         
         # Initialize orchestrator
         orchestrator = FlowOrchestrator()
         
         # Run the query
         result = orchestrator.process_query("أنا حزين وقلق جدا")
         
         print("\n================ TEST RESULTS ================")
         print(f"Original Query:          أنا حزين وقلق جدا")
         print(f"Detected Language:       {result['detected_language']}")
         print(f"English Query Translation: {result['english_query']}")
         print(f"Detected Emotion:        {result['emotion']} (confidence: {result['emotion_confidence']})")
         print(f"Detected Intent:         {result['intent']}")
         print(f"English response:        {result['english_response']}")
         print(f"Final translated response: {result['response']}")
         print("==============================================\n")
         
         assert result['detected_language'] == "Arabic"
         assert result['intent'] == "asking_mental_health_question"
         assert result['emotion'] == "sadness" # Classified by local emotion model
         assert result['response'] == mock_translate_back_response
         print("End-to-end orchestrator flow test passed successfully!")

if __name__ == "__main__":
    test_orchestrator_flow()
