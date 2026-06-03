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
    from app.services.orchestrator import QueryHandler
    from modules.intent_classification.intent_classifier import IntentResponse

def test_orchestrator_flow():
    # 1. Intent classifier mock response
    mock_intent_response = IntentResponse(
        intent="asking_mental_health_question",
        confidence=0.95
    )
    
    # 2. LLM response mock
    mock_llm_response = "Arabic: I am so sorry you are feeling sad and anxious. Here is support..."
    
    # Mock documents returned by retriever
    mock_docs = [
        Document(page_content="Expert advice context", metadata={"answers": "expert answer"})
    ]

    with patch('app.services.orchestrator.LanguageDetector.predict') as mock_language_predict, \
         patch('app.services.orchestrator.GroqTranslator.translate_to_english') as mock_translate_to_english, \
         patch('modules.intent_classification.intent_classifier.client.chat.completions.create') as mock_intent_client_create, \
         patch('app.services.orchestrator.retrieve_context') as mock_retrieve_context, \
         patch('app.services.orchestrator.get_mental_health_chain') as mock_get_mental_health_chain:
         
         # Configure mocks
         mock_language_predict.return_value = ["ar"]
         mock_translate_to_english.return_value = "I feel very sad and anxious."
         
         # intent_classifier.client.chat.completions.create is called once
         mock_intent_client_create.return_value = mock_intent_response
         
         mock_retrieve_context.return_value = ("Mock formatted context", mock_docs)
         
         # Mock LLM chain
         mock_chain = MagicMock()
         mock_chain.invoke.return_value = mock_llm_response
         mock_get_mental_health_chain.return_value = mock_chain
         
         # Initialize orchestrator
         orchestrator = QueryHandler()
         
         # Pre-populate history for session
         from langchain_core.chat_history import InMemoryChatMessageHistory
         session_id = "test-session-123"
         orchestrator.session_store[session_id] = InMemoryChatMessageHistory()
         orchestrator.session_store[session_id].add_user_message("hello")
         
         # Run the query
         result = orchestrator.process_query("أنا حزين وقلق جدا", session_id=session_id)
         
         print("\n================ TEST RESULTS ================")
         print(f"Original Query:          أنا حزين وقلق جدا")
         print(f"Language:       {result['language']}")
         print(f"English Query Translation: {result['english_query']}")
         print(f"Detected Emotion:        {result['emotion']} (confidence: {result['emotion_confidence']})")
         print(f"Detected Intent:         {result['intent']}")
         print(f"English response:        {result['english_response']}")
         print(f"Final response:          {result['response']}")
         print("==============================================\n")
         
         assert result['language'] == "Arabic"
         assert result['intent'] == "asking_mental_health_question"
         assert result['emotion'] == "sadness" # Classified by local emotion model
         assert result['response'] == mock_llm_response
         assert len(result['references']) == 1
         assert result['references'][0]['snippet'] == "Expert advice context"
         
         # Verify that get_mental_health_chain was called with correct intent and invoke was called with chat history
         mock_get_mental_health_chain.assert_called_once_with("asking_mental_health_question")
         mock_chain.invoke.assert_called_once()
         
         called_args = mock_chain.invoke.call_args[0][0]
         assert called_args["chat_history"] == "Patient: hello\n"
         assert called_args["language"] == "Arabic"
         assert called_args["emotion"] == "sadness"
         
         # Verify history was updated with current turn
         messages = orchestrator.session_store[session_id].messages
         assert len(messages) == 3
         assert messages[1].content == "أنا حزين وقلق جدا"
         assert messages[2].content == mock_llm_response
         
         print("End-to-end orchestrator flow test passed successfully!")

if __name__ == "__main__":
    test_orchestrator_flow()
