import sys
import os

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dummy environment variables to avoid loading heavy modules / errors
os.environ["GROQ_API_KEY"] = "dummy_groq_key"
os.environ["QDRANT_URL"] = ":memory:"
os.environ["COLLECTION_NAME"] = "test-collection"
os.environ["MODEL_LOCAL_PATH"] = "/home/sharaf/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
os.environ["HUGGINGFACE_HUB_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["VECTOR_SIZE"] = "384"
os.environ["CLEANED_DATA_PATH"] = "./data/cleaned_mental_health_data.csv"

# Mock/patch necessary components before importing main app
from unittest.mock import MagicMock, patch

mock_embeddings = MagicMock()
mock_embeddings.embed_documents.return_value = [[0.1] * 384]
mock_embeddings.embed_query.return_value = [0.1] * 384

with patch('modules.rag.helpers.data_helpers.load_embedding_model', return_value=mock_embeddings):
    from fastapi.testclient import TestClient
    from main import app

client = TestClient(app)

def test_language_detection_api():
    print("Testing language detection API endpoint...")
    
    # 1. Test English
    response = client.post("/api/test/language-detection", json={"text": "Hello this is a test"})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    data = response.json()
    assert data["language"] == "en", f"Expected 'en', got {data['language']}"
    print("English detection: PASS")

    # 2. Test Arabic
    response = client.post("/api/test/language-detection", json={"text": "أنا حزين وأشعر بالقلق"})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    data = response.json()
    assert data["language"] == "ar", f"Expected 'ar', got {data['language']}"
    print("Arabic detection: PASS")

    # 3. Test Spanish
    response = client.post("/api/test/language-detection", json={"text": "Hola a todos, cómo están?"})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    data = response.json()
    assert data["language"] == "es", f"Expected 'es', got {data['language']}"
    print("Spanish detection: PASS")

if __name__ == "__main__":
    test_language_detection_api()
    print("All language detection API tests passed successfully!")
