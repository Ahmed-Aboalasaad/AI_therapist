from qdrant_client import QdrantClient
from app.core.config import get_settings

class QdrantRepository:
    """
    Abstractions for Qdrant client connections and lower-level collection interactions.
    """
    def __init__(self):
        self.settings = get_settings()
        self._client = None

    @property
    def client(self) -> QdrantClient:
        if self._client is not None:
            return self._client
        
        url = self.settings.QDRANT_URL
        if url.startswith("http://") or url.startswith("https://"):
            self._client = QdrantClient(
                url=url,
                api_key=self.settings.QDRANT_API_KEY
            )
        elif url == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(path=url)
        return self._client
