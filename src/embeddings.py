import os
import requests
from typing import List


class EmbeddingClient:

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Use the model you already have pulled
        self.model = os.getenv("EMBEDDING_MODEL", "yxchia/multilingual-e5-base")
        self.api_url = f"{self.base_url}/api/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        if not self._check_ollama_connection():
            print("Warning: Ollama not available. Using fallback embeddings.")
            return [[0.0] * 768 for _ in texts]
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding if embedding else [0.0] * 768)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            if not self._check_ollama_connection():
                return [0.0] * 768
            payload = {"model": self.model, "prompt": text}
            response = requests.post(self.api_url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("embedding", [0.0] * 768)
            else:
                print(f"Embedding API error: {response.status_code} - {response.text}")
                return [0.0] * 768
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 768

    def _check_ollama_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False