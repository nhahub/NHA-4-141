import os
import requests
from typing import Optional


class OllamaClient:

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.chat_model = "qwen2.5-coder:1.5b"
        self.coder_model = "qwen2.5-coder:1.5b"
        self.api_url = f"{self.base_url}/api/generate"

    def is_code_request(self, question: str) -> bool:
        code_keywords = [
            "write", "code", "function", "generate", "create",
            "explain", "debug", "fix", "script", "program",
            "class", "loop", "algorithm", "implement", "error",
            "python", "java", "javascript", "sql", "html", "css",
            "اكتب", "كود", "برنامج", "دالة", "شرح", "اشرح"
        ]
        return any(kw in question.lower() for kw in code_keywords)

    def generate_response(self, question: str, context: str) -> str:
        try:
            if not self.check_connection():
                return self._get_fallback_response()

            if self.is_code_request(question):
                model = self.coder_model
                prompt = self._create_code_prompt(question, context)
            else:
                model = self.chat_model
                prompt = self._create_rag_prompt(question, context)

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 2000,
                    "num_gpu": 0        # Force CPU — avoids CUDA crash
                }
            }

            response = requests.post(self.api_url, json=payload, timeout=300)

            if response.status_code == 200:
                return response.json().get("response", "Sorry, I couldn't generate a response.")
            else:
                return f"Error: Ollama API returned status code {response.status_code}: {response.text}"

        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. Try a simpler question or wait for the model to warm up."
        except requests.exceptions.RequestException as e:
            return f"Connection error: {str(e)}. Make sure Ollama is running."
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def _create_rag_prompt(self, question: str, context: str) -> str:
        return f"""You are a helpful AI assistant. Answer the question using the context below.
If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    def _create_code_prompt(self, question: str, context: str) -> str:
        context_section = f"\nRelevant context:\n{context}\n" if context.strip() else ""
        return f"""You are an expert programming assistant. Write clean working code with comments.

Rules:
- Wrap ALL code in markdown code blocks like ```python
- Add comments explaining key lines  
- After the code, briefly explain how it works
{context_section}
Request: {question}

Response:"""

    def _get_fallback_response(self) -> str:
        return "Cannot connect to Ollama. Please make sure it is running with 'ollama serve'."

    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> list:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                return [m["name"] for m in response.json().get("models", [])]
            return []
        except:
            return []