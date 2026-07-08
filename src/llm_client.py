import os
import requests
from typing import Optional


class OllamaClient:
    """Client for interacting with Ollama LLM for response generation."""

    def __init__(self, model: Optional[str] = None, role: str = "chat"):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        self.role = role
        self.api_url = f"{self.base_url}/api/generate"

    def _call_ollama(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the response text."""
        try:
            if not self.check_connection():
                return self._get_fallback_response()

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1000,
                    "num_gpu": 0
                }
            }

            response = requests.post(self.api_url, json=payload, timeout=180)

            if response.status_code == 200:
                return response.json().get("response", "Sorry, I could not generate a response.")
            else:
                return f"Error: Ollama API returned status code {response.status_code}"

        except requests.exceptions.Timeout:
            return "Request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def generate_response(self, question: str, context: str, chat_history: str = "") -> str:
        """Generate a RAG or code response depending on this client's role."""
        if self.role == "code":
            prompt = self._create_code_prompt(question, context, chat_history)
        else:
            prompt = self._create_rag_prompt(question, context, chat_history)
        return self._call_ollama(prompt)

    def generate_web_search_response(self, question: str, context: str, chat_history: str = "") -> str:
        """
        Dedicated method for web search responses.
        Uses a prompt that tells the model to answer from web results directly,
        NOT to say the context doesn't contain the answer.
        """
        history_block = ""
        if chat_history.strip():
            history_block = f"\n\nRecent conversation:\n{chat_history}\n"

        prompt = (
            "You are a helpful AI assistant with access to live web search results.\n"
            "Answer the user's question using the web search results provided below.\n"
            "Be direct, informative, and summarize the key points.\n"
            "Always use the web results to answer — do NOT say the information is unavailable.\n"
            + history_block
            + "\nWeb Search Results:\n"
            + context
            + "\n\nQuestion: " + question
            + "\n\nAnswer:"
        )

        return self._call_ollama(prompt)

    def _create_rag_prompt(self, question: str, context: str, chat_history: str = "") -> str:
        history_block = ""
        if chat_history.strip():
            history_block = (
                "\n\nRecent conversation (for context only):\n"
                + chat_history
                + "\n"
            )

        prompt = (
            "You are a helpful AI assistant that answers questions based on the provided context.\n"
            "Use the following context to answer the user's question.\n"
            "If the answer cannot be found in the context, say so clearly.\n"
            + history_block
            + "\nContext:\n"
            + context
            + "\n\nQuestion: " + question
            + "\n\nAnswer: Based on the provided context, "
        )

        return prompt

    def _create_code_prompt(self, question: str, context: str, chat_history: str = "") -> str:
        context_block = ""
        if context.strip():
            context_block = "\n\nRelevant context from the user's documents:\n" + context

        history_block = ""
        if chat_history.strip():
            history_block = "\n\nRecent conversation:\n" + chat_history + "\n"

        prompt = (
            "You are an expert programming assistant.\n"
            "Write clean, working code with comments and a brief explanation.\n"
            "Always wrap code in markdown code blocks like ```python\n"
            + context_block
            + history_block
            + "\n\nQuestion: " + question
            + "\n\nAnswer:"
        )

        return prompt

    def _get_fallback_response(self) -> str:
        return (
            "I cannot connect to the local AI model (Ollama).\n\n"
            "Please make sure:\n"
            "1. Ollama is installed and running\n"
            "2. Run 'ollama serve' in your terminal\n"
            f"3. The model is downloaded: 'ollama pull {self.model}'"
        )

    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                return [m["name"] for m in response.json().get("models", [])]
            return []
        except Exception:
            return []