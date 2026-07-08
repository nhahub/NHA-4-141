import os
import base64
import requests


class QwenVisionClient:
    """
    Client for image analysis / OCR via a vision-capable Ollama model
    (e.g. qwen2.5-vl). Uses Ollama's /api/chat endpoint, which accepts
    an "images" field (base64-encoded, no data-URI prefix) on a message.
    """

    def __init__(self, model: str = None):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_VISION_MODEL", "qwen2.5-vl")
        self.api_url = f"{self.base_url}/api/chat"

    def generate_response(self, question: str, image_bytes: bytes, context: str = "", chat_history: str = "") -> str:
        """
        Args:
            question: User's question about the image (e.g. "what does this
                       say?" for OCR, or "describe this chart")
            image_bytes: Raw bytes of the uploaded image (png/jpg/etc.)
            context: Optional extra text context (e.g. from retrieved docs)
                     to help interpret the image
            chat_history: Optional recent conversation transcript, in case
                          the question about the image references earlier
                          turns (e.g. "does this match what we discussed?")
        Returns:
            The model's text response.
        """
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            prompt = question or "Describe this image in detail, including any visible text (OCR)."
            if context.strip():
                prompt = f"{prompt}\n\nAdditional context that may help:\n{context}"
            if chat_history.strip():
                prompt = f"{prompt}\n\nRecent conversation (for context only):\n{chat_history}"

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt, "images": [image_b64]}
                ],
                "stream": False,
                "options": {"temperature": 0.3},
            }

            response = requests.post(self.api_url, json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "Sorry, I couldn't analyze the image.")
            return f"Error: Ollama vision API returned status code {response.status_code}"

        except requests.exceptions.RequestException as e:
            return (
                f"Error connecting to Ollama for image analysis: {e}. "
                f"Make sure Ollama is running and '{self.model}' is pulled (ollama pull {self.model})."
            )
        except Exception as e:
            return f"Unexpected error during image analysis: {e}"