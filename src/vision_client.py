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
        self.model = model or os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:3b")
        self.api_url = f"{self.base_url}/api/chat"
        self.tags_url = f"{self.base_url}/api/tags"
        
        # Check if they want to override num_gpu (e.g. force CPU)
        num_gpu_env = os.getenv("OLLAMA_VISION_NUM_GPU")
        self.num_gpu = None
        if num_gpu_env is not None and num_gpu_env.strip() != "":
            try:
                self.num_gpu = int(num_gpu_env)
            except ValueError:
                pass

    def _model_is_available(self) -> bool:
        """Check whether the configured vision model is actually pulled in Ollama."""
        try:
            resp = requests.get(self.tags_url, timeout=5)
            if resp.status_code != 200:
                return True  # can't verify — don't block on this check
            names = {m.get("name", "") for m in resp.json().get("models", [])}
            # Ollama tags are stored as e.g. "qwen2.5vl:3b"; also allow a bare
            # match without the ":tag" suffix in case the user pulled a
            # differently-tagged version.
            if self.model in names:
                return True
            base = self.model.split(":")[0]
            return any(n == self.model or n.split(":")[0] == base for n in names)
        except requests.exceptions.RequestException:
            return True  # Ollama unreachable — let the real call surface that error

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
            if not self._model_is_available():
                return (
                    f"Error: the vision model '{self.model}' is not available in Ollama. "
                    f"Pull it first with: ollama pull {self.model}\n"
                    f"(Run 'ollama list' to see what's currently installed.)"
                )

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            prompt = question or "Describe this image in detail, including any visible text (OCR)."
            if context.strip():
                prompt = f"{prompt}\n\nAdditional context that may help:\n{context}"
            if chat_history.strip():
                prompt = f"{prompt}\n\nRecent conversation (for context only):\n{chat_history}"

            options = {"temperature": 0.3}
            if self.num_gpu is not None:
                options["num_gpu"] = self.num_gpu

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt, "images": [image_b64]}
                ],
                "stream": False,
                "options": options,
                "keep_alive": "10m",
            }

            response = requests.post(self.api_url, json=payload, timeout=300)

            # Check for CUDA or toolchain compilation errors to trigger CPU fallback
            is_cuda_error = False
            detail = ""
            if response.status_code != 200:
                try:
                    detail = response.json().get("error", "")
                except ValueError:
                    detail = response.text[:300]
                
                err_lower = detail.lower()
                if "cuda error" in err_lower or "ptx" in err_lower or "toolchain" in err_lower:
                    is_cuda_error = True

            # If CUDA error occurred, retry with forced CPU execution (num_gpu: 0)
            if is_cuda_error and self.num_gpu != 0:
                payload["options"]["num_gpu"] = 0
                try:
                    response = requests.post(self.api_url, json=payload, timeout=300)
                except requests.exceptions.RequestException as e:
                    return f"Error retrying vision API on CPU: {e}"

            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "Sorry, I couldn't analyze the image.")
                if is_cuda_error:
                    content += "\n\n*(Note: Ran on CPU fallback due to local CUDA toolchain issues)*"
                return content

            # Redo detail extraction in case of retry failure
            detail = ""
            try:
                detail = response.json().get("error", "")
            except ValueError:
                detail = response.text[:300]

            hint = ""
            if "image input" in detail.lower() or "not found" in detail.lower():
                hint = f" Try: ollama pull {self.model}"

            return (
                f"Error: Ollama vision API returned status code {response.status_code}"
                + (f" — {detail}" if detail else "")
                + hint
            )

        except requests.exceptions.RequestException as e:
            return (
                f"Error connecting to Ollama for image analysis: {e}. "
                f"Make sure Ollama is running and '{self.model}' is pulled (ollama pull {self.model})."
            )
        except Exception as e:
            return f"Unexpected error during image analysis: {e}"