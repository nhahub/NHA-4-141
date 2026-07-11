import os
from typing import List, Optional
from .openai_client import OpenAIChatClient, QuotaOrAuthError
from .llm_client import OllamaClient
from .vision_client import QwenVisionClient
from .intent_router import IntentRouter


class LLMRouter:
    """
    Holds the configured backends and two ways of picking between them:

      - Manual modes ("auto" / "openai" / "local"): the user picks exactly
        which backend answers, as before.
      - "smart" mode: an IntentRouter decides per-question whether this is
        a code question (-> qwen2.5-coder), an image question (-> qwen2.5-vl,
        triggered whenever an image is attached, regardless of mode), or a
        general chat question (-> the local chat model).

    Backends:
      - openai_backends: ordered list of configured OpenAI tiers (primary,
        secondary), used for mode="openai" and as the first part of "auto".
      - local_backend: local Ollama chat model (e.g. qwen2.5:1.5b) — used
        for mode="local", as the final fallback for "auto", and as the
        "chat" leaf of "smart".
      - code_backend: local Ollama code model (e.g. qwen2.5-coder:1.5b) —
        used as the "code" leaf of "smart".
      - vision_backend: local Ollama vision model (e.g. qwen2.5-vl) — used
        whenever an image is attached, regardless of mode.
    """

    def __init__(self):
        self.openai_backends = []  # list of (name, client)
        self.last_used_backend = None

        primary_model = os.getenv("OPENAI_MODEL_PRIMARY", "")
        if primary_model:
            self.openai_backends.append((
                f"openai:{primary_model}",
                OpenAIChatClient(model=primary_model, api_key_env="OPENAI_API_KEY_PRIMARY"),
            ))

        secondary_model = os.getenv("OPENAI_MODEL_SECONDARY", "")
        if secondary_model:
            self.openai_backends.append((
                f"openai:{secondary_model}",
                OpenAIChatClient(model=secondary_model, api_key_env="OPENAI_API_KEY_SECONDARY"),
            ))

        local_model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        self.local_backend = (f"ollama:{local_model}", OllamaClient(model=local_model, role="chat"))

        code_model = os.getenv("OLLAMA_CODE_MODEL", "qwen2.5-coder:1.5b")
        self.code_backend = (f"ollama:{code_model}", OllamaClient(model=code_model, role="code"))

        vision_model = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:3b")
        self.vision_backend = (f"ollama:{vision_model}", QwenVisionClient(model=vision_model))

        self.intent_router = IntentRouter(classifier_model=local_model)

    def _chain_for_mode(self, mode: str):
        if mode == "openai":
            return list(self.openai_backends)
        if mode == "local":
            return [self.local_backend]
        # "auto" (default): OpenAI tiers first, then local as final fallback
        return list(self.openai_backends) + [self.local_backend]

    def generate_response(
        self,
        question: str,
        context: str,
        mode: str = "auto",
        image_bytes: Optional[bytes] = None,
        chat_history: str = "",
    ) -> str:
        # An attached image always wins, regardless of mode — there's no
        # ambiguity about intent once an image is in the message, and none
        # of the text backends can actually look at it.
        if image_bytes is not None:
            name, client = self.vision_backend
            try:
                response = client.generate_response(question, image_bytes, context, chat_history)
                self.last_used_backend = name
                return response
            except Exception as e:
                self.last_used_backend = None
                return f"Image analysis failed ({name}): {e}"

        if mode == "smart":
            intent = self.intent_router.classify(question)
            name, client = self.code_backend if intent == "code" else self.local_backend
            try:
                response = client.generate_response(question, context, chat_history)
                self.last_used_backend = f"{name} (smart:{intent})"
                return response
            except Exception as e:
                self.last_used_backend = None
                return f"'{name}' failed to respond: {e}"

        chain = self._chain_for_mode(mode)
        if not chain:
            self.last_used_backend = None
            return f"No backend is configured for mode '{mode}'. Check your .env settings."

        errors = []
        for name, client in chain:
            try:
                response = client.generate_response(question, context, chat_history)
                self.last_used_backend = name
                return response
            except QuotaOrAuthError as e:
                errors.append(f"{name}: {e}")
                continue
            except Exception as e:
                errors.append(f"{name}: {e}")
                continue

        self.last_used_backend = None
        return (
            f"The selected backend(s) for mode '{mode}' all failed to respond:\n- "
            + "\n- ".join(errors)
            + "\n\nTry switching the model in the sidebar, or check your OpenAI API key/quota."
        )

    def available_modes(self) -> List[tuple]:
        """Returns [(mode_value, friendly_label), ...] for populating the UI dropdown."""
        modes = []
        if self.openai_backends:
            modes.append(("auto", "Auto (GPT first, Qwen fallback)"))
            modes.append(("openai", "ChatGPT (OpenAI)"))
        local_name = self.local_backend[0].split(":", 1)[1]
        modes.append(("local", f"Qwen ({local_name})"))
        modes.append(("smart", "🧭 Smart Auto (detects code/image)"))
        return modes

    def get_backend_chain(self) -> List[str]:
        """Return all configured backend names (for display/debugging)."""
        names = [name for name, _ in self.openai_backends]
        names.append(self.local_backend[0])
        names.append(self.code_backend[0])
        names.append(self.vision_backend[0])
        return names