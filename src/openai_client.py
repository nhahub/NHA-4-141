import os
from openai import OpenAI, RateLimitError, AuthenticationError, APIStatusError, APIConnectionError


class QuotaOrAuthError(Exception):
    """Raised when the OpenAI call fails for a reason that means 'this key/
    account can't serve requests right now' (out of credit, invalid key,
    rate limited) — as opposed to some other unexpected error. The router
    uses this to decide whether to move on to the next backend."""
    pass


class OpenAIChatClient:
    """
    Client for generating chat responses via the OpenAI API, exposing the
    same generate_response(question, context) interface as OllamaClient so
    it can be swapped in/out of LLMRouter transparently.
    """

    def __init__(self, model: str, api_key_env: str = "OPENAI_API_KEY"):
        """
        Args:
            model: OpenAI model name, e.g. "gpt-4o-mini"
            api_key_env: name of the environment variable holding the API key
                         (lets you use two different OpenAI accounts/keys for
                         the primary and secondary tiers if you want)
        """
        self.model = model
        api_key = os.getenv(api_key_env)
        self.client = OpenAI(api_key=api_key) if api_key else None

    def generate_response(self, question: str, context: str, chat_history: str = "") -> str:
        if self.client is None:
            raise QuotaOrAuthError(f"No API key configured for {self.model}")

        prompt = self._create_rag_prompt(question, context, chat_history)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000,
            )
            return completion.choices[0].message.content

        except (RateLimitError, AuthenticationError) as e:
            # RateLimitError also covers "insufficient_quota" (subscription/
            # credit exhausted) — OpenAI reuses the 429 status for both.
            raise QuotaOrAuthError(str(e)) from e
        except APIStatusError as e:
            if e.status_code in (401, 403, 429):
                raise QuotaOrAuthError(str(e)) from e
            raise
        except APIConnectionError as e:
            raise QuotaOrAuthError(f"Could not reach OpenAI: {e}") from e

    def _create_rag_prompt(self, question: str, context: str, chat_history: str = "") -> str:
        history_block = (
            f"\n\nRecent conversation (for context only — don't repeat it back):\n{chat_history}\n"
            if chat_history.strip() else ""
        )
        return f"""You are a helpful AI assistant that answers questions based on the provided context.
Use the following context to answer the user's question. The context is split into numbered chunks
like [1], [2], etc. — when you use information from a chunk, cite its number inline, e.g.
"...as mentioned in the report [1]." If the answer cannot be found in the context,
say so clearly and don't make up information.
{history_block}
Context:
{context}

Question: {question}

Answer: Based on the provided context, """