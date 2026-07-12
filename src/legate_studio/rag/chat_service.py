"""
Chat Service

Handles LLM interactions for RAG-enabled chat.
Supports Claude, OpenAI, and Google Gemini with model selection.
"""

import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)


class ChatProvider(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"


class ChatService:
    """Service for LLM chat interactions."""

    # Default models
    DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
    DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        provider: ChatProvider = ChatProvider.CLAUDE,
        model: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize the chat service.

        Args:
            provider: Which LLM provider to use
            model: Specific model to use (defaults to provider's default)
            api_key: Optional API key to use directly. If not provided, falls back
                     to environment variable (backward compat / single-tenant mode).
        """
        self.provider = provider
        self._api_key = api_key

        if provider == ChatProvider.CLAUDE:
            self.model = model or self.DEFAULT_CLAUDE_MODEL
            self._init_claude()
        elif provider == ChatProvider.GEMINI:
            self.model = model or self.DEFAULT_GEMINI_MODEL
            self._init_gemini()
        else:
            self.model = model or self.DEFAULT_OPENAI_MODEL
            self._init_openai()

        logger.info(f"ChatService initialized: {provider.value}:{self.model}")

    def _init_claude(self):
        """Initialize Anthropic client."""
        import anthropic

        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=api_key)

    def _init_openai(self):
        """Initialize OpenAI client."""
        import openai

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.client = openai.OpenAI(api_key=api_key)

    def _init_gemini(self):
        """Initialize Google Gemini client.

        Note: google.generativeai configures globally via genai.configure(), which is
        stateful. For per-user BYOK keys, we pass the api_key directly to the
        GenerativeModel constructor via client_options, avoiding the global state issue.
        """
        import google.generativeai as genai

        api_key = self._api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        # Store key for use in _chat_gemini (we create a new model per request there)
        self._gemini_api_key = api_key
        # Initialize with the key for list_models / other class methods that need a configured client
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict:
        """Send messages to the LLM and get a response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum response tokens
            temperature: Sampling temperature

        Returns:
            Dict with keys:
              - "text": str — the assistant's response text
              - "usage": dict with "input_tokens" and "output_tokens" (both int)
        """
        if self.provider == ChatProvider.CLAUDE:
            return self._chat_claude(messages, max_tokens, temperature)
        elif self.provider == ChatProvider.GEMINI:
            return self._chat_gemini(messages, max_tokens, temperature)
        else:
            return self._chat_openai(messages, max_tokens, temperature)

    def _chat_claude(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Chat via Claude API."""
        # Extract system messages
        system_parts = []
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                chat_messages.append(msg)

        system_prompt = "\n\n".join(system_parts) if system_parts else None

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=chat_messages,
            )

            return {
                "text": response.content[0].text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }

        except Exception as e:
            logger.error(f"Claude chat failed: {e}")
            raise

    def _chat_openai(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Chat via OpenAI API."""
        import openai

        # Newer/reasoning models require max_completion_tokens instead of max_tokens
        model_lower = self.model.lower()
        uses_new_param = (
            model_lower.startswith("o1")
            or model_lower.startswith("o3")
            or model_lower.startswith("o4")
            or model_lower.startswith("gpt-5")
        )

        try:
            if uses_new_param:
                # Reasoning models: no temperature, use max_completion_tokens
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=max_tokens,
                    messages=messages,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                )

            usage = response.usage
            return {
                "text": response.choices[0].message.content,
                "usage": {
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "output_tokens": usage.completion_tokens if usage else 0,
                },
            }

        except openai.BadRequestError as e:
            # Handle case where model requires max_completion_tokens
            if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                logger.info(f"Retrying with max_completion_tokens for {self.model}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=max_tokens,
                    messages=messages,
                )
                usage = response.usage
                return {
                    "text": response.choices[0].message.content,
                    "usage": {
                        "input_tokens": usage.prompt_tokens if usage else 0,
                        "output_tokens": usage.completion_tokens if usage else 0,
                    },
                }
            raise

        except Exception as e:
            logger.error(f"OpenAI chat failed: {e}")
            raise

    def _chat_gemini(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Chat via Google Gemini API."""
        import google.generativeai as genai

        # Convert messages to Gemini format
        # Gemini uses 'user' and 'model' roles, and system instruction is separate
        system_parts = []
        chat_history = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            elif msg["role"] == "assistant":
                chat_history.append({"role": "model", "parts": [msg["content"]]})
            else:
                chat_history.append({"role": "user", "parts": [msg["content"]]})

        try:
            # Always create a fresh model instance with the stored API key.
            # This avoids the global genai.configure() state issue for per-user BYOK keys.
            api_key = getattr(self, "_gemini_api_key", None) or os.environ.get("GEMINI_API_KEY")
            model = genai.GenerativeModel(
                self.model,
                system_instruction=(
                    "\n\n".join(system_parts) if system_parts else None
                ),
            )
            # Re-configure with this instance's key before the call
            genai.configure(api_key=api_key)

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )

            # Use chat for multi-turn, or generate_content for single turn
            if len(chat_history) == 1:
                response = model.generate_content(
                    chat_history[0]["parts"][0],
                    generation_config=generation_config,
                )
            else:
                chat = model.start_chat(history=chat_history[:-1])
                response = chat.send_message(
                    chat_history[-1]["parts"][0],
                    generation_config=generation_config,
                )

            # Extract token usage from usage_metadata
            usage_meta = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0

            return {
                "text": response.text,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            }

        except Exception as e:
            logger.error(f"Gemini chat failed: {e}")
            raise

    # Fallback Anthropic models if API fetch fails
    ANTHROPIC_MODELS_FALLBACK = [
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
        {"id": "claude-opus-4-20250514", "name": "Claude Opus 4"},
        {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"},
        {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku"},
        {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
    ]

    # Fallback Gemini models if API fetch fails
    GEMINI_MODELS_FALLBACK = [
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
        {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
        {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash Lite"},
    ]

    @classmethod
    def get_available_models(cls, provider: ChatProvider) -> list[dict[str, str]]:
        """Get list of available models for a provider.

        Fetches dynamically from API for all providers.

        Returns:
            List of dicts with 'id' and 'name' keys
        """
        if provider == ChatProvider.CLAUDE:
            return cls.fetch_anthropic_models()
        elif provider == ChatProvider.GEMINI:
            return cls.fetch_gemini_models()
        else:
            return cls.fetch_openai_models()

    @classmethod
    def fetch_anthropic_models(cls) -> list[dict[str, str]]:
        """Fetch available models from Anthropic API."""
        import requests

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, returning fallback models")
            return cls.ANTHROPIC_MODELS_FALLBACK

        try:
            response = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            models = []
            for model in data.get("data", []):
                model_id = model.get("id", "")
                display_name = model.get("display_name", model_id)
                # Only include chat-capable models (claude-*)
                if model_id.startswith("claude-"):
                    models.append({"id": model_id, "name": display_name})

            if models:
                logger.info(f"Fetched {len(models)} Anthropic models")
                return models
            else:
                return cls.ANTHROPIC_MODELS_FALLBACK

        except Exception as e:
            logger.error(f"Failed to fetch Anthropic models: {e}")
            return cls.ANTHROPIC_MODELS_FALLBACK

    @classmethod
    def fetch_openai_models(cls) -> list[dict[str, str]]:
        """Fetch available models from OpenAI API."""
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, returning default models")
            return [
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
                {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
            ]

        try:
            client = openai.OpenAI(api_key=api_key)
            model_list = client.models.list()

            # Filter for GPT chat models
            models = []
            for model in model_list.data:
                model_id = model.id
                # Only include GPT models suitable for chat
                if model_id.startswith("gpt") and "instruct" not in model_id:
                    # Create friendly name
                    name = model_id.replace("-", " ").title()
                    models.append({"id": model_id, "name": name})

            # Sort by ID
            models.sort(key=lambda x: x["id"])
            logger.info(f"Fetched {len(models)} OpenAI models")
            return models

        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {e}")
            # Return defaults on error
            return [
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
                {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
            ]

    @classmethod
    def fetch_gemini_models(cls) -> list[dict[str, str]]:
        """Fetch available models from Google Gemini API."""
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set, returning fallback models")
            return cls.GEMINI_MODELS_FALLBACK

        try:
            genai.configure(api_key=api_key)
            models = []
            for model in genai.list_models():
                # Only include models that support generateContent (chat-capable)
                if "generateContent" in model.supported_generation_methods:
                    model_id = model.name.removeprefix("models/")
                    display_name = model.display_name or model_id
                    models.append({"id": model_id, "name": display_name})

            # Sort by ID so the list is stable
            models.sort(key=lambda x: x["id"])

            if models:
                logger.info(f"Fetched {len(models)} Gemini models")
                return models
            else:
                return cls.GEMINI_MODELS_FALLBACK

        except Exception as e:
            logger.error(f"Failed to fetch Gemini models: {e}")
            return cls.GEMINI_MODELS_FALLBACK

    @classmethod
    def from_config(cls, config: dict) -> "ChatService":
        """Create a ChatService from configuration dict.

        Args:
            config: Dict with 'provider' and optional 'model'

        Returns:
            Configured ChatService instance
        """
        provider_str = config.get("provider", "claude").lower()

        if provider_str == "gemini":
            provider = ChatProvider.GEMINI
        elif provider_str == "openai":
            provider = ChatProvider.OPENAI
        else:
            provider = ChatProvider.CLAUDE

        model = config.get("model")

        return cls(provider=provider, model=model)
