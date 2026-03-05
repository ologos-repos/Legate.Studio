"""
Chat Service

Handles LLM interactions for RAG-enabled chat.
Supports Claude and OpenAI with model selection.
"""

import os
import logging
from typing import List, Dict, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ChatProvider(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"


class ChatService:
    """Service for LLM chat interactions."""

    # Default models
    DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        provider: ChatProvider = ChatProvider.CLAUDE,
        model: Optional[str] = None,
    ):
        """Initialize the chat service.

        Args:
            provider: Which LLM provider to use
            model: Specific model to use (defaults to provider's default)
        """
        self.provider = provider

        if provider == ChatProvider.CLAUDE:
            self.model = model or self.DEFAULT_CLAUDE_MODEL
            self._init_claude()
        else:
            self.model = model or self.DEFAULT_OPENAI_MODEL
            self._init_openai()

        logger.info(f"ChatService initialized: {provider.value}:{self.model}")

    def _init_claude(self):
        """Initialize Anthropic client."""
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=api_key)

    def _init_openai(self):
        """Initialize OpenAI client."""
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.client = openai.OpenAI(api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Send messages to the LLM and get a response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum response tokens
            temperature: Sampling temperature

        Returns:
            The assistant's response text
        """
        if self.provider == ChatProvider.CLAUDE:
            return self._chat_claude(messages, max_tokens, temperature)
        else:
            return self._chat_openai(messages, max_tokens, temperature)

    def _chat_claude(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Chat via Claude API."""
        # Extract system messages
        system_parts = []
        chat_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_parts.append(msg['content'])
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

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude chat failed: {e}")
            raise

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Chat via OpenAI API."""
        import openai

        # Newer/reasoning models require max_completion_tokens instead of max_tokens
        model_lower = self.model.lower()
        uses_new_param = (
            model_lower.startswith('o1') or
            model_lower.startswith('o3') or
            model_lower.startswith('o4') or
            model_lower.startswith('gpt-5')
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

            return response.choices[0].message.content

        except openai.BadRequestError as e:
            # Handle case where model requires max_completion_tokens
            if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                logger.info(f"Retrying with max_completion_tokens for {self.model}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=max_tokens,
                    messages=messages,
                )
                return response.choices[0].message.content
            raise

        except Exception as e:
            logger.error(f"OpenAI chat failed: {e}")
            raise

    # Fallback Anthropic models if API fetch fails
    ANTHROPIC_MODELS_FALLBACK = [
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
        {"id": "claude-opus-4-20250514", "name": "Claude Opus 4"},
        {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"},
        {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku"},
        {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
    ]

    @classmethod
    def get_available_models(cls, provider: ChatProvider) -> List[Dict[str, str]]:
        """Get list of available models for a provider.

        Fetches dynamically from API for both providers.

        Returns:
            List of dicts with 'id' and 'name' keys
        """
        if provider == ChatProvider.CLAUDE:
            return cls.fetch_anthropic_models()
        else:
            return cls.fetch_openai_models()

    @classmethod
    def fetch_anthropic_models(cls) -> List[Dict[str, str]]:
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
    def fetch_openai_models(cls) -> List[Dict[str, str]]:
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
    def from_config(cls, config: Dict) -> "ChatService":
        """Create a ChatService from configuration dict.

        Args:
            config: Dict with 'provider' and optional 'model'

        Returns:
            Configured ChatService instance
        """
        provider_str = config.get('provider', 'claude').lower()
        provider = ChatProvider.CLAUDE if provider_str == 'claude' else ChatProvider.OPENAI
        model = config.get('model')

        return cls(provider=provider, model=model)
