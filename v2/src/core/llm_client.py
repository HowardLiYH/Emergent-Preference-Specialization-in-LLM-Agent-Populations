"""
LLM client wrapper for Gemini 2.5 Flash.

Handles API calls, rate limiting, retries, and logging.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    text: str
    tokens_used: int
    model: str
    latency_ms: float
    finish_reason: Optional[str] = None


class LLMClient:
    """
    Wrapper for Gemini 2.5 Flash API.

    Features:
    - Rate limiting
    - Automatic retries
    - Token counting
    - Logging
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Google API key (or from GOOGLE_API_KEY env var)
            model: Model name (default: gemini-2.5-flash)
            temperature: Generation temperature
            max_output_tokens: Maximum output tokens
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_errors = 0

        # Initialize client
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Gemini client."""
        if not self.api_key:
            logger.warning("No API key provided. Set GOOGLE_API_KEY environment variable.")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
            logger.info(f"Initialized LLM client with model: {self.model}")
        except ImportError:
            logger.error("google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            LLMResponse with generated text
        """
        if self._client is None:
            raise RuntimeError("LLM client not initialized. Check API key.")

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_output_tokens

        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                response = self._client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temp,
                        "max_output_tokens": max_tok,
                    }
                )

                latency_ms = (time.time() - start_time) * 1000

                # Extract text
                text = response.text if hasattr(response, 'text') else str(response)

                # Estimate tokens (rough approximation)
                input_tokens = len(prompt.split()) // 0.75
                output_tokens = len(text.split()) // 0.75
                total_tokens = int(input_tokens + output_tokens)

                self.total_requests += 1
                self.total_tokens += total_tokens

                return LLMResponse(
                    text=text,
                    tokens_used=total_tokens,
                    model=self.model,
                    latency_ms=latency_ms,
                    finish_reason="stop"
                )

            except Exception as e:
                self.total_errors += 1
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    raise

    def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """
        Generate text from prompt and image (vision).

        Args:
            prompt: Text prompt
            image_base64: Base64-encoded image
            temperature: Override default temperature

        Returns:
            LLMResponse with generated text
        """
        if self._client is None:
            raise RuntimeError("LLM client not initialized.")

        import base64

        temp = temperature if temperature is not None else self.temperature
        start_time = time.time()

        try:
            # Decode image
            image_bytes = base64.b64decode(image_base64)

            # Create image part
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }

            response = self._client.generate_content(
                [prompt, image_part],
                generation_config={"temperature": temp}
            )

            latency_ms = (time.time() - start_time) * 1000
            text = response.text if hasattr(response, 'text') else str(response)

            self.total_requests += 1

            return LLMResponse(
                text=text,
                tokens_used=int(len(text.split()) / 0.75),
                model=self.model,
                latency_ms=latency_ms
            )

        except Exception as e:
            self.total_errors += 1
            raise

    async def generate_async(
        self,
        prompt: str,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """Async version of generate."""
        # Run sync version in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, temperature)
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'model': self.model,
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'total_errors': self.total_errors,
            'error_rate': self.total_errors / max(1, self.total_requests),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.total_requests = 0
        self.total_tokens = 0
        self.total_errors = 0


# Singleton client for convenience
_default_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    """Get the default LLM client."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def set_client(client: LLMClient) -> None:
    """Set the default LLM client."""
    global _default_client
    _default_client = client
