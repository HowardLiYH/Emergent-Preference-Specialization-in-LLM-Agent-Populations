"""
LLM Client wrapper with rate limiting, retries, and cost tracking.

Provides a unified interface for OpenAI-compatible APIs.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import logging
import httpx

from .config import LLMConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Track token usage and costs."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Approximate costs (GPT-4 pricing)
    COST_PER_1K_PROMPT = 0.03
    COST_PER_1K_COMPLETION = 0.06

    @property
    def estimated_cost(self) -> float:
        """Estimate cost in USD."""
        prompt_cost = (self.prompt_tokens / 1000) * self.COST_PER_1K_PROMPT
        completion_cost = (self.completion_tokens / 1000) * self.COST_PER_1K_COMPLETION
        return prompt_cost + completion_cost

    def add(self, prompt: int, completion: int):
        """Add token counts from a request."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion


@dataclass
class RateLimiter:
    """Simple rate limiter for API calls."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000

    _request_times: List[float] = field(default_factory=list)
    _token_counts: List[tuple] = field(default_factory=list)  # (timestamp, tokens)

    async def wait_if_needed(self, estimated_tokens: int = 500):
        """Wait if rate limits would be exceeded."""
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        self._request_times = [t for t in self._request_times if t > minute_ago]
        self._token_counts = [(t, c) for t, c in self._token_counts if t > minute_ago]

        # Check request rate
        if len(self._request_times) >= self.requests_per_minute:
            sleep_time = self._request_times[0] - minute_ago + 0.1
            logger.info(f"Rate limit: waiting {sleep_time:.1f}s (requests)")
            await asyncio.sleep(sleep_time)

        # Check token rate
        recent_tokens = sum(c for _, c in self._token_counts)
        if recent_tokens + estimated_tokens > self.tokens_per_minute:
            sleep_time = self._token_counts[0][0] - minute_ago + 0.1
            logger.info(f"Rate limit: waiting {sleep_time:.1f}s (tokens)")
            await asyncio.sleep(sleep_time)

        # Record this request
        self._request_times.append(time.time())

    def record_tokens(self, tokens: int):
        """Record token usage for rate limiting."""
        self._token_counts.append((time.time(), tokens))


class ChatCompletion:
    """Response wrapper matching OpenAI format."""

    def __init__(self, content: str, usage: Dict[str, int]):
        self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})()})]
        self.usage = usage


class ChatCompletions:
    """Chat completions API matching OpenAI interface."""

    def __init__(self, client: 'LLMClient'):
        self.client = client

    async def create(
        self,
        model: str = None,
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> ChatCompletion:
        """Create a chat completion."""
        return await self.client._create_completion(
            model=model or self.client.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


class Chat:
    """Chat API namespace."""

    def __init__(self, client: 'LLMClient'):
        self.completions = ChatCompletions(client)


class LLMClient:
    """
    LLM Client with rate limiting, retries, and cost tracking.

    Usage:
        client = LLMClient()
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
        print(f"Cost so far: ${client.usage.estimated_cost:.2f}")
    """

    def __init__(self, config: LLMConfig = None):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration (defaults to loading from env)
        """
        self.config = config or get_config()
        self.usage = TokenUsage()
        self.rate_limiter = RateLimiter()

        # Create HTTP client
        self._http_client = httpx.AsyncClient(timeout=self.config.timeout)

        # API interface
        self.chat = Chat(self)

    async def _create_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> ChatCompletion:
        """Internal method to create a completion with retries."""

        # Estimate tokens for rate limiting
        estimated_tokens = sum(len(m.get('content', '')) // 4 for m in messages) + max_tokens
        await self.rate_limiter.wait_if_needed(estimated_tokens)

        # Prepare request
        # Check if the API base already includes the full path
        if '/chat/completions' in self.config.api_base:
            url = self.config.api_base
        else:
            url = f"{self.config.api_base.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        # Retry loop
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self._http_client.post(
                    url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()

                # Extract content
                content = data['choices'][0]['message']['content']

                # Track usage
                usage = data.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', estimated_tokens // 2)
                completion_tokens = usage.get('completion_tokens', len(content) // 4)

                self.usage.add(prompt_tokens, completion_tokens)
                self.rate_limiter.record_tokens(prompt_tokens + completion_tokens)

                return ChatCompletion(content, usage)

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                elif e.response.status_code >= 500:
                    # Server error - retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error {e.response.status_code}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except httpx.TimeoutException as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(f"Timeout, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        raise last_error or RuntimeError("Max retries exceeded")

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system: str = None
    ) -> str:
        """
        Convenience method for simple text generation.

        Args:
            prompt: The user prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system: Optional system message

        Returns:
            Generated text
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.chat.completions.create(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    def get_usage_report(self) -> str:
        """Get a summary of token usage and costs."""
        return (
            f"Token Usage Report\n"
            f"------------------\n"
            f"Prompt tokens:     {self.usage.prompt_tokens:,}\n"
            f"Completion tokens: {self.usage.completion_tokens:,}\n"
            f"Total tokens:      {self.usage.total_tokens:,}\n"
            f"Estimated cost:    ${self.usage.estimated_cost:.2f}"
        )

    async def close(self):
        """Close the HTTP client."""
        await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function
def create_client(config: LLMConfig = None) -> LLMClient:
    """Create an LLM client with default configuration."""
    return LLMClient(config)


async def test_connection():
    """Test the API connection."""
    client = LLMClient()
    try:
        response = await client.generate("Say 'Hello, Genesis!' in exactly those words.")
        print(f"✓ Connection successful!")
        print(f"Response: {response}")
        print(f"\n{client.get_usage_report()}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False
    finally:
        await client.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_connection())
