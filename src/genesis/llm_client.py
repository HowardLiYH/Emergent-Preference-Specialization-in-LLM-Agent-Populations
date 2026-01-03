"""
LLM Client wrapper with rate limiting, retries, and cost tracking.

Supports:
- OpenAI (GPT-4, GPT-5 series)
- Anthropic (Claude)
- Google Gemini (FREE with Google AI Studio!)
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal
import logging
import os

logger = logging.getLogger(__name__)


# Model pricing per 1M tokens
MODEL_PRICING = {
    # OpenAI
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    # Anthropic
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-opus-4.5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    # Google Gemini - FREE TIER AVAILABLE!
    "gemini-2.0-flash": {"input": 0.00, "output": 0.00},  # FREE!
    "gemini-1.5-flash": {"input": 0.00, "output": 0.00},  # FREE!
    "gemini-1.5-pro": {"input": 0.00, "output": 0.00},    # FREE!
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},    # Paid tier
}


@dataclass
class LLMConfig:
    """Configuration for LLM API access."""
    provider: Literal["openai", "gemini", "anthropic"] = "gemini"
    api_key: str = ""
    api_base: str = ""
    model: str = "gemini-2.0-flash"  # Default to FREE Gemini!
    max_retries: int = 5
    timeout: int = 60

    @classmethod
    def from_env(cls, provider: str = None) -> 'LLMConfig':
        """Load configuration from environment variables."""
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            env_path = Path(__file__).parent.parent.parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
        except ImportError:
            pass

        # Auto-detect provider based on available keys
        if provider is None:
            if os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
                provider = "gemini"
            elif os.getenv('OPENAI_API_KEY'):
                provider = "openai"
            elif os.getenv('ANTHROPIC_API_KEY'):
                provider = "anthropic"
            else:
                provider = "gemini"  # Default to Gemini (free tier)

        if provider == "gemini":
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY', '')
            return cls(
                provider="gemini",
                api_key=api_key,
                model="gemini-2.0-flash",
            )
        elif provider == "openai":
            return cls(
                provider="openai",
                api_key=os.getenv('OPENAI_API_KEY', ''),
                api_base=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
                model="gpt-4o-mini",
            )
        elif provider == "anthropic":
            return cls(
                provider="anthropic",
                api_key=os.getenv('ANTHROPIC_API_KEY', ''),
                model="claude-3-5-sonnet",
            )

        raise ValueError(f"Unknown provider: {provider}")


@dataclass
class TokenUsage:
    """Track token usage and costs."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = "gemini-2.0-flash"

    @property
    def estimated_cost(self) -> float:
        """Estimate cost in USD."""
        pricing = MODEL_PRICING.get(self.model, {"input": 0, "output": 0})
        prompt_cost = (self.prompt_tokens / 1_000_000) * pricing["input"]
        completion_cost = (self.completion_tokens / 1_000_000) * pricing["output"]
        return prompt_cost + completion_cost

    def add(self, prompt: int, completion: int):
        """Add token counts from a request."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion


@dataclass
class RateLimiter:
    """Simple rate limiter for API calls."""
    requests_per_minute: int = 10  # For gemini-2.0-flash-exp (10 RPM limit)
    tokens_per_minute: int = 250_000  # 250K TPM for flash-exp
    min_delay_seconds: float = 6.5  # 6.5 seconds between requests (10 RPM = 6s/request + buffer)

    _request_times: List[float] = field(default_factory=list)
    _token_counts: List[tuple] = field(default_factory=list)
    _last_request_time: float = 0

    async def wait_if_needed(self, estimated_tokens: int = 500):
        """Wait if rate limits would be exceeded."""
        now = time.time()

        # Enforce minimum delay between requests
        time_since_last = now - self._last_request_time
        if time_since_last < self.min_delay_seconds:
            wait_time = self.min_delay_seconds - time_since_last
            await asyncio.sleep(wait_time)

        now = time.time()
        minute_ago = now - 60

        self._request_times = [t for t in self._request_times if t > minute_ago]
        self._token_counts = [(t, c) for t, c in self._token_counts if t > minute_ago]

        if len(self._request_times) >= self.requests_per_minute:
            sleep_time = self._request_times[0] - minute_ago + 0.5
            logger.info(f"Rate limit: waiting {sleep_time:.1f}s (requests)")
            await asyncio.sleep(sleep_time)

        recent_tokens = sum(c for _, c in self._token_counts)
        if recent_tokens + estimated_tokens > self.tokens_per_minute:
            sleep_time = self._token_counts[0][0] - minute_ago + 0.5
            logger.info(f"Rate limit: waiting {sleep_time:.1f}s (tokens)")
            await asyncio.sleep(sleep_time)

        self._last_request_time = time.time()
        self._request_times.append(time.time())

    def record_tokens(self, tokens: int):
        """Record token usage for rate limiting."""
        self._token_counts.append((time.time(), tokens))


class LLMClient:
    """
    Unified LLM Client supporting OpenAI, Gemini, and Anthropic.

    Usage:
        # Use FREE Gemini (recommended!)
        client = LLMClient.for_gemini(api_key="your-google-api-key")

        # Or use OpenAI
        client = LLMClient.for_openai(api_key="sk-...")

        response = await client.generate("Hello!")
        print(response)
        print(f"Cost: ${client.usage.estimated_cost:.4f}")
    """

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig.from_env()
        self.usage = TokenUsage(model=self.config.model)
        self.rate_limiter = RateLimiter()
        self._http_client = None

    @classmethod
    def for_gemini(cls, api_key: str = None, model: str = "gemini-2.0-flash") -> 'LLMClient':
        """Create a client for Google Gemini (FREE!)."""
        config = LLMConfig(
            provider="gemini",
            api_key=api_key or os.getenv('GOOGLE_API_KEY', ''),
            model=model,
        )
        return cls(config)

    @classmethod
    def for_openai(cls, api_key: str = None, model: str = "gpt-4o-mini",
                   api_base: str = None) -> 'LLMClient':
        """Create a client for OpenAI."""
        config = LLMConfig(
            provider="openai",
            api_key=api_key or os.getenv('OPENAI_API_KEY', ''),
            api_base=api_base or os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
            model=model,
        )
        return cls(config)

    @classmethod
    def for_anthropic(cls, api_key: str = None, model: str = "claude-3-5-sonnet") -> 'LLMClient':
        """Create a client for Anthropic Claude."""
        config = LLMConfig(
            provider="anthropic",
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY', ''),
            model=model,
        )
        return cls(config)

    async def _get_http_client(self):
        """Lazy initialization of HTTP client."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._http_client

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system: str = None
    ) -> str:
        """Generate text using the configured LLM."""
        if self.config.provider == "gemini":
            return await self._generate_gemini(prompt, temperature, max_tokens, system)
        elif self.config.provider == "openai":
            return await self._generate_openai(prompt, temperature, max_tokens, system)
        elif self.config.provider == "anthropic":
            return await self._generate_anthropic(prompt, temperature, max_tokens, system)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    async def _generate_gemini(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str = None
    ) -> str:
        """Generate using Google Gemini API (direct REST - more stable)."""
        import httpx

        await self.rate_limiter.wait_if_needed()

        # Build content with system instruction
        full_prompt = prompt
        if system:
            full_prompt = f"[System Instructions: {system}]\n\n{prompt}"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent?key={self.config.api_key}"

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }

        for attempt in range(self.config.max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()

                # Extract content
                content = data["candidates"][0]["content"]["parts"][0]["text"]

                # Track usage
                usage_meta = data.get("usageMetadata", {})
                prompt_tokens = usage_meta.get("promptTokenCount", len(prompt) // 4)
                completion_tokens = usage_meta.get("candidatesTokenCount", len(content) // 4)

                self.usage.add(prompt_tokens, completion_tokens)
                self.rate_limiter.record_tokens(prompt_tokens + completion_tokens)

                return content

            except httpx.HTTPStatusError as e:
                if e.response.status_code in [429, 500, 502, 503]:
                    # For 429: start at 30s since we know that's what's needed
                    # For other errors: shorter waits
                    if e.response.status_code == 429:
                        wait_time = 30 + (attempt * 15)  # 30s, 45s, 60s, 75s, 90s
                    else:
                        wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s, 16s, 32s
                    logger.warning(f"HTTP {e.response.status_code}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError) as e:
                wait_time = 2 ** attempt
                logger.warning(f"Connection error: {e}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Gemini error: {e}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    async def _generate_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str = None
    ) -> str:
        """Generate using OpenAI API."""
        import httpx

        await self.rate_limiter.wait_if_needed()

        # Handle URL construction
        base = self.config.api_base.rstrip('/')
        if '/chat/completions' in base:
            url = base
        else:
            url = f"{base}/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        client = await self._get_http_client()

        for attempt in range(self.config.max_retries):
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"]

                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", len(prompt) // 4)
                completion_tokens = usage.get("completion_tokens", len(content) // 4)

                self.usage.add(prompt_tokens, completion_tokens)
                self.rate_limiter.record_tokens(prompt_tokens + completion_tokens)

                return content

            except httpx.HTTPStatusError as e:
                if e.response.status_code in [429, 500, 502, 503]:
                    # For 429: start at 30s since we know that's what's needed
                    # For other errors: shorter waits
                    if e.response.status_code == 429:
                        wait_time = 30 + (attempt * 15)  # 30s, 45s, 60s, 75s, 90s
                    else:
                        wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s, 16s, 32s
                    logger.warning(f"HTTP {e.response.status_code}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError) as e:
                wait_time = 2 ** (attempt + 1)
                logger.warning(f"Connection error: {type(e).__name__}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        raise RuntimeError("Max retries exceeded")

    async def _generate_anthropic(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str = None
    ) -> str:
        """Generate using Anthropic Claude API."""
        import httpx

        await self.rate_limiter.wait_if_needed()

        url = "https://api.anthropic.com/v1/messages"

        payload = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        client = await self._get_http_client()

        for attempt in range(self.config.max_retries):
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

                content = data["content"][0]["text"]

                usage = data.get("usage", {})
                prompt_tokens = usage.get("input_tokens", len(prompt) // 4)
                completion_tokens = usage.get("output_tokens", len(content) // 4)

                self.usage.add(prompt_tokens, completion_tokens)
                self.rate_limiter.record_tokens(prompt_tokens + completion_tokens)

                return content

            except httpx.HTTPStatusError as e:
                if e.response.status_code in [429, 500, 502, 503]:
                    # For 429: start at 30s since we know that's what's needed
                    # For other errors: shorter waits
                    if e.response.status_code == 429:
                        wait_time = 30 + (attempt * 15)  # 30s, 45s, 60s, 75s, 90s
                    else:
                        wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s, 16s, 32s
                    logger.warning(f"HTTP {e.response.status_code}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    # OpenAI-compatible interface for backwards compatibility
    @property
    def chat(self):
        """OpenAI-compatible chat interface."""
        return _ChatNamespace(self)

    def get_usage_report(self) -> str:
        """Get a summary of token usage and costs."""
        return (
            f"Token Usage Report ({self.config.provider} - {self.config.model})\n"
            f"{'─' * 50}\n"
            f"Prompt tokens:     {self.usage.prompt_tokens:,}\n"
            f"Completion tokens: {self.usage.completion_tokens:,}\n"
            f"Total tokens:      {self.usage.total_tokens:,}\n"
            f"Estimated cost:    ${self.usage.estimated_cost:.4f}"
        )

    async def close(self):
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class _ChatNamespace:
    """OpenAI-compatible chat namespace."""
    def __init__(self, client: LLMClient):
        self.completions = _CompletionsNamespace(client)


class _CompletionsNamespace:
    """OpenAI-compatible completions namespace."""
    def __init__(self, client: LLMClient):
        self.client = client

    async def create(
        self,
        model: str = None,
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ):
        """Create a chat completion (OpenAI-compatible)."""
        # Extract system and user messages
        system = None
        prompt_parts = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                prompt_parts.append(msg["content"])

        prompt = "\n".join(prompt_parts)

        content = await self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system
        )

        # Return OpenAI-compatible response
        return type('Response', (), {
            'choices': [type('Choice', (), {
                'message': type('Message', (), {'content': content})()
            })()]
        })()


# Convenience functions
def create_client(provider: str = None, **kwargs) -> LLMClient:
    """
    Create an LLM client.

    Args:
        provider: "gemini" (FREE!), "openai", or "anthropic"
        **kwargs: Additional config options

    Returns:
        LLMClient instance
    """
    if provider == "gemini" or provider is None:
        return LLMClient.for_gemini(**kwargs)
    elif provider == "openai":
        return LLMClient.for_openai(**kwargs)
    elif provider == "anthropic":
        return LLMClient.for_anthropic(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def test_connection(provider: str = "gemini"):
    """Test the API connection."""
    print(f"Testing {provider} connection...")

    if provider == "gemini":
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ No GOOGLE_API_KEY found!")
            print("\nTo get a FREE API key:")
            print("1. Go to: https://aistudio.google.com/")
            print("2. Click 'Get API Key'")
            print("3. Set: export GOOGLE_API_KEY='your-key'")
            return False
        client = LLMClient.for_gemini(api_key=api_key)
    else:
        client = create_client(provider)

    try:
        response = await client.generate("Say 'Hello, Genesis!' in exactly those words.")
        print(f"✅ Connection successful!")
        print(f"Response: {response}")
        print(f"\n{client.get_usage_report()}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    finally:
        await client.close()


if __name__ == "__main__":
    import sys
    provider = sys.argv[1] if len(sys.argv) > 1 else "gemini"
    asyncio.run(test_connection(provider))
