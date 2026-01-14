"""
Configuration management for Genesis experiments.

Supports multiple LLM providers:
- Google Gemini (FREE tier available!)
- OpenAI
- Anthropic Claude

API keys are loaded from environment variables.
"""

import os
from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for LLM API access."""
    provider: Literal["gemini", "openai", "anthropic"] = "gemini"
    api_key: str = ""
    api_base: str = ""
    model: str = "gemini-2.0-flash"
    max_retries: int = 3
    timeout: int = 60

    @classmethod
    def from_env(cls, provider: str = None) -> 'LLMConfig':
        """Load configuration from environment variables."""
        # Try to load .env file
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).parent.parent.parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
        except ImportError:
            pass

        # Auto-detect provider
        if provider is None:
            if os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
                provider = "gemini"
            elif os.getenv('OPENAI_API_KEY'):
                provider = "openai"
            elif os.getenv('ANTHROPIC_API_KEY'):
                provider = "anthropic"
            else:
                provider = "gemini"

        return cls._create_for_provider(provider)

    @classmethod
    def _create_for_provider(cls, provider: str) -> 'LLMConfig':
        """Create config for a specific provider."""
        if provider == "gemini":
            return cls(
                provider="gemini",
                api_key=os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY', ''),
                model="gemini-2.0-flash",  # FREE!
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
        else:
            raise ValueError(f"Unknown provider: {provider}")


def get_config(provider: str = None) -> LLMConfig:
    """
    Get LLM configuration.

    Args:
        provider: "gemini" (FREE!), "openai", or "anthropic"
                  If None, auto-detects based on available API keys.

    Returns:
        LLMConfig instance
    """
    return LLMConfig.from_env(provider)


# Default working configuration (from environment variables)
DEFAULT_PROXY_CONFIG = LLMConfig(
    provider=os.getenv("LLM_PROVIDER", "gemini"),
    api_key=os.getenv("OPENAI_API_KEY", os.getenv("GEMINI_API_KEY", "")),
    api_base=os.getenv("OPENAI_API_BASE", ""),
    model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
    max_retries=5,
    timeout=120,  # Increased timeout for slow responses
)


# Available models by provider
AVAILABLE_MODELS = {
    "gemini": [
        "gemini-2.0-flash",      # FREE - Fast, good quality
        "gemini-1.5-flash",      # FREE - Very fast
        "gemini-1.5-pro",        # FREE - Best quality in free tier
        "gemini-2.5-pro",        # Paid - Deep Think capabilities
    ],
    "openai": [
        "gpt-4o-mini",           # Cheap - Good for development
        "gpt-4o",                # Balanced
        "gpt-4-turbo",           # High quality
        "gpt-5-nano",            # Ultra cheap
        "gpt-5-mini",            # Budget
        "gpt-5",                 # Recommended
    ],
    "anthropic": [
        "claude-3-haiku",        # Fast, cheap
        "claude-3-5-sonnet",     # Balanced
        "claude-opus-4.5",       # Best quality
    ],
}


def print_setup_instructions():
    """Print setup instructions for each provider."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    LLM PROVIDER SETUP                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🆓 GOOGLE GEMINI (RECOMMENDED - FREE!)                      ║
║  ─────────────────────────────────────                       ║
║  1. Go to: https://aistudio.google.com/                      ║
║  2. Click "Get API Key"                                      ║
║  3. Create a new API key                                     ║
║  4. Set environment variable:                                ║
║     export GOOGLE_API_KEY='your-api-key'                     ║
║                                                              ║
║  💰 OPENAI                                                   ║
║  ──────────                                                  ║
║  1. Go to: https://platform.openai.com/api-keys              ║
║  2. Create API key                                           ║
║  3. Set: export OPENAI_API_KEY='sk-...'                      ║
║                                                              ║
║  💰 ANTHROPIC CLAUDE                                         ║
║  ───────────────────                                         ║
║  1. Go to: https://console.anthropic.com/                    ║
║  2. Create API key                                           ║
║  3. Set: export ANTHROPIC_API_KEY='sk-ant-...'               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_setup_instructions()
