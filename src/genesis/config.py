"""
Configuration management for Genesis experiments.

API keys are loaded from environment variables or .env file.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for LLM API access."""
    api_key: str
    api_base: str
    model: str = "gpt-4"
    max_retries: int = 3
    timeout: int = 60
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load configuration from environment variables."""
        # Try to load .env file if python-dotenv is available
        try:
            from dotenv import load_dotenv
            # Look for .env in project root
            env_path = Path(__file__).parent.parent.parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
        except ImportError:
            pass
        
        api_key = os.getenv('OPENAI_API_KEY')
        api_base = os.getenv('OPENAI_API_BASE')
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in environment or .env file.\n"
                "Example .env file:\n"
                "  OPENAI_API_KEY=sk-xxx\n"
                "  OPENAI_API_BASE=http://your-api-endpoint/v1/chat/completions"
            )
        
        if not api_base:
            api_base = "https://api.openai.com/v1/chat/completions"
        
        return cls(api_key=api_key, api_base=api_base)


# Default configuration values
DEFAULT_CONFIG = {
    "OPENAI_API_KEY": "sk-MFElZTdYUPrwTIGV2hvavvkUqVeP1fhfJcJMbQjddOSxP75h",
    "OPENAI_API_BASE": "http://123.129.219.111:3000/v1/chat/completions",
}


def get_config() -> LLMConfig:
    """
    Get LLM configuration.
    
    Priority:
    1. Environment variables
    2. .env file
    3. Default values (for development only)
    """
    api_key = os.getenv('OPENAI_API_KEY', DEFAULT_CONFIG['OPENAI_API_KEY'])
    api_base = os.getenv('OPENAI_API_BASE', DEFAULT_CONFIG['OPENAI_API_BASE'])
    
    return LLMConfig(api_key=api_key, api_base=api_base)

