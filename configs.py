"""
Configuration settings for the workflows package.

This module contains configuration settings and constants used across the workflows package,
including model configurations, workflow settings, and other package-wide constants.
"""

AVAILABLE_MODELS = {
    "DeepSeek/V3": {
        "model": "deepseek-chat",
        "logprobs": False,
    },
    "OpenAI/gpt-4o": {
        "model": "gpt-4o-2024-11-20",
        "logprobs": True,
    },
    "OpenAI/gpt-4o-mini": {
        "model": "gpt-4o-mini-2024-07-18",
        "logprobs": True,
    },
    "OpenAI/gpt-3.5-turbo": {
        "model": "gpt-3.5-turbo-0125",
    },
    "Anthropic/claude-3-7-sonnet": {
        "model": "claude-3-7-sonnet-20250219",
    },
    "Anthropic/claude-3-5-sonnet": {
        "model": "claude-3-5-sonnet-20241022",
    },
    "Anthropic/claude-3-5-haiku": {
        "model": "claude-3-5-haiku-20241022",
    },
    "Cohere/command-r": {
        "model": "command-r-08-2024",
        "logprobs": True,
    },
    "Cohere/command-r-plus": {
        "model": "command-r-plus-08-2024",
        "logprobs": True,
    },
    "Cohere/command-r7b": {
        "model": "command-r7b-12-2024",
        "logprobs": False,
    },
}

# Function mapping for input/output transformations
TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}

FUNCTION_MAP = {
    "upper": str.upper,
    "lower": str.lower,
    "len": len,
    "split": str.split,
}
