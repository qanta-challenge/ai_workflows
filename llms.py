# %%

import json
import os
from typing import Any, Optional

import cohere
import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic._internal._core_utils import CoreSchemaOrField, is_core_schema
from pydantic.json_schema import GenerateJsonSchema
from rich import print as rprint

# Initialize global cache
try:
    from src.envs import CACHE_PATH, LLM_CACHE_REPO
except ImportError:
    logger.error(
        "Either module src.envs not found, or CACHE_PATH or LLM_CACHE_REPO not found, trying to look in environment"
    )
    CACHE_PATH = os.environ.get("LLM_CACHE_PATH", ".")
    LLM_CACHE_REPO = None


from .configs import AVAILABLE_MODELS
from .llmcache import LLMCache

llm_cache = LLMCache(cache_dir=CACHE_PATH, hf_repo=LLM_CACHE_REPO)


class CohereSchemaGenerator(GenerateJsonSchema):
    """Generates JSON schema for Cohere models without default titles."""

    def field_title_should_be_set(self, schema: CoreSchemaOrField) -> bool:
        return_value = super().field_title_should_be_set(schema)
        if return_value and is_core_schema(schema):
            return False
        return return_value


def _openai_is_json_mode_supported(model_name: str) -> bool:
    if model_name.startswith("gpt-4"):
        return True
    if model_name.startswith("gpt-3.5"):
        return False
    logger.warning(f"OpenAI model {model_name} is not available in this app, skipping JSON mode, returning False")
    return False


class LLMOutput(BaseModel):
    content: str = Field(description="The content of the response")
    logprob: Optional[float] = Field(None, description="The log probability of the response")


def _get_langchain_chat_output(llm: BaseChatModel, system: str, prompt: str) -> str:
    output = llm.invoke([("system", system), ("human", prompt)])
    ai_message = output["raw"]
    content = {"content": ai_message.content, "tool_calls": ai_message.tool_calls}
    content_str = json.dumps(content)
    return {"content": content_str, "output": output["parsed"].model_dump()}


def _cohere_completion(
    model: str, system: str, prompt: str, response_model, temperature: float | None = None, logprobs: bool = True
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    schema = response_model.model_json_schema(schema_generator=CohereSchemaGenerator)
    if "title" in schema:
        del schema["title"]
    response_format = {
        "type": "json_object",
        "schema": schema,
    }
    response = client.chat(
        model=model,
        messages=messages,
        response_format=response_format,
        logprobs=logprobs,
        temperature=temperature,
    )
    output = {}
    output["content"] = response.message.content[0].text
    output["output"] = response_model.model_validate_json(response.message.content[0].text).model_dump()
    if logprobs:
        output["logprob"] = sum(lp.logprobs[0] for lp in response.logprobs)
        output["prob"] = np.exp(output["logprob"])
    return output


def _langchain_completion(
    provider: str, model: str, system: str, prompt: str, response_model, temperature: float | None = None
) -> str:
    if provider == "OpenAI":
        model_cls = ChatOpenAI
    elif provider == "Anthropic":
        model_cls = ChatAnthropic
    elif provider == "DeepSeek":
        model_cls = ChatDeepSeek
    else:
        raise ValueError(f"Provider {provider} not supported")
    llm = model_cls(model=model, temperature=temperature).with_structured_output(response_model, include_raw=True)
    return _get_langchain_chat_output(llm, system, prompt)


def _openai_completion(
    model: str, system: str, prompt: str, response_model, temperature: float | None = None, logprobs: bool = True
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_model,
        logprobs=logprobs,
        temperature=temperature,
    )
    output = {}
    output["content"] = response.choices[0].message.content
    output["output"] = response.choices[0].message.parsed.model_dump()
    if logprobs:
        output["logprob"] = sum(lp.logprob for lp in response.choices[0].logprobs.content)
        output["prob"] = np.exp(output["logprob"])
    return output


def _llm_completion(
    model: str, system: str, prompt: str, response_format, temperature: float | None = None, logprobs: bool = False
) -> dict[str, Any]:
    """
    Generate a completion from an LLM provider with structured output without caching.

    Args:
        model (str): Provider and model name in format "provider/model" (e.g. "OpenAI/gpt-4")
        system (str): System prompt/instructions for the model
        prompt (str): User prompt/input
        response_format: Pydantic model defining the expected response structure
        logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
            Note: Not supported by Anthropic models.

    Returns:
        dict: Contains:
            - output: The structured response matching response_format
            - logprob: (optional) Sum of log probabilities if logprobs=True
            - prob: (optional) Exponential of logprob if logprobs=True

    Raises:
        ValueError: If logprobs=True with Anthropic models
    """
    model_name = AVAILABLE_MODELS[model]["model"]
    provider = model.split("/")[0]
    if provider == "Cohere":
        return _cohere_completion(model_name, system, prompt, response_format, temperature, logprobs)
    elif provider == "OpenAI":
        if _openai_is_json_mode_supported(model_name):
            return _openai_completion(model_name, system, prompt, response_format, temperature, logprobs)
        elif logprobs:
            raise ValueError(f"{model} does not support logprobs feature.")
        else:
            return _langchain_completion("OpenAI", model_name, system, prompt, response_format, temperature)
    elif provider in {"Anthropic", "DeepSeek"}:
        if logprobs:
            raise ValueError(f"{provider} models do not support logprobs")
        return _langchain_completion(provider, model_name, system, prompt, response_format, temperature)
    else:
        raise ValueError(f"Provider {provider} not supported")


def completion(
    model: str, system: str, prompt: str, response_format, temperature: float | None = None, logprobs: bool = True
) -> dict[str, Any]:
    """
    Generate a completion from an LLM provider with structured output with caching.

    Args:
        model (str): Provider and model name in format "provider/model" (e.g. "OpenAI/gpt-4")
        system (str): System prompt/instructions for the model
        prompt (str): User prompt/input
        response_format: Pydantic model defining the expected response structure
        logprobs (bool, optional): Whether to return log probabilities. Defaults to True.
            Note: Not supported by Anthropic models.

    Returns:
        dict: Contains:
            - output: The structured response matching response_format
            - logprob: (optional) Sum of log probabilities if logprobs=True
            - prob: (optional) Exponential of logprob if logprobs=True

    Raises:
        ValueError: If logprobs=True with Anthropic models
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model} not supported")
    if logprobs and not AVAILABLE_MODELS[model].get("logprobs", False):
        # logger.warning(f"{model} does not support logprobs feature, setting logprobs to False")
        logprobs = False

    # Check cache first
    cached_response = llm_cache.get(model, system, prompt, response_format, temperature)
    if cached_response and (not logprobs or cached_response.get("logprob")):
        logger.trace(f"Cache hit for model {model}")
        return cached_response

    logger.trace(f"Cache miss for model {model}, calling API. Logprobs: {logprobs}")

    # Continue with the original implementation for cache miss
    response = _llm_completion(model, system, prompt, response_format, temperature, logprobs)

    # Update cache with the new response
    llm_cache.set(
        model,
        system,
        prompt,
        response_format,
        temperature,
        response,
    )

    return response


# %%
if __name__ == "__main__":
    from tqdm import tqdm

    class ExplainedAnswer(BaseModel):
        """
        The answer to the question and a terse explanation of the answer.
        """

        answer: str = Field(description="The short answer to the question")
        explanation: str = Field(description="5 words terse best explanation of the answer.")

    TEST_MODELS = ["Cohere/command-r7b", "Anthropic/claude-3-5-haiku", "DeepSeek/V3"]

    models = TEST_MODELS
    system = "You are an accurate and concise explainer of scientific concepts."
    prompt = "Which planet is closest to the sun in the Milky Way galaxy? Answer directly, no explanation needed."

    llm_cache = LLMCache(cache_dir="/tmp/cache", hf_repo="qanta-challenge/advcal-llm-cache", reset=True)

    # First call - should be a cache miss
    logger.info("First call - should be a cache miss")
    for model in tqdm(models):
        response = completion(model, system, prompt, ExplainedAnswer, logprobs=False)
        rprint(response)

    # Second call - should be a cache hit
    logger.info("Second call - should be a cache hit")
    for model in tqdm(models):
        response = completion(model, system, prompt, ExplainedAnswer, logprobs=False)
        rprint(response)

    # Slightly different prompt - should be a cache miss
    logger.info("Different prompt - should be a cache miss")
    prompt2 = "Which planet is closest to the sun? Answer directly."
    for model in tqdm(models):
        response = completion(model, system, prompt2, ExplainedAnswer, logprobs=False)
        rprint(response)

    # Get cache entries count from SQLite
    try:
        cache_entries = llm_cache.get_all_entries()
        logger.info(f"Cache now has {len(cache_entries)} items")
    except Exception as e:
        logger.error(f"Failed to get cache entries: {e}")

    # Test adding entry with temperature parameter
    logger.info("Testing with temperature parameter")
    response = completion(models[0], system, "What is Mars?", ExplainedAnswer, temperature=0.7, logprobs=False)
    rprint(response)

    # Demonstrate forced sync to HF if repo is configured
    if llm_cache.hf_repo_id:
        logger.info("Forcing sync to HF dataset")
        try:
            llm_cache.sync_to_hf()
            logger.info("Successfully synced to HF dataset")
        except Exception as e:
            logger.exception(f"Failed to sync to HF: {e}")
    else:
        logger.info("HF repo not configured, skipping sync test")

# %%
