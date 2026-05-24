"""Unified LLM call interface based on LiteLLM.

This module provides a single entry point `call_llm()` for invoking any
supported LLM provider (OpenAI, Anthropic, Google Gemini, self-hosted vLLM/SGLang,
and any OpenAI-compatible endpoint).

Configuration via environment variables (only set the ones you need):
    OPENAI_API_KEY        - for GPT-x / o3 / o4 models
    ANTHROPIC_API_KEY     - for Claude models
    GEMINI_API_KEY        - for Gemini models
    OPENAI_BASE_URL       - default base URL for OpenAI-compatible endpoints
                            (e.g. for Kimi, GLM, DeepSeek aggregator services)
    OPENAI_API_KEY        - reused as default key for the above

For self-hosted vLLM / SGLang servers, set per-model URLs in
`evaluate.constants.vllm_models` and pass `model` as listed there;
the function will route automatically through the OpenAI-compatible protocol.

See README for a full configuration guide.
"""

import os
import time
from typing import Optional, Tuple, Dict, Any, List

import litellm

# Read constants lazily; do not import at module load time so this file stays
# usable from `analyze/` without depending on evaluate-side constants.
_VLLM_MODELS_CACHE: Optional[Dict[str, str]] = None


def _get_vllm_models() -> Dict[str, str]:
    """Lazily load the vllm_models dict from evaluate.constants if available."""
    global _VLLM_MODELS_CACHE
    if _VLLM_MODELS_CACHE is not None:
        return _VLLM_MODELS_CACHE
    try:
        # Try sibling import first (evaluate/util/llm.py -> evaluate/constants.py)
        from constants import vllm_models  # type: ignore
        _VLLM_MODELS_CACHE = dict(vllm_models)
    except ImportError:
        try:
            from evaluate.constants import vllm_models  # type: ignore
            _VLLM_MODELS_CACHE = dict(vllm_models)
        except ImportError:
            _VLLM_MODELS_CACHE = {}
    return _VLLM_MODELS_CACHE


def _resolve_model(model: str, base_url: Optional[str]) -> Tuple[str, Optional[str]]:
    """Resolve a user-provided model name to a LiteLLM model spec + api_base.

    Routing rules:
      1. If `base_url` is explicitly given, use OpenAI-compatible protocol.
      2. If `model` is registered in vllm_models, use its URL via OpenAI-compatible.
      3. Otherwise pass `model` directly to LiteLLM (e.g. "gpt-5", "claude-opus-4-6",
         "gemini-3-pro" — LiteLLM auto-routes to the right provider).
    """
    if base_url:
        return f"openai/{model}", base_url

    vllm_models = _get_vllm_models()
    if model in vllm_models:
        return f"openai/{model}", vllm_models[model]

    return model, None


def call_llm(
    messages: List[Dict[str, Any]],
    model: str,
    max_tokens: int = 32768,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    """Call an LLM and return (response_text, metadata).

    Args:
        messages: OpenAI-format chat messages.
        model: Model identifier. Can be a public model name (e.g. "gpt-5",
            "claude-opus-4-6", "gemini-3-pro") or a key in `vllm_models`
            for self-hosted endpoints.
        max_tokens: Output token cap. For models that use
            `max_completion_tokens` (e.g. GPT-5, o3, o4), LiteLLM handles
            the field translation automatically.
        base_url: Override the API base URL (forces OpenAI-compatible protocol).
        api_key: Override the API key for this call.
        **kwargs: Additional parameters forwarded to litellm.completion()
            (e.g. temperature, reasoning_effort, thinking).

    Returns:
        (response_text, metadata) where metadata has:
            usage:       provider-reported usage dict (completion_tokens, etc.)
            total_time:  wall-clock seconds as string with 3 sig figs
            tokens:      best-effort completion token count (fallback to 0)
    """
    resolved_model, resolved_base = _resolve_model(model, base_url)

    call_kwargs: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if resolved_base:
        call_kwargs["api_base"] = resolved_base
        # Self-hosted vLLM/SGLang typically does not check the key,
        # but litellm requires *something* to be set.
        call_kwargs["api_key"] = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    elif api_key:
        call_kwargs["api_key"] = api_key

    call_kwargs.update(kwargs)

    metadata: Dict[str, Any] = {"usage": {}, "total_time": 0, "tokens": 0}
    content = ""
    t0 = time.time()

    try:
        response = litellm.completion(**call_kwargs)
        content = response.choices[0].message.content or ""

        # Strip <think>...</think> blocks if any reasoning model leaks them.
        if "</think>" in content:
            content = content.split("</think>")[-1]

        usage = getattr(response, "usage", None)
        if usage is not None:
            try:
                usage = usage.model_dump()
            except AttributeError:
                usage = dict(usage) if not isinstance(usage, dict) else usage
            metadata["usage"] = usage
            metadata["tokens"] = int(usage.get("completion_tokens", 0))

        metadata["total_time"] = "%.3g" % (time.time() - t0)
        return content, metadata

    except Exception as e:
        print(f"[llm] call_llm failed for model={model}: {e}")
        metadata["total_time"] = "%.3g" % (time.time() - t0)
        return content, metadata

