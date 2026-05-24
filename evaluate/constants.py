"""Configuration for the LiveK12Bench evaluation pipeline.

For LLM API credentials, set the appropriate environment variables
(see README and `evaluate/util/llm.py` for details). Do NOT hardcode keys here.
"""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prompt language: "en" (default; English prompts that handle Chinese content)
# or "zh" (Chinese prompts archived from the original closed-source version,
# preserved for users who want maximum performance on Chinese-only papers).
PROMPT_LANG = "en"

# ---------------------------------------------------------------------------
# Self-hosted model endpoints (vLLM / SGLang)
# ---------------------------------------------------------------------------
# Map model name -> OpenAI-compatible base URL.
# Models listed here will be routed to the given URL via the OpenAI protocol.
# Public-API models (gpt-x, claude-x, gemini-x, kimi-x, glm-x, qwen-x served by
# aggregator services) do NOT need to be listed here — LiteLLM resolves them
# automatically based on the model name and standard environment variables.
vllm_models = {
    # "qwen3-vl-32b":  "http://your-vllm-host:8081/v1",
}

# ---------------------------------------------------------------------------
# Paths and run configuration
# ---------------------------------------------------------------------------
paper_dir = os.path.join(ROOT_DIR, "data/chinese_k12/bio")          # parsed question JSONs
prediction_dir = os.path.join(ROOT_DIR, "predictions/predictions_e2e")  # model outputs

# Solving mode for solve.py:
#   e2e   - end-to-end: text + raw images go to the model directly
#   photo - photo-based: feed a per-question screenshot
#   exam  - whole-paper images + question-number instructions
mode_cfg = "e2e"

# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------
metrics_path = os.path.join(ROOT_DIR, "metrics/metrics.xlsx")

# Verifier ensemble used by evaluate.py to grade open-ended responses.
verifiers = [
    "gpt-4o-2024-11-20",
    "gemini-2.5-flash",
    "qwen3-30b-a3b-instruct-2507",
    "gpt-5-nano",
]

# Solver models under evaluation.
solvers = [
    "gemini-3.1-pro",
    "gpt-5",
    "claude-opus-4-6",
    "kimi-k2.5",
    # "qwen3-vl-8b",   # uncomment after registering the URL in vllm_models above
]

# Process-evaluation critic models, currently disabled in the pipeline but
# kept here in case you want to re-enable the legacy critic loop.
critics = ["deepseek-v3-local-II", "gpt-4o-2024-11-20", "claude-4-sonnet-20250514"]
