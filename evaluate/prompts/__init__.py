"""Prompt loader.

Selects the prompt set based on `PROMPT_LANG` in `evaluate/constants.py`.
Defaults to "en" (English prompts that explicitly handle Chinese content).

Re-exports:
    system_prompt                  - solver system prompt
    verifier_system_prompt         - verifier system prompt
    verifier_prompt_template       - verifier user prompt template
    question_num_prompt_template   - sub-question count probe template
    critic_prompt_template         - legacy 3-type critic prompt template
    process_eval_prompt_template   - CIE/LAE/DRE process eval prompt template
"""

try:
    from constants import PROMPT_LANG  # type: ignore
except ImportError:
    try:
        from evaluate.constants import PROMPT_LANG  # type: ignore
    except ImportError:
        PROMPT_LANG = "en"

if PROMPT_LANG == "zh":
    from .zh import (
        system_prompt,
        verifier_system_prompt,
        verifier_prompt_template,
        question_num_prompt_template,
        critic_prompt_template,
        process_eval_prompt_template,
    )
else:
    from .en import (
        system_prompt,
        verifier_system_prompt,
        verifier_prompt_template,
        question_num_prompt_template,
        critic_prompt_template,
        process_eval_prompt_template,
    )

__all__ = [
    "system_prompt",
    "verifier_system_prompt",
    "verifier_prompt_template",
    "question_num_prompt_template",
    "critic_prompt_template",
    "process_eval_prompt_template",
]
