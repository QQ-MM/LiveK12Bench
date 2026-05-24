"""Default prompts (English) for LiveK12Bench.

These prompts are designed to handle exam papers in any language while
keeping the grading instructions themselves in English. They explicitly
ask the model to respond in the same language as the question.

Switch back to Chinese-native prompts by setting `PROMPT_LANG = "zh"`
in `evaluate/constants.py`.
"""

# ---------------------------------------------------------------------------
# Solver prompt: instructs the model how to format its final answer.
# ---------------------------------------------------------------------------
system_prompt = (
    "Solve the question below. The question may be in Chinese or English; "
    "respond in the same language as the question. Show your reasoning, "
    "and format your final answer(s) using the following rules:\n"
    "(1) Single-choice question: put the letter of the correct option inside one "
    "\\boxed{}, e.g. \\boxed{A}.\n"
    "(2) Multiple-choice question: put all correct option letters inside a single "
    "\\boxed{}, e.g. \\boxed{ACD}.\n"
    "(3) Fill-in-the-blank question: place each blank's answer in its own "
    "\\boxed{}, in order — e.g. \\boxed{x} for one blank, or \\boxed{x} and \\boxed{y} for two.\n"
    "(4) Free-response question: place each sub-question's answer in its own "
    "\\boxed{}, in order — e.g. \\boxed{x} for one sub-question, or \\boxed{x} and \\boxed{y} for two."
)


# ---------------------------------------------------------------------------
# Verifier system prompt.
# ---------------------------------------------------------------------------
verifier_system_prompt = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Verifier user-prompt template for free-response / fill-in-the-blank.
# Use .format(question=..., gold_answer=..., gold_solution=..., answer=...).
# ---------------------------------------------------------------------------
verifier_prompt_template = (
    "You are an expert exam grader. The exam content may be in Chinese or English. "
    "Determine whether the student's answer matches the reference answer. Apply these criteria:\n"
    "1. Question types include fill-in-the-blank and free-response. Some questions contain multiple "
    "sub-questions (e.g. a fill-in question with several blanks, or a free-response question with "
    "several parts). You must judge each sub-question independently.\n"
    "2. Different surface forms of the same answer count as equivalent — e.g. mathematically equal "
    "expressions, equivalent textual descriptions. If you cannot decide whether two forms are "
    "equivalent, mark the student's answer as wrong.\n"
    "3. Do NOT recompute the answer yourself. The reference answer is given. Just compare whether "
    "the student's answer matches the reference, given the question's expected answer format.\n"
    "4. For questions with a definite final answer, compare the student's final answer (typically "
    "wrapped in \\boxed{{}}) against the reference; the intermediate process need not match. "
    "For open-ended questions without a definite final answer (e.g. proofs), judge whether the "
    "student's reasoning is correct (a correct line of argument suffices).\n\n"
    "Apply the above criteria and output the number of sub-questions the student got correct, "
    "wrapped in \\boxed{{}}. For example: if the question has one part and the student is correct, "
    "output \\boxed{{1}}; if the question has three parts and the student got two right, "
    "output \\boxed{{2}}. The content to grade follows:\n"
    "Question: {question}\n"
    "Reference answer and solution: {gold_answer}{gold_solution}\n"
    "Student answer: {answer}\n\n"
    "Analysis:"
)


# ---------------------------------------------------------------------------
# Sub-question count probe.
# ---------------------------------------------------------------------------
question_num_prompt_template = (
    "{question}\nHow many sub-questions does the question above contain? "
    "Output only a number, nothing else."
)


# ---------------------------------------------------------------------------
# Critic prompt (legacy 3-error-type process evaluation).
# Not currently used by the pipeline; kept for users who want to plug it
# back in for ablation studies.
# ---------------------------------------------------------------------------
critic_prompt_template = (
    "Below is an exam question, a reference solution, and a student's solution "
    "(content may be in Chinese or English):\n"
    "**Question**\n{question}\n"
    "**Reference solution**\n{gold_solution}\n"
    "**Student solution**\n{answer}\n\n"
    "Review the student's solution step by step, identify the erroneous steps, and "
    "output the total number of errors. Consider only three types of objective errors: "
    "computation errors, hallucinations, and logical errors. Subjective issues such as "
    "lack of clarity or redundancy are NOT counted.\n"
    "(1) Computation error: incorrect numerical or algebraic operations.\n"
    "(2) Hallucination: a statement fabricated out of nothing — not derivable from the "
    "question, axioms/theorems, or earlier steps.\n"
    "(3) Logical error: the prior steps cannot support the current inference, "
    "e.g. wrong cause-effect relations.\n"
    "Output the count of erroneous steps inside \\boxed{{}}; if there are no errors, "
    "output \\boxed{{0}}. Note: count only root-cause errors; downstream errors caused by "
    "an earlier error do NOT count.\nAnalysis:"
)


# ---------------------------------------------------------------------------
# Process-evaluation prompt (CIE / LAE / DRE error categorisation).
# ---------------------------------------------------------------------------
process_eval_prompt_template = (
    "You are an expert evaluator of STEM solutions. You receive an exam question, "
    "a reference solution, and a student's solution (content may be in Chinese or English):\n"
    "**Question**\n{question}\n"
    "**Reference solution**\n{gold_solution}\n"
    "**Student solution**\n{answer}\n\n"
    "Your task is to review the student's solution step by step, identify erroneous steps, "
    "and output the total number of errors along with their types.\n"
    "Consider only three objective error categories: Condition Interpretation Error (CIE), "
    "Logical Assumption Error (LAE), and Deductive Reasoning Error (DRE). Subjective issues "
    "(e.g. lack of clarity, redundancy) are NOT counted.\n"
    "Definitions:\n"
    " (1) Condition Interpretation Error (CIE): the student's statements contradict the "
    "information in the question (including diagrams), the relevant axioms/theorems, or the "
    "stated conditions. This causes a systematic misunderstanding of the problem's facts, "
    "variable meanings, conditions, or solution objective.\n"
    " (2) Logical Assumption Error (LAE): the student fails to fully use the constraints or "
    "data given (or directly derivable) in the question, and instead introduces extra "
    "assumptions, simplifications, or defaults not supported by the question, leading to a "
    "systematic deviation in reasoning direction or conclusion. The essence is insufficient "
    "use of given information combined with unsupported assumptions.\n"
    " (3) Deductive Reasoning Error (DRE): during the derivation, the inference step, "
    "computation, or logical structure cannot support the resulting claim. This manifests as "
    "internal incorrectness or non-rigorousness in the deductive chain, conflicting with "
    "established laws, theoretical frameworks, or axiomatic systems, so that the conclusion is "
    "incompatible with established knowledge.\n"
    " Compare the student's solution against the reference following these definitions, and "
    "carefully identify the erroneous steps. A single question may contain multiple error types.\n"
    " Output format: first the count of erroneous steps, then three indicators for "
    "(1) CIE, (2) LAE, (3) DRE — each 1 if present, 0 if absent — in order, each inside its "
    "own \\boxed{{}}. If there are no errors, output \\boxed{{0}} \\boxed{{0}} \\boxed{{0}} \\boxed{{0}}.\n"
    " Count only root-cause errors; downstream errors caused by an earlier error do NOT count.\n"
    "Analysis:"
)
