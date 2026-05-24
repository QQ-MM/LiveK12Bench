"""Chinese prompts archive — preserved for users who want to maximize
performance on Chinese exam papers. Activate by setting
`PROMPT_LANG = "zh"` in `evaluate/constants.py`.

These prompts are extracted verbatim from the original closed-source
version of the codebase, where the entire benchmark was built around
Chinese K12 exam content.
"""

# ---------------------------------------------------------------------------
# Solver prompt: instructs the model how to format its final answer.
# ---------------------------------------------------------------------------
system_prompt = (
    "请解答下面的题目，输出解题过程并按照如下规则标记出你的最终答案：\n"
    "（1）如果是单选题，将正确选项的编号写到一个\\boxed{}内，如\\boxed{A}；\n"
    "（2）如果是多选题，将所有正确选项的编号用一个\\boxed{}括起来，如\\boxed{ACD}；\n"
    "（3）如果是填空题，将各个空的答案依次用\\boxed{}括起来，"
    "如填空题有一个空时答案可以写作\\boxed{x}、有两个空时答案可以写作\\boxed{x}和\\boxed{y}；\n"
    "（4）如果是解答题，将各个子问题的答案依次用\\boxed{}括起来，"
    "如解答题只有一个子问题时答案可以写作\\boxed{x}、有两个子问题时答案可以写作\\boxed{x}和\\boxed{y}。"
)


# ---------------------------------------------------------------------------
# Verifier system prompt for grading open-ended questions.
# ---------------------------------------------------------------------------
verifier_system_prompt = "你是一个得力的助手。"


# ---------------------------------------------------------------------------
# Verifier user-prompt template for filling-blank / open-ended questions.
# Use .format(question=..., gold_answer=..., gold_solution=..., answer=...).
# ---------------------------------------------------------------------------
verifier_prompt_template = (
    "请你作为一个阅卷专家，判断下面的答案是否与标准答案一致，即考生是否回答正确。下面是评判标准：\n"
    "1. 题目类型包括填空题和解答题。部分题目可能包含多个子问题，例如有多个空的填空题和有多个小问的解答题，"
    "你需要分别判断每一个子问题是否回答正确。\n"
    "2. 有些答案可能通过不同的方式表达，比如有些答案可能是一个数学表达式，有些答案可能是一个文字描述，"
    "只要表达的意思一致即可。且有些公式通过不同的方式表达，但等价，也是正确的。"
    "如果你难以判断是否一致，判定为学生答案错误。\n"
    "3. 你不需要重新计算问题答案，因为标准答案已经给出，只需要根据问题形式来判断考生的答案是否与标准答案一致，是否正确即可。\n"
    "4. 对于有明确结果的问题，将考生最终答案（通常置于\\boxed{{}}中）其与标准答案对比即可,不需要考虑过程正确与否。"
    "对于无明确结果的题目（例如证明题），需要判断解题过程是否正确（思路正确即可）。\n\n"
    "请你根据上述标准阅卷，输出考生做对的子问题数并将其置于\\boxed{{}}中。"
    "例如：当题目只有一个问题且答案正确时，输出\\boxed{{1}}；当题目有3个小问学生答对了两个时，输出\\boxed{{2}}。"
    "以下是你需要评阅的内容：\n"
    "原问题：{question}\n标准答案及解析：{gold_answer}{gold_solution}\n考生答案：{answer}\n\n分析："
)


# ---------------------------------------------------------------------------
# Sub-question count probe (used to determine how many sub-questions a
# free-form question contains; the verifier output is normalized against this).
# ---------------------------------------------------------------------------
question_num_prompt_template = "{question}\n上面这道题目有几个子问题？注意仅输出数字，不要输出其他内容！"


# ---------------------------------------------------------------------------
# Critic prompt (legacy 3-error-type process evaluation).
# Critic prompt (legacy 3-error-type process evaluation).
# Not currently used by the pipeline; kept for users who want to plug it
# back in for ablation studies.
# ---------------------------------------------------------------------------
critic_prompt_template = (
    "以下是一个考试题目、标准解题过程以及一个考生的解题过程：\n"
    "**题目**\n{question}\n"
    "**标准解题过程**\n{gold_solution}\n"
    "**考生解题过程**\n{answer}\n\n"
    "你的任务是逐句审查该考生解答，分析其中错误的步骤并最终输出错误数量。"
    "错误类型只考虑计算错误、幻觉和逻辑错误这三种客观错误情况，其他诸如不清晰、冗余等主观错误不考虑。\n"
    "（1）计算错误指过程中的数字运算和代数运算等步骤错误；\n"
    "（2）幻觉指过程中出现凭空捏造的陈述，不来源于题目条件、公理定理和之前的解题过程。\n"
    "（3）逻辑错误指之前的过程无法支撑当前步骤的推理，例如错误的因果关系等。\n"
    "输出错误步骤的数量置于\\boxed{{}}中，如果没有错误输出\\boxed{{0}}。"
    "注意：错误数量仅计算源头性的错误步骤，由之前步骤错误导致的错误不计入。分析："
)


# ---------------------------------------------------------------------------
# Process-evaluation prompt (CIE / LAE / DRE error categorisation).
# Used by evaluate_process.py to grade reasoning process quality.
# ---------------------------------------------------------------------------
process_eval_prompt_template = (
    "你作为理科解题评估专家，接收到的信息是一个考试题目、标准解题过程以及一个考生的解题过程，具体内容为：\n"
    "**题目**\n{question}\n"
    "**标准解题过程**\n{gold_solution}\n"
    "**考生解题过程**\n{answer}\n\n"
    "任务要求是逐句审查该考生解答，分析其中错误的步骤并最终输出错误数量及类型。\n"
    "错误类型只考虑三类客观错误情况，分别为题干理解偏差 (Condition Interpretation Error)、"
    "计算错误 (Logical Assumption Error)、和演绎推理错误 (Deductive Reasoning Error)，"
    "其他诸如不清晰、冗余等主观错误无需考虑在内。\n"
    "错误类型的区分性准确定义为：\n"
    " (1) 题干理解偏差指在解题过程中出现了与题目图像中蕴含的信息，领域相关的基本公理定理，以及题干给定的基本条件不一致的数值信息或知识陈述。"
    "该错误会导致对原始问题的事实陈述、变量含义、条件关系或题目求解目标出现系统性理解偏差的错误。\n"
    " (2) 逻辑假设错误是指未充分利用题干已给或可直接推得的约束与数据，而擅自引入未被题干支持的额外假设、简化或默认条件，"
    "并据此开展推理，导致推理方向或结论发生系统性偏差的错误。该错误的本质在于信息利用不足与无根据假设的叠加；\n"
    " (3) 演绎推理错误是指基于现有条件进行推导的过程中，所采用的推理步骤、计算操作或逻辑结构无法支撑当前步骤的推理结果。"
    "该错误体现为演绎链条内部的不正确性或不严密性，与公认的学科定律、理论框架或公理体系发生冲突，"
    "从而导致推理结论与既有知识并不自洽相容的错误。\n"
    " 请按照上述错误定义将考生解题过程与标准解题过程进行对照分析，并严格筛选出相应的错误解题步骤，"
    "一道题目可能同时存在多种类型的解答错误。\n"
    " 输出格式要求为先输出错误步骤的数量，再分别输出三种错误类型 "
    "(1)题干理解偏差, (2)逻辑假设错误, (3)推理演绎错误是否存在 (存在=1，不存在=0), "
    "所有输出依次置于\\boxed{{}}中，如果没有错误则输出\\boxed{{0}} \\boxed{{0}} \\boxed{{0}} \\boxed{{0}}.\n"
    " 请注意错误数量仅计算源头性的错误步骤，由之前步骤错误导致的错误不计入。分析："
)
