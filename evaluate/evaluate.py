"""Outcome-level evaluation: compare model predictions against gold answers
and aggregate metrics into an Excel sheet.

For each paper x model:
- Multiple-choice questions are graded mechanically by parsing \\boxed{}.
- Open-ended (fill-in / free-response) questions are graded by an ensemble
  of LLM verifiers; the per-verifier responses are saved back into the
  prediction JSON for auditability.

Results are also pushed into `metrics_path` (Excel) for cross-model comparison.
"""

import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, wait, as_completed

from tqdm import tqdm

from util.llm import call_llm
from util.save_metrics import save_dict_to_excel
from util.dataset_loader import (
    load_questions as _loader_load,
    split_subject_to_run_id,
    MULTIPLE_CHOICE,
    FILL_IN_BLANK,
    OPEN_ENDED,
    PROVING,
    UNKNOWN,
)
from solve import predict
from prompts import (
    verifier_system_prompt,
    verifier_prompt_template,
    question_num_prompt_template,
)
from constants import metrics_path, verifiers, solvers, prediction_dir, mode_cfg

current_dir = os.path.dirname(os.path.abspath(__file__))


def extract_boxed_content(text):
    """Extract every `\\boxed{...}` payload from `text`, supporting nested braces."""
    pattern = re.compile(r"\\boxed\{")
    matches = pattern.finditer(text)
    result = []
    for match in matches:
        start_pos = match.end()
        brace_count = 1
        end_pos = None
        for i in range(start_pos, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
            if brace_count == 0:
                end_pos = i
                break
        if end_pos is not None:
            result.append(text[start_pos:end_pos])
    return result


def analyze_question_num(question):
    """Ask a small model how many sub-questions the question contains."""
    prompt = [
        {"type": "text", "text": question_num_prompt_template.format(question=question)}
    ]
    messages = [{"role": "user", "content": prompt}]
    response, _ = call_llm(messages=messages, model="gpt-4.1-mini")
    print(f"sub-question count probe response: {response}")
    return int(response)


def verify_answer_for_filling_problem(question, gold_answer="", gold_solution="",
                                      answer="", score=10, model="gpt-4o-2024-11-20"):
    """Ask one verifier model to grade a free-response answer.

    Returns (awarded_score: int, raw_response_text or None).
    """
    try:
        prompt_text = verifier_prompt_template.format(
            question=question,
            gold_answer=gold_answer,
            gold_solution=gold_solution,
            answer=answer,
        )
        prompt = [{"type": "text", "text": prompt_text}]
        messages = [
            {"role": "system", "content": verifier_system_prompt},
            {"role": "user", "content": prompt},
        ]

        response, _ = call_llm(messages=messages, model=model, max_tokens=16384)

        if response is None:
            print(f"verifier model: {model}, response is None")
            return 0, None

        preview = response[:200] if len(response) > 200 else response
        print(f"verifier model: {model}, response: {preview}...")

        pattern = r"\\boxed\{(\d+)\}"
        matches = re.findall(pattern, response)

        try:
            total_num = len(gold_answer) if gold_answer else analyze_question_num(question)
            total_num = max(1, total_num)
        except Exception as e:
            print(f"Error analyzing question num: {e}")
            total_num = 1

        if matches:
            try:
                correct_num = int(matches[-1])
                if correct_num >= total_num:
                    return score, response
                if correct_num >= 0:
                    return int(round(score * correct_num / total_num)), response
                return 0, response
            except (ValueError, IndexError) as e:
                print(f"Error parsing correct_num: {e}")
                return 0, response

        print(f"no matches in response from {model}")
        return 0, response

    except Exception as e:
        print(f"Exception in verify_answer_for_filling_problem with model {model}: {e}")
        import traceback
        traceback.print_exc()
        return 0, f"Error: {e}"


def verify_answer_for_choice_problem(response, gold, score):
    """Mechanically grade a multiple-choice answer by parsing \\boxed{}.

    Supports both `\\boxed{A}` and `\\boxed{\\text{A}}` formats. For multi-select
    answers stored as a flat list of single-letter strings, splits each \\boxed{ABC}
    into {A, B, C}; otherwise treats each \\boxed{...} as a single answer token.
    """
    boxed_pattern = re.compile(r"\\boxed\{(?:\\text\{)?([A-Za-z]+)\}?\}")
    boxed_matches = boxed_pattern.findall(response)

    gold_upper = [g.upper() for g in gold]
    gold_set = set(gold_upper)
    should_split = all(len(g) == 1 for g in gold)

    all_matches = set()
    for match in boxed_matches:
        m = match.upper()
        if should_split:
            all_matches.update(list(m))
        else:
            all_matches.add(m)

    print(all_matches, gold_set)

    if all_matches == gold_set:
        return score
    if all_matches < gold_set and len(gold) > 1:
        return 1
    return 0


def check(
    *,
    run_id,
    model,
    split=None,
    subject=None,
    json_path=None,
    limit=None,
    ids=None,
):
    """Grade ``model``'s predictions on a run, write metrics back to JSON + Excel.

    The question source is resolved via the dataset loader: pass
    ``--split`` (HF dataset) or ``--json`` (analyze.py output).
    """
    records = _resolve_records(
        run_id=run_id, split=split, subject=subject,
        json_path=json_path, limit=limit, ids=ids,
    )
    answers = [r["answer"] for r in records]
    questions = [r["question"] for r in records]
    types = [r["question_type"] for r in records]
    solutions = [r["solution"] for r in records]
    scores = [r["point_value"] or 10 for r in records]

    preds, metadata = [], []
    null = 0
    prediction_path = os.path.join(prediction_dir, model, f"{run_id}.json")
    with open(prediction_path, "r") as file:
        data = json.load(file)
        data_ = [data[str(i)] for i in range(len(data))]
        for d in data_:
            preds.append(d["prediction"])
            meta = d["metadata"]
            if meta.get("usage"):
                meta["tokens"] = meta["usage"]["completion_tokens"]
            else:
                meta["tokens"] = (meta.get("tokens") or 0) + 1
            metadata.append(meta)
            if not d["prediction"]:
                null += 1
    assert len(answers) == len(preds), (
        f"length mismatch: {len(answers)} questions vs {len(preds)} predictions "
        f"— re-run solve.py with the same split/subject/limit/ids."
    )

    # Build the verifier ensemble for this run, excluding the model under test
    # to avoid self-grading bias. If too few are left, recycle.
    available_verifiers = [v for v in verifiers if v != model]
    if len(available_verifiers) < 3:
        base = available_verifiers.copy() if available_verifiers else verifiers.copy()
        while len(available_verifiers) < 3:
            available_verifiers.append(base[len(available_verifiers) % len(base)])

    metrics_list = []
    correct, correct_16k, correct_arl, incorrect = 0, 0, 0, []
    choice_score_gain, filling_score_gain, choice_score_sum, filling_score_sum = 0, 0, 0, 0
    proving_score_gain, proving_score_sum = 0, 0

    print("Grading...")
    for i in tqdm(range(len(answers))):
        metrics = {
            "correctness": False,
            "valid": bool(preds[i]),
            "outcome_score": 0.0,
        }

        if types[i] == MULTIPLE_CHOICE or types[i] == UNKNOWN:
            print(f"Question {i}")
            choice_score = verify_answer_for_choice_problem(preds[i], answers[i], scores[i])
            metrics["outcome_score"] = float(round(choice_score, 2))

            choice_score_sum += scores[i]
            choice_score_gain += round(choice_score, 2)
            if choice_score == scores[i]:
                correct += 1
                metrics["correctness"] = True
                correct_arl += 2048 / metadata[i]["tokens"]
                if metadata[i]["tokens"] <= 16384:
                    correct_16k += 1
            else:
                incorrect.append(i)

        else:
            selected_verifiers = available_verifiers[:3]
            # Initialize cot_* fields only for verifiers actually used on this question.
            # This avoids stale None fields for verifiers that never ran (e.g. the
            # solver-under-test, which is excluded from grading itself).
            for v in selected_verifiers:
                metrics[f"cot_{v}"] = None

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                for v in selected_verifiers:
                    fut = executor.submit(
                        verify_answer_for_filling_problem,
                        questions[i], answers[i], solutions[i], preds[i], scores[i], v
                    )
                    futures[fut] = v

                filling_scores = []
                for fut in as_completed(futures):
                    v = futures[fut]
                    try:
                        score_result, response = fut.result()
                        filling_scores.append(score_result)
                        metrics[f"cot_{v}"] = response
                    except Exception as e:
                        print(f"Error calling verifier {v}: {e}")
                        filling_scores.append(0)
                        metrics[f"cot_{v}"] = None

            filling_scores = [s if isinstance(s, int) else 0 for s in filling_scores]
            filling_score = sum(filling_scores) / 3
            metrics["outcome_score"] = float(round(filling_score, 2))

            if types[i] in (OPEN_ENDED, FILL_IN_BLANK):
                filling_score_sum += scores[i]
                filling_score_gain += round(filling_score, 2)
            elif types[i] == PROVING:
                proving_score_sum += scores[i]
                proving_score_gain += round(filling_score, 2)
            if filling_score == scores[i]:
                correct += 1
                metrics["correctness"] = True
                correct_arl += 2048 / metadata[i]["tokens"]
                if metadata[i]["tokens"] <= 16384:
                    correct_16k += 1
            else:
                incorrect.append(i)

        metrics_list.append(metrics)

    # Write per-question metrics back into the prediction file.
    with open(prediction_path, "r") as file:
        pred_data = json.load(file)
    for i, m in enumerate(metrics_list):
        pred_data[str(i)]["metrics"] = m
    with open(prediction_path, "w", encoding="utf-8") as file:
        json.dump(pred_data, file, ensure_ascii=False, indent=2)
    print(f"Per-question metrics saved to: {prediction_path}")

    # Aggregate metrics for cross-model comparison.
    print(f"Results for {model} on {run_id}:")
    print(
        f"1. Accuracy:\n"
        f"   Accuracy: {correct * 100 / len(answers):.1f}\n"
        f"   Accuracy (≤16k tokens): {correct_16k * 100 / len(answers):.1f}\n"
        f"   ARL: {correct_arl * 100 / len(answers):.1f}\n"
        f"   Invalid: {null}    Incorrect indices: {incorrect}\n"
    )

    thrput = sum(x["tokens"] for x in metadata) / sum(float(x["total_time"]) for x in metadata)
    ave_length = sum(x["tokens"] for x in metadata) / len(answers)
    ave_time = sum(float(x["total_time"]) for x in metadata) / len(answers)
    print(
        f"2. Efficiency:\n"
        f"   Throughput: {thrput:.2f} tokens/s\n"
        f"   Average time: {ave_time:.2f}s    Average response length: {ave_length:.2f}\n"
    )

    exam_total = choice_score_sum + filling_score_sum + proving_score_sum
    exam_score = (
        (choice_score_gain + filling_score_gain + proving_score_gain) / exam_total * 100
        if exam_total else 0.0
    )
    print(
        f"3. Exam simulation:\n"
        f"   Total: {exam_score:.2f}\n"
        f"   Choice: {choice_score_gain}/{choice_score_sum}\n"
        f"   Filling: {filling_score_gain}/{filling_score_sum}\n"
        f"   Proving: {proving_score_gain}/{proving_score_sum}\n"
    )

    metric_dict = {
        "Accuracy": {
            "accuracy": f"{correct / len(answers):.2f}",
            "accuracy_within_16k_tokens": f"{correct_16k / len(answers):.2f}",
            "latency_weighted_accuracy": f"{correct_arl / len(answers):.2f}",
        },
        "Efficiency": {
            "throughput_tokens_per_sec": f"{thrput:.2f}",
            "avg_solve_time_sec": f"{ave_time:.2f}",
            "avg_response_length_tokens": f"{ave_length:.2f}",
        },
        "Exam simulation": {
            "total_score": f"{exam_score:.2f}",
            "multiple_choice_score": f"{choice_score_gain}/{choice_score_sum}",
            "non_multiple_choice_score": f"{filling_score_gain}/{filling_score_sum}",
        },
    }
    save_dict_to_excel(model, run_id, metric_dict, metrics_path)


def _resolve_records(*, run_id, split, subject, json_path, limit, ids):
    """Mirror solve._resolve_records so grading reads the same source."""
    if split:
        return _loader_load(
            split=split, subject=subject, limit=limit, ids=ids,
        )
    if json_path:
        return _loader_load(
            json_path=json_path, subject=subject or "", limit=limit,
        )
    raise FileNotFoundError(
        f"No question source available for run_id={run_id!r}. Pass --split or --json."
    )


def parallel_predict(run_id, models, *, split=None, subject=None, json_path=None,
                     limit=None, ids=None):
    """Solve a run with multiple models in parallel."""
    print(f"Solving {run_id} with {models} in parallel...")
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(
                predict,
                run_id=run_id, model=model, max_tokens=32 * 1024,
                max_workers=5, mode=mode_cfg,
                split=split, subject=subject, json_path=json_path,
                limit=limit, ids=ids,
            )
            for model in models
        ]
        wait(futures)
        results = [f.result() for f in futures]
        print("Done:", results)


def parallel_check(run_id, models, *, split=None, subject=None, json_path=None,
                   limit=None, ids=None):
    """Grade run predictions for multiple models in parallel."""
    print(f"Checking {models} on {run_id} in parallel...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                check, run_id=run_id, model=model,
                split=split, subject=subject, json_path=json_path,
                limit=limit, ids=ids,
            )
            for model in models
        ]
        wait(futures)
        results = [f.result() for f in futures]
        print("Done:", results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Grade model predictions on a run using a verifier ensemble."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--split", help="HuggingFace split (e.g. en_2603, zh_2605).")
    src.add_argument("--json", dest="json_path",
                     help="Local JSON file produced by analyze/analyze.py.")

    parser.add_argument("--subject",
                        help="Subject filter for --split (math/physics/chemistry/biology).")
    parser.add_argument("--limit", type=int,
                        help="Take only the first N questions after filtering.")
    parser.add_argument("--ids", help="Comma-separated list of question ids to keep.")
    parser.add_argument("--run-id",
                        help="Override the prediction filename stem "
                        "(default: <split>__<subject>).")

    parser.add_argument("--model", default=None,
                        help="Single model to grade. If omitted, all `solvers` are graded.")
    args = parser.parse_args()

    if args.run_id:
        run_id = args.run_id
    elif args.split:
        run_id = split_subject_to_run_id(args.split, args.subject)
    else:
        run_id = os.path.splitext(os.path.basename(args.json_path))[0]

    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None

    common_kwargs = dict(
        split=args.split, subject=args.subject, json_path=args.json_path,
        limit=args.limit, ids=ids,
    )
    if args.model:
        check(run_id=run_id, model=args.model, **common_kwargs)
    else:
        parallel_check(run_id=run_id, models=solvers, **common_kwargs)
