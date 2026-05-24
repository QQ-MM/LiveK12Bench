"""Process-level evaluation: critique each model's reasoning chain and compute
ARL_w (token-weighted accuracy proxy), error counts, and per-error-type
indicators (CIE / LAE / DRE), plus PES / OES scores.

This pipeline runs after `evaluate.py` has computed outcome scores and runs
in parallel across questions per model. Per-question process metrics are
written back into the prediction JSON under the `metrics` key.
"""

import os
import re
import json
import random
import threading
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from util.llm import call_llm
from util.dataset_loader import (
    load_questions as _loader_load,
    split_subject_to_run_id,
)
from prompts import process_eval_prompt_template
from constants import prediction_dir, solvers as DEFAULT_MODEL_LIST

random.seed(42)

# Models in this list are used to compute the per-question token-count baseline
# for the ARL_w metric. Override at call site if your model set differs.
MODEL_LIST_ARL = list(DEFAULT_MODEL_LIST)

# Critic models for process evaluation. Adjust as needed.
PROCESS_CRITICS = ["claude-sonnet-4-6", "gemini-3-flash"]

# Thread-safe printing.
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def get_token_count(metadata: Dict) -> int:
    """Best-effort token count: usage.completion_tokens > tokens > 5000."""
    try:
        if metadata:
            usage = metadata.get("usage")
            if usage and isinstance(usage, dict):
                completion_tokens = usage.get("completion_tokens")
                if completion_tokens is not None:
                    return int(completion_tokens)
            tokens = metadata.get("tokens")
            if tokens is not None:
                return int(tokens)
    except (TypeError, ValueError) as e:
        safe_print(f"Warning: error parsing token count: {e}")
    return 5000


def calculate_arl_w(current_tokens: int, all_model_tokens: List[int]) -> float:
    """ARL_w = mean(other-model tokens on this question) / current model's tokens."""
    if not all_model_tokens or current_tokens <= 0:
        return 1.0
    return sum(all_model_tokens) / len(all_model_tokens) / current_tokens


def extract_boxes(response: str) -> List[int]:
    """Extract integer values from `\\boxed{N}` (or `\\\\boxed{N}`) in `response`."""
    for pattern in (r"\\boxed\{(\d+)\}", r"\\\\boxed\{(\d+)\}"):
        matches = re.findall(pattern, response)
        if matches:
            return [int(m) for m in matches]
    return []


def call_llm_for_evaluation(question: str, gold_solution: str, answer: str) -> str:
    """Run the process-evaluation prompt against a randomly-chosen critic model."""
    prompt_text = process_eval_prompt_template.format(
        question=question, gold_solution=gold_solution, answer=answer
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    critic = random.choice(PROCESS_CRITICS)
    response, _ = call_llm(messages=messages, model=critic, max_tokens=16 * 1024)
    return response


def collect_all_tokens(pred_dir: str, model_list: List[str], paper: str) -> Dict[str, List[int]]:
    """Collect per-question token counts across `model_list` for ARL_w baseline."""
    out: Dict[str, List[int]] = {}
    for model in model_list:
        pred_file = os.path.join(pred_dir, model, f"{paper}.json")
        if not os.path.exists(pred_file):
            continue
        with open(pred_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        for q_id, pred_data in predictions.items():
            out.setdefault(q_id, []).append(get_token_count(pred_data.get("metadata", {})))
    return out


def process_single_question(
    q_id: str,
    pred_data: Dict,
    question_info: Dict,
    all_tokens: List[int],
) -> Tuple[str, Dict[str, Any], bool]:
    """Run process evaluation for one question. Returns (q_id, new_metrics, success)."""
    try:
        current_tokens = get_token_count(pred_data.get("metadata", {}))
        arl_w = calculate_arl_w(current_tokens, all_tokens)

        question = question_info.get("question", "")
        gold_solution = question_info.get("solution", "")
        answer = pred_data.get("prediction", "")
        score = float(question_info.get("point_value", 10) or 10)

        try:
            response = call_llm_for_evaluation(question, gold_solution, answer)
        except Exception as e:
            safe_print(f"    ❌ LLM call failed for question {q_id}: {e}")
            response = ""

        boxes = extract_boxes(response)
        if len(boxes) >= 4:
            error_num, cie, lae, dre = boxes[0], bool(boxes[1]), bool(boxes[2]), bool(boxes[3])
        else:
            safe_print(f"    ⚠️  Could not extract 4 boxes from response for question {q_id}, found: {boxes}")
            error_num = boxes[0] if len(boxes) > 0 else 0
            cie = bool(boxes[1]) if len(boxes) > 1 else False
            lae = bool(boxes[2]) if len(boxes) > 2 else False
            dre = bool(boxes[3]) if len(boxes) > 3 else False

        outcome_score = pred_data.get("metrics", {}).get("outcome_score", 0) or 0

        # Aggressive scoring: PES collapses to 0 if any error is present;
        # OES tightens further when more than one error is detected.
        if error_num == 0:
            pes = score / 2 * outcome_score / score
        else:
            pes = 0.0
            if error_num > 1:
                outcome_score = 0
        oes = outcome_score / 2 + pes if score > 0 else 0

        new_metrics = {
            "ARL_w": arl_w,
            "process_eval": response,
            "error_num": error_num,
            "CIE": cie,
            "LAE": lae,
            "DRE": dre,
            "PES": pes,
            "OES": oes,
        }

        safe_print(f"    ✓ Question {q_id} done (errors={error_num}, PES={pes:.2f}, OES={oes:.2f}, MAX_S={score})")
        return q_id, new_metrics, True

    except Exception as e:
        safe_print(f"    ❌ Error processing question {q_id}: {e}")
        return q_id, {}, False


def process_model_predictions(
    model: str,
    pred_file: str,
    questions: List[Dict],
    all_tokens_per_question: Dict[str, List[int]],
    skip_existing: bool = False,
    max_workers: int = 5,
    save_interval: int = 10,
) -> bool:
    """Run process evaluation for all questions of a single model concurrently."""
    with open(pred_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    tasks = []
    for q_id, pred_data in predictions.items():
        try:
            q_idx = int(q_id)
        except ValueError:
            safe_print(f"    ⚠️  Invalid question id: {q_id}")
            continue

        if q_idx < 0 or q_idx >= len(questions):
            safe_print(f"    ⚠️  Question index out of range: {q_id} (total: {len(questions)})")
            continue

        if skip_existing and pred_data.get("metrics", {}).get("process_eval"):
            continue

        question_info = questions[q_idx]
        all_tokens = all_tokens_per_question.get(
            q_id, [get_token_count(pred_data.get("metadata", {}))]
        )
        tasks.append((q_id, pred_data, question_info, all_tokens))

    if not tasks:
        safe_print(f"  ℹ️  No tasks to process for model {model}")
        return False

    safe_print(f"  📋 Processing {len(tasks)} questions with {max_workers} workers...")

    predictions_lock = threading.Lock()
    processed_count = 0
    failed_count = 0
    modified = False

    def save_predictions():
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        safe_print(f"  💾 Auto-saved: {pred_file} (processed: {processed_count})")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_qid = {
            executor.submit(process_single_question, q_id, pred_data, qi, tokens): q_id
            for q_id, pred_data, qi, tokens in tasks
        }
        for fut in as_completed(future_to_qid):
            q_id = future_to_qid[fut]
            try:
                q_id, new_metrics, success = fut.result()
                with predictions_lock:
                    if success and new_metrics:
                        if "metrics" not in predictions[q_id]:
                            predictions[q_id]["metrics"] = {}
                        predictions[q_id]["metrics"].update(new_metrics)
                        modified = True
                        processed_count += 1
                    else:
                        failed_count += 1
                    if modified and processed_count % save_interval == 0:
                        save_predictions()
            except Exception as e:
                safe_print(f"    ❌ Exception for question {q_id}: {e}")
                failed_count += 1

    if modified:
        save_predictions()

    safe_print(
        f"  📈 Processed: {processed_count}, Failed: {failed_count}, "
        f"Skipped: {len(predictions) - len(tasks)}"
    )
    return modified


def process_predictions(
    pred_dir: str,
    model_list: List[str],
    runs: List[Dict[str, Any]],
    skip_existing: bool = False,
    max_workers: int = 5,
):
    """Run process evaluation across all (run, model) pairs.

    Each entry of ``runs`` describes one evaluation run::

        {
            "run_id": "en_2603__math",
            "split": "en_2603",       # optional (HF source)
            "subject": "math",         # optional
            "json_path": None,         # optional (analyze.py output)
            "limit": None,             # optional row cap
            "ids": None,               # optional id whitelist
        }

    The ``run_id`` is the prediction filename stem under ``<pred_dir>/<model>/``.
    """
    for run in runs:
        run_id = run["run_id"]
        print(f"\n{'=' * 60}")
        print(f"Processing run: {run_id}")
        print(f"{'=' * 60}")

        try:
            questions = _resolve_records(
                run_id=run_id,
                split=run.get("split"),
                subject=run.get("subject"),
                json_path=run.get("json_path"),
                limit=run.get("limit"),
                ids=run.get("ids"),
            )
        except FileNotFoundError as e:
            print(f"  ❌ {e}")
            continue

        print(f"  📝 Loaded {len(questions)} questions")

        all_tokens_per_question = collect_all_tokens(pred_dir, MODEL_LIST_ARL, run_id)
        print(f"  📊 Collected token counts for {len(all_tokens_per_question)} questions")

        for model in model_list:
            pred_file = os.path.join(pred_dir, model, f"{run_id}.json")
            if not os.path.exists(pred_file):
                print(f"  ⚠️  Prediction file not found for model {model}")
                continue

            print(f"\n  🤖 Processing model: {model}")
            process_model_predictions(
                model=model,
                pred_file=pred_file,
                questions=questions,
                all_tokens_per_question=all_tokens_per_question,
                skip_existing=skip_existing,
                max_workers=max_workers,
            )


def _resolve_records(*, run_id, split, subject, json_path, limit, ids):
    """Mirror the resolution logic used by solve.py / evaluate.py."""
    if split:
        return _loader_load(
            split=split, subject=subject, limit=limit, ids=ids,
        )
    if json_path:
        return _loader_load(
            json_path=json_path, subject=subject or "", limit=limit,
        )
    raise FileNotFoundError(
        f"No question source for run_id={run_id!r}. Pass --split or --json."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run process-level evaluation (CIE / LAE / DRE / PES / OES) on predictions."
    )
    parser.add_argument("--pred-dir", default=prediction_dir,
                        help="Predictions root directory.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_LIST,
                        help="Models to evaluate.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--split", help="HuggingFace split (e.g. en_2603).")
    src.add_argument("--json", dest="json_path",
                     help="Local JSON file produced by analyze/analyze.py.")

    parser.add_argument("--subject", help="Subject filter for --split.")
    parser.add_argument("--limit", type=int,
                        help="Take only the first N questions after filtering.")
    parser.add_argument("--ids", help="Comma-separated list of question ids to keep.")
    parser.add_argument("--run-id",
                        help="Override the prediction filename stem.")

    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip questions whose process_eval is already populated.")
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()

    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None

    if args.run_id:
        run_id = args.run_id
    elif args.split:
        run_id = split_subject_to_run_id(args.split, args.subject)
    else:
        run_id = os.path.splitext(os.path.basename(args.json_path))[0]
    runs = [{
        "run_id": run_id,
        "split": args.split,
        "subject": args.subject,
        "json_path": args.json_path,
        "limit": args.limit,
        "ids": ids,
    }]

    print("🚀 Starting prediction evaluation...")
    print(f"📁 Predictions: {args.pred_dir}")
    print(f"🤖 Models:       {args.models}")
    print(f"📝 Runs:         {[r['run_id'] for r in runs]}")
    print(f"🔄 Max workers:  {args.max_workers}")

    process_predictions(
        pred_dir=args.pred_dir,
        model_list=args.models,
        runs=runs,
        skip_existing=args.skip_existing,
        max_workers=args.max_workers,
    )
    print("\n✅ Processing completed!")
