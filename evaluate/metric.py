"""Aggregate per-question metrics across models into an Excel summary.

Reads `metrics` blocks written into prediction JSONs by `evaluate.py` and
`evaluate_process.py`, computes paper-level aggregates per model, and merges
them into a multi-sheet Excel workbook (one sheet per paper, plus an optional
`--subset` mode that aggregates over questions tagged with a particular
subset).

A question belongs to subset `<name>` if either:
    * its `subset` field is a list[str] and contains `<name>`, or
    * `q[<name>]` is a truthy boolean (legacy form).

Usage examples:
    # Normal mode: aggregate one paper across all models
    python evaluate/metric.py --paper math_extention

    # Subset mode: aggregate questions tagged 'complex_layout' across papers
    python evaluate/metric.py --subset --subset-field complex_layout \\
        --papers biology_extention chemistry_extention
"""

import os
import json
import math
import random
import argparse
from pathlib import Path

import pandas as pd

from constants import paper_dir as DEFAULT_QUESTIONS_DIR
from constants import prediction_dir as DEFAULT_PRED_DIR
from constants import metrics_path as DEFAULT_OUTPUT_XLSX
from constants import solvers as DEFAULT_MODELS
from util.dataset_loader import (
    load_questions as _loader_load,
    split_subject_to_run_id,
)

# ===================== Runtime configuration =====================
# These defaults come from constants.py; they can be overridden via CLI flags
# in main(). Initializing them to real defaults (rather than empty strings)
# makes the module safe to import and use programmatically without going
# through the argparse entry point.
PAPER = ""
MODELS = list(DEFAULT_MODELS)
ROOT_DIR = DEFAULT_PRED_DIR
QUESTIONS_DIR = DEFAULT_QUESTIONS_DIR
OUTPUT_XLSX = DEFAULT_OUTPUT_XLSX
ENABLE_FILTERING = False
EXCLUSION_PROBABILITY = 1
REFRESH_EXCLUDED = False
ENABLE_SUBSET_MODE = False
SUBSET_FIELD = ""
PAPER_LIST = []

# HuggingFace dataset source (overrides QUESTIONS_DIR lookup when non-empty).
HF_SPLIT = ""
HF_SUBJECT = ""
HF_LIMIT = None

# ===================== Utility helpers =====================

def load_json(model: str, paper: str) -> dict:
    """Load the prediction JSON for one model on one paper."""
    json_path = Path(ROOT_DIR) / model / f"{paper}.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_json(model: str, paper: str, data: dict):
    """Write a model's prediction JSON."""
    json_path = Path(ROOT_DIR) / model / f"{paper}.json"
    # Ensure the parent directory exists.
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_questions(paper: str) -> list:
    """Load question records for a run.

    Resolution order:
      1. If ``HF_SPLIT`` is set (via CLI), pull from the HuggingFace dataset.
      2. Otherwise read ``QUESTIONS_DIR/<paper>.json`` from disk.

    Returns ``None`` if no source resolves — callers fall back to the
    legacy ``point_value=0`` behaviour, which is harmless for downstream
    aggregation.
    """
    if HF_SPLIT:
        try:
            return _loader_load(
                split=HF_SPLIT, subject=HF_SUBJECT or None, limit=HF_LIMIT,
            )
        except Exception as exc:  # pragma: no cover - surface but don't crash
            print(f"  ⚠️  HF dataset load failed for {HF_SPLIT}/{HF_SUBJECT}: {exc}")
            return None
    json_path = Path(QUESTIONS_DIR) / f"{paper}.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def get_question_score(questions: list, question_id: str) -> float:
    """
    Return the point_value for the given question id.

    ``question_id`` is a numeric string ("0", "5", ...) matching the
    index in ``questions``.
    """
    if questions is None:
        return 0
    try:
        idx = int(question_id)
        if 0 <= idx < len(questions):
            q = questions[idx]
            # Accept both English and legacy Chinese key.
            return q.get("point_value", 0) or 0
    except (ValueError, TypeError):
        pass
    return 0


def get_all_question_ids(models: list, paper: str) -> list:
    """Collect the union of question ids seen across all models."""
    question_ids = set()
    for model in models:
        data = load_json(model, paper)
        if data:
            question_ids.update(data.keys())
    # Sort numerically when possible; fall back to lexical ordering.
    try:
        return sorted(question_ids, key=lambda x: int(x))
    except ValueError:
        return sorted(question_ids)

def get_token_count(metadata) -> int:
    """
    Return the response token count for one prediction.

    Resolution order:
        metadata.usage.completion_tokens > metadata.tokens > 5000.
    """
    try:
        if metadata:
            # Prefer usage.completion_tokens when available.
            usage = metadata.get("usage")
            if usage and isinstance(usage, dict):
                completion_tokens = usage.get("completion_tokens")
                if completion_tokens is not None:
                    return int(completion_tokens)
            
            # Fall back to the flat 'tokens' field.
            tokens = metadata.get("tokens")
            if tokens is not None:
                return int(tokens)
    except (TypeError, ValueError) as e:
        print(f"Warning: Error parsing token count: {e}")
    
    return 5000


# ===================== Sample-filtering logic =====================

def determine_excluded_questions(models: list, paper: str, probability: float) -> dict:
    """
    Decide which question ids to exclude from aggregation.

    Rule: if every model scored zero on a question, drop it with the
    configured probability.
    """
    # Load prediction data for every model.
    all_data = {}
    for model in models:
        data = load_json(model, paper)
        if data:
            all_data[model] = data
    
    if not all_data:
        print("Warning: no model data found.")
        return {}
    
    question_ids = get_all_question_ids(models, paper)
    excluded = {}
    
    for qid in question_ids:
        # Check whether every model scored zero on this question.
        zero_count = 0
        has_data = False
        
        for model in models:
            if model in all_data and qid in all_data[model]:
                has_data = True
                metrics = all_data[model][qid].get('metrics', {})
                outcome_score = metrics.get('outcome_score', None)
                # An outcome_score that is neither 0 nor None means at least one model scored on this question.
                if outcome_score is not None and outcome_score == 0:
                    zero_count += 1
        
        if zero_count >= 8 and has_data:
            # Exclude with the configured probability.
            excluded[qid] = random.random() < probability
        else:
            excluded[qid] = False
    
    return excluded


def update_excluded_field(models: list, paper: str, excluded: dict):
    """Write the excluded mask into every model's prediction JSON."""
    for model in models:
        data = load_json(model, paper)
        if data:
            for qid in data.keys():
                if 'metrics' not in data[qid]:
                    data[qid]['metrics'] = {}
                data[qid]['metrics']['excluded'] = excluded.get(qid, False)
            save_json(model, paper, data)
            print(f"  Updated excluded mask in {model}/{paper}.json")


def get_existing_excluded(models: list, paper: str) -> dict:
    """Read the excluded mask from existing model JSONs."""
    excluded = {}
    for model in models:
        data = load_json(model, paper)
        if data:
            for qid, value in data.items():
                if qid not in excluded:
                    metrics = value.get('metrics', {})
                    excluded[qid] = metrics.get('excluded', False)
    return excluded


# ===================== Metric computation =====================

def calculate_metrics(model: str, paper: str, excluded: dict, questions: list) -> dict:
    """Compute per-model metric aggregates for one paper.

    Metrics:
        - accuracy:                  fraction of questions with correctness=True
        - latency_weighted_accuracy: sum(correctness * ARL_w) / sample_count
        - error_count:               mean error_num
        - PES:                       100 * sum(PES) / sum(point_value) / 2
        - CIE / LAE / DRE:           fraction of False labels
        - total_score (OES):         100 * sum(OES) / sum(point_value)
        - outcome_score:             100 * sum(outcome_score) / sum(point_value)
    """
    data = load_json(model, paper)
    if not data:
        print(f"Warning: failed to load {model}/{paper}.json")
        return None
    
    # Filter out questions in the excluded mask.
    filtered_data = {
        qid: value for qid, value in data.items() 
        if not excluded.get(qid, False)
    }
    
    if not filtered_data:
        print(f"Warning: {model} has no valid samples after filtering")
        return None
    
    sample_count = len(filtered_data)
    
    # Aggregate total point value over surviving samples.
    total_score = sum(get_question_score(questions, qid) for qid in filtered_data.keys())
    
    # Initialise accumulators.
    correctness_count = 0
    arl_w_weighted_sum = 0
    error_num_sum = 0
    error_num_count = 0
    
    pes_sum = 0
    
    cie_false_count = 0
    cie_total = 0
    lae_false_count = 0
    lae_total = 0
    dre_false_count = 0
    dre_total = 0
    
    oes_sum = 0
    oes_r_sum = 0
    outcome_score_sum = 0
    
    for qid, value in filtered_data.items():
        metrics = value.get('metrics', {})

        # Token-count statistics.
        metadata = value.get('metadata', {})
        token_num = get_token_count(metadata)
        
        # (2) Accuracy
        if metrics.get('correctness', False):
            correctness_count += 1
        
        # (3) Latency-weighted accuracy
        correctness_val = 1 if metrics.get('correctness', False) else 0
        arl_w = metrics.get('ARL_w')
        if arl_w is not None:
            arl_w_weighted_sum += correctness_val * (1 + 0.1 * math.log(arl_w))
        
        # (4) Error count
        error_num = metrics.get('error_num')
        if error_num is not None:
            error_num_sum += error_num
            error_num_count += 1
        
        # (5) PES
        pes = metrics.get('PES')
        if pes is not None:
            pes_sum += pes
        
        # (6) CIE/LAE/DRE - fraction of false labels
        if 'CIE' in metrics:
            cie_total += 1
            if metrics['CIE'] == False:
                cie_false_count += 1
        
        if 'LAE' in metrics:
            lae_total += 1
            if metrics['LAE'] == False:
                lae_false_count += 1
        
        if 'DRE' in metrics:
            dre_total += 1
            if metrics['DRE'] == False:
                dre_false_count += 1
        
        # (7) Total score (OES)
        oes = metrics.get('OES')
        if oes is not None:
            oes_sum += oes
            if token_num <= 12*1024:
                oes_r_sum += oes
        
        # (8) Outcome score
        outcome_score = metrics.get('outcome_score')
        if outcome_score is not None:
            outcome_score_sum += outcome_score
    
    # Compute final metrics.
    result = {
        'accuracy': round(100 * correctness_count / sample_count, 1) if sample_count > 0 else 0,
        'latency_weighted_accuracy': round(100 * arl_w_weighted_sum / sample_count, 1) if sample_count > 0 else 0,
        'error_count': round(error_num_sum / error_num_count, 4) if error_num_count > 0 else 0,
        'PES': round(100 * pes_sum / (total_score / 2), 1) if total_score > 0 else 0,
        'CIE': round(100 * cie_false_count / cie_total, 1) if cie_total > 0 else None,
        'LAE': round(100 * lae_false_count / lae_total, 1) if lae_total > 0 else None,
        'DRE': round(100 * dre_false_count / dre_total, 1) if dre_total > 0 else None,
        'total_score': round(100 * oes_sum / total_score, 1) if total_score > 0 else 0,
        'outcome_score': round(100 * outcome_score_sum / total_score, 1) if total_score > 0 else 0,
        'OES_r': round(100 * oes_r_sum / total_score, 1) if total_score > 0 else 0,
    }
    
    return result


# ===================== Subset-mode helpers =====================

def get_subset_samples(paper_list: list, subset_field: str) -> dict:
    """Collect questions that match the subset filter, grouped by paper.

    Returns ``{paper: {qid: question_data, ...}, ...}``.
    """
    subset_samples = {}
    total_count = 0
    
    for paper in paper_list:
        questions = load_questions(paper)
        if questions:
            valid_questions = {}
            for i, q in enumerate(questions):
                # A question is in the slice if any of:
                #   1. q['subset'] is a list and contains subset_field   (canonical form)
                #   2. q[subset_field] is truthy                          (legacy: per-field boolean)
                tags = q.get("subset")
                in_slice = (
                    (isinstance(tags, list) and subset_field in tags)
                    or bool(q.get(subset_field, False))
                )
                if in_slice:
                    valid_questions[str(i)] = q
            if valid_questions:
                subset_samples[paper] = valid_questions
                total_count += len(valid_questions)
                print(f"    {paper}: {len(valid_questions)} questions in subset '{subset_field}'")
    
    print(f"  Total: {total_count} questions")
    return subset_samples


def get_excluded_for_subset(models: list, paper_list: list) -> dict:
    """Collect per-paper excluded masks for subset mode.

    Returns ``{paper: {qid: bool, ...}, ...}``.
    """
    excluded_per_paper = {}
    for paper in paper_list:
        excluded_per_paper[paper] = get_existing_excluded(models, paper)
    return excluded_per_paper


def calculate_metrics_for_subset(model: str, subset_samples: dict, excluded_per_paper: dict) -> dict:
    """Compute metrics for one model over a subset-filtered sample.

    Args:
        model:               Model name.
        subset_samples:      ``{paper: {qid: question_data, ...}, ...}``.
        excluded_per_paper:  ``{paper: {qid: bool, ...}, ...}``.
    """
    # Initialise accumulators.
    sample_count = 0
    total_score = 0
    
    correctness_count = 0
    arl_w_weighted_sum = 0
    error_num_sum = 0
    error_num_count = 0
    
    pes_sum = 0
    
    cie_false_count = 0
    cie_total = 0
    lae_false_count = 0
    lae_total = 0
    dre_false_count = 0
    dre_total = 0
    
    oes_sum = 0
    outcome_score_sum = 0
    
    # ACC_r aggregation accumulators.
    r_values = [90, 80, 70, 60, 50, 40, 30, 20, 10]
    acc_r_counts = {r: 0 for r in r_values}   # correct samples that fit each token threshold
    acc_r_totals = {r: 0 for r in r_values}   # total samples that fit each token threshold
    
    # Iterate over every (paper, subset_sample) pair.
    for paper, questions_dict in subset_samples.items():
        # Load this paper's predictions for the model.
        model_data = load_json(model, paper)
        if not model_data:
            print(f"    Warning: failed to load {model}/{paper}.json")
            continue
        
        excluded = excluded_per_paper.get(paper, {})
        
        for qid, question_info in questions_dict.items():
            # Skip questions in the excluded mask.
            if excluded.get(qid, False):
                continue
            
            # Skip questions that this model never produced a prediction for.
            if qid not in model_data:
                continue
            
            sample_count += 1
            question_score = question_info.get('point_value', 0)
            total_score += question_score
            
            metrics = model_data[qid].get('metrics', {})
            
            # Pull the token count for ACC_r aggregation.
            metadata = model_data[qid].get('metadata', {})
            token_count = get_token_count(metadata)
            
            # (2) Accuracy
            if metrics.get('correctness', False):
                correctness_count += 1

            # ACC_r: only samples whose token count is at or below the threshold.
            is_correct = metrics.get('correctness', False)
            for r in r_values:
                threshold = 32 * 1024 * r / 100
                if token_count <= threshold:
                    acc_r_totals[r] += 1
                    if is_correct:
                        acc_r_counts[r] += 1
            
            # (3) Latency-weighted accuracy
            correctness_val = 1 if metrics.get('correctness', False) else 0
            arl_w = metrics.get('ARL_w')
            if arl_w is not None:
                arl_w_weighted_sum += correctness_val * (1 + 0.1 * math.log(arl_w))
            
            # (4) Error count
            error_num = metrics.get('error_num')
            if error_num is not None:
                error_num_sum += error_num
                error_num_count += 1
            
            # (5) PES
            pes = metrics.get('PES')
            if pes is not None:
                pes_sum += pes
            
            # (6) CIE/LAE/DRE - fraction of false labels
            if 'CIE' in metrics:
                cie_total += 1
                if metrics['CIE'] == False:
                    cie_false_count += 1
            
            if 'LAE' in metrics:
                lae_total += 1
                if metrics['LAE'] == False:
                    lae_false_count += 1
            
            if 'DRE' in metrics:
                dre_total += 1
                if metrics['DRE'] == False:
                    dre_false_count += 1
            
            # (7) Total score (OES)
            oes = metrics.get('OES')
            if oes is not None:
                oes_sum += oes
            
            # (8) Outcome score
            outcome_score = metrics.get('outcome_score')
            if outcome_score is not None:
                outcome_score_sum += outcome_score
    
    if sample_count == 0:
        print(f"    Warning: {model} has no valid samples in the subset")
        return None
    
    # Compute final metrics.
    result = {
        'accuracy': round(100 * correctness_count / sample_count, 1) if sample_count > 0 else 0,
        'latency_weighted_accuracy': round(100 * arl_w_weighted_sum / sample_count, 1) if sample_count > 0 else 0,
        'error_count': round(error_num_sum / error_num_count, 4) if error_num_count > 0 else 0,
        'PES': round(100 * pes_sum / (total_score / 2), 1) if total_score > 0 else 0,
        'CIE': round(100 * cie_false_count / cie_total, 1) if cie_total > 0 else None,
        'LAE': round(100 * lae_false_count / lae_total, 1) if lae_total > 0 else None,
        'DRE': round(100 * dre_false_count / dre_total, 1) if dre_total > 0 else None,
        'total_score': round(100 * oes_sum / total_score, 1) if total_score > 0 else 0,
        'outcome_score': round(100 * outcome_score_sum / total_score, 1) if total_score > 0 else 0,
        'sample_count': sample_count,
        'total_point_value': total_score,
    }
    
      # ACC_r metric
    for r in r_values:
        col_name = f'ACC_{r}'
        result[col_name] = round(100 * acc_r_counts[r] / sample_count, 1) if acc_r_totals[r] > 0 else None
    
    return result


def update_xlsx_for_subset(subset_field: str, models: list, metrics_dict: dict, xlsx_path: str):
    """Write subset-mode metrics into the workbook.

    The worksheet name is the subset field itself.
    """
    sheets = {}
    
    # Read the existing workbook (if any).
    if os.path.exists(xlsx_path):
        try:
            xlsx = pd.ExcelFile(xlsx_path)
            for sheet_name in xlsx.sheet_names:
                sheets[sheet_name] = xlsx.parse(sheet_name, index_col=0)
            print(f"Read existing workbook: {xlsx_path}")
        except Exception as e:
            print(f"Error reading workbook: {e}")
            sheets = {}
    
    # Open (or create) the worksheet for this subset.
    if subset_field in sheets:
        df = sheets[subset_field].copy()
    else:
        df = pd.DataFrame()
    
    # Columns to write into the worksheet.
    # columns_to_update = ['accuracy', 'latency_weighted_accuracy', 'error_count', 'PES', 'CIE', 'LAE', 'DRE', 'total_score', 'outcome_score', 'sample_count', 'total_point_value']
    columns_to_update = [
        'accuracy', 'latency_weighted_accuracy', 'error_count', 'PES', 'CIE', 'LAE', 'DRE', 'total_score', 'outcome_score', 'sample_count', 'total_point_value',
        'ACC_90', 'ACC_80', 'ACC_70', 'ACC_60', 'ACC_50', 'ACC_40', 'ACC_30', 'ACC_20', 'ACC_10'
    ]
    
    # Update the row for this run.
    for model, metrics in metrics_dict.items():
        if metrics:
            for col in columns_to_update:
                if col in metrics and metrics[col] is not None:
                    df.loc[model, col] = metrics[col]
    
    sheets[subset_field] = df
    
    # Save the workbook.
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name)
    
    print(f"Saved results to {xlsx_path}, sheet: {subset_field}")


# ===================== Excel writer =====================

def update_xlsx(paper: str, models: list, metrics_dict: dict, xlsx_path: str):
    """Write per-paper metrics into the workbook.

    One worksheet per paper; rows are indexed by model name.
    """
    sheets = {}
    
    # Read the existing workbook (if any).
    if os.path.exists(xlsx_path):
        try:
            xlsx = pd.ExcelFile(xlsx_path)
            for sheet_name in xlsx.sheet_names:
                sheets[sheet_name] = xlsx.parse(sheet_name, index_col=0)
            print(f"Read existing workbook: {xlsx_path}")
        except Exception as e:
            print(f"Error reading workbook: {e}")
            sheets = {}
    
    # Open (or create) the worksheet for this run.
    if paper in sheets:
        df = sheets[paper].copy()
    else:
        df = pd.DataFrame()
    
    # Columns to write into the worksheet (other columns are preserved).
    columns_to_update = ['accuracy', 'latency_weighted_accuracy', 'error_count', 'PES', 'CIE', 'LAE', 'DRE', 'total_score', 'outcome_score', 'OES_r']
    # columns_to_update = ['OES_r']
    
    # Update the row for this run.
    for model, metrics in metrics_dict.items():
        if metrics:
            for col in columns_to_update:
                if col in metrics and metrics[col] is not None:
                    df.loc[model, col] = metrics[col]
    
    sheets[paper] = df
    
    # Save the workbook.
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name)
    
    print(f"Saved results to {xlsx_path}, sheet: {paper}")


# ===================== Main entry points =====================

def main_subset_mode():
    """Entry point for subset mode."""
    print("=" * 60)
    print(f"[Subset mode] Starting metric aggregation")
    print(f"Subset field: {SUBSET_FIELD}")
    print(f"Papers: {PAPER_LIST}")
    print(f"Models: {MODELS}")
    print("=" * 60)
    
    # Step 0: Collect subset samples
    print("\n[Step 0] Collecting subset samples...")
    subset_samples = get_subset_samples(PAPER_LIST, SUBSET_FIELD)
    
    if not subset_samples:
        print("Error: no samples matched the subset filter.")
        return
    
    # Step 1: Collect per-paper excluded masks
    print("\n[Step 1] Loading excluded mask...")
    excluded_per_paper = get_excluded_for_subset(MODELS, PAPER_LIST)
    for paper, excluded in excluded_per_paper.items():
        excluded_count = sum(excluded.values())
        print(f"  {paper}: {excluded_count} questions excluded")
    
    # Step 2: Compute per-model metrics
    print("\n[Step 2] Computing per-model metrics...")
    metrics_dict = {}
    for model in MODELS:
        print(f"\n  Computing metrics for {model}...")
        metrics = calculate_metrics_for_subset(model, subset_samples, excluded_per_paper)
        metrics_dict[model] = metrics
        if metrics:
            for key, value in metrics.items():
                if value is not None:
                    print(f"    {key}: {value}")
    
    # Step 3: Update Excel workbook
    print("\n[Step 3] Updating Excel workbook...")
    update_xlsx_for_subset(SUBSET_FIELD, MODELS, metrics_dict, OUTPUT_XLSX)
    
    print("\n" + "=" * 60)
    print("[Subset mode] Aggregation complete.")
    print("=" * 60)


def main_normal_mode():
    """Entry point for normal (per-paper) mode."""
    print("=" * 60)
    print(f"Starting metric aggregation")
    print(f"Paper: {PAPER}")
    print(f"Models: {MODELS}")
    print("=" * 60)
    
    # Step 0: Load question data
    print("\n[Step 0] Loading questions...")
    questions = load_questions(PAPER)
    if questions:
        # point_value field (English schema from HF dataset / dataset_loader).
        total_score = sum(
            (q.get("point_value") or 0) for q in questions
        )
        print(f"  Loaded {len(questions)} questions; total point value: {total_score}")
    else:
        print(f"  Warning: failed to load questions from {QUESTIONS_DIR}/{PAPER}.json")
        questions = []
    
    # Step 1: Determine excluded mask
    excluded = {}
    if ENABLE_FILTERING:
        if REFRESH_EXCLUDED:
            print("\n[Step 1] Regenerating excluded mask...")
            excluded = determine_excluded_questions(MODELS, PAPER, EXCLUSION_PROBABILITY)
            excluded_count = sum(excluded.values())
            total_count = len(excluded)
            print(f"  {total_count} questions total, {excluded_count} excluded (exclusion probability: {EXCLUSION_PROBABILITY})")
            
            # Surface excluded question ids.
            excluded_qids = [qid for qid, is_excluded in excluded.items() if is_excluded]
            if excluded_qids:
                print(f"  Excluded question ids: {excluded_qids}")
            
            # Persist the excluded mask back into every model's prediction JSON.
            print("  Updating excluded mask inside model JSONs...")
            update_excluded_field(MODELS, PAPER, excluded)
        else:
            print("\n[Step 1] Using existing excluded mask...")
            excluded = get_existing_excluded(MODELS, PAPER)
            excluded_count = sum(excluded.values())
            excluded_qids = [qid for qid, is_excluded in excluded.items() if is_excluded]
            print(f"  Total excluded: {excluded_count} questions: {excluded_qids}")
    else:
        print("\n[Step 1] Filtering disabled; using all samples")
    
    # Step 2: Compute per-model metrics
    print("\n[Step 2] Computing per-model metrics...")
    metrics_dict = {}
    for model in MODELS:
        print(f"\n  Computing metrics for {model}...")
        metrics = calculate_metrics(model, PAPER, excluded, questions)
        metrics_dict[model] = metrics
        if metrics:
            for key, value in metrics.items():
                if value is not None:
                    print(f"    {key}: {value}")
    
    # Step 3: Update Excel workbook
    print("\n[Step 3] Updating Excel workbook...")
    update_xlsx(PAPER, MODELS, metrics_dict, OUTPUT_XLSX)
    
    print("\n" + "=" * 60)
    print("Aggregation complete.")
    print("=" * 60)


def main():
    """Argparse entry point."""
    global PAPER, MODELS, ROOT_DIR, QUESTIONS_DIR, OUTPUT_XLSX
    global ENABLE_FILTERING, EXCLUSION_PROBABILITY, REFRESH_EXCLUDED
    global ENABLE_SUBSET_MODE, SUBSET_FIELD, PAPER_LIST
    global HF_SPLIT, HF_SUBJECT, HF_LIMIT

    parser = argparse.ArgumentParser(
        description="Aggregate per-question prediction metrics into an Excel summary."
    )
    # Common options
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Models to aggregate. Defaults to constants.solvers.")
    parser.add_argument("--pred-dir", default=DEFAULT_PRED_DIR,
                        help="Predictions root directory (one subdir per model).")
    parser.add_argument("--questions-dir", default=DEFAULT_QUESTIONS_DIR,
                        help="Directory containing parsed paper JSON files "
                             "(used when --split is not provided).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_XLSX,
                        help="Output Excel file path.")

    # HuggingFace dataset source (overrides --questions-dir lookup).
    parser.add_argument("--split",
                        help="HuggingFace split name (e.g. en_2603). "
                             "When provided, questions come from the HF dataset.")
    parser.add_argument("--subject",
                        help="Subject filter for --split.")
    parser.add_argument("--limit", type=int,
                        help="Take only the first N questions after filtering.")

    # Normal mode
    parser.add_argument("--paper", default=None,
                        help="Run id used as the prediction filename stem and Excel sheet name. "
                             "With --split, defaults to <split>__<subject>.")
    parser.add_argument("--enable-filtering", action="store_true",
                        help="(normal mode) drop questions where every model scored 0.")
    parser.add_argument("--exclusion-probability", type=float, default=1.0,
                        help="(normal mode) probability of excluding an all-zero question.")
    parser.add_argument("--refresh-excluded", action="store_true",
                        help="(normal mode) regenerate the excluded mask in prediction JSONs.")

    # Subset mode
    parser.add_argument("--subset", action="store_true",
                        help="Enable subset mode: aggregate metrics over questions tagged with a particular subset.")
    parser.add_argument("--subset-field", default="",
                        help="(subset mode) name of the subset, e.g. 'complex_layout'. "
                             "Matches questions where q['subset'] (a list[str]) contains this name, "
                             "or where q[<name>] is a truthy boolean.")
    parser.add_argument("--papers", nargs="+", default=[],
                        help="(subset mode) list of papers (without .json) to combine.")

    args = parser.parse_args()

    MODELS[:] = args.models
    ROOT_DIR = args.pred_dir
    QUESTIONS_DIR = args.questions_dir
    OUTPUT_XLSX = args.output
    ENABLE_FILTERING = args.enable_filtering
    EXCLUSION_PROBABILITY = args.exclusion_probability
    REFRESH_EXCLUDED = args.refresh_excluded
    ENABLE_SUBSET_MODE = args.subset
    SUBSET_FIELD = args.subset_field
    PAPER_LIST[:] = args.papers
    HF_SPLIT = args.split or ""
    HF_SUBJECT = args.subject or ""
    HF_LIMIT = args.limit

    # Resolve PAPER (used as run-id / sheet name) when not given explicitly.
    if args.paper:
        PAPER = args.paper
    elif HF_SPLIT:
        PAPER = split_subject_to_run_id(HF_SPLIT, HF_SUBJECT or None)
    else:
        PAPER = ""

    if ENABLE_SUBSET_MODE:
        if not SUBSET_FIELD or not PAPER_LIST:
            parser.error("--subset requires --subset-field and --papers")
        main_subset_mode()
    else:
        if not PAPER:
            parser.error(
                "normal mode requires --paper, or --split (which auto-derives --paper as <split>__<subject>), "
                "or use --subset."
            )
        main_normal_mode()


if __name__ == "__main__":
    main()