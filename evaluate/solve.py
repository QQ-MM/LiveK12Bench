"""Run a model over all questions in a paper and save predictions.

Supports three solving modes:
    e2e   - end-to-end: text + raw images (per-question) go to the model.
    photo - photo-based: per-question screenshot is fed to the model.
    exam  - whole-paper images + question-number instructions per call.

The mode is chosen via `evaluate.constants.mode_cfg` (or the `mode` argument
when calling `predict()` directly).

Output JSON layout:
    {
        "0": {"prediction": "...", "metadata": {"usage": {...}, "total_time": "...", "tokens": N}},
        "1": {"prediction": "...", "metadata": {...}},
        ...
    }
"""

import os
import re
import json
import math
import time
import base64
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from util.llm import call_llm
from util.dataset_loader import (
    load_questions as _loader_load,
    split_subject_to_run_id,
)
from prompts import system_prompt
from constants import paper_dir, prediction_dir, mode_cfg


def concatenate_images(image_paths, max_images=5):
    """Vertically concatenate images so that no more than `max_images` are produced.

    Used when a model server limits the number of images per request.
    """
    if len(image_paths) <= max_images:
        result = []
        for img_path in image_paths:
            with open(img_path, "rb") as f:
                result.append(base64.b64encode(f.read()).decode("utf8"))
        return result

    images_per_group = math.ceil(len(image_paths) / max_images)
    result_base64 = []

    for i in range(0, len(image_paths), images_per_group):
        group = image_paths[i:i + images_per_group]

        if len(group) == 1:
            with open(group[0], "rb") as f:
                result_base64.append(base64.b64encode(f.read()).decode("utf8"))
            continue

        imgs = [Image.open(p).convert("RGB") for p in group]
        max_width = max(img.width for img in imgs)
        total_height = sum(img.height for img in imgs)
        combined = Image.new("RGB", (max_width, total_height), "white")
        y_offset = 0
        for img in imgs:
            x_offset = (max_width - img.width) // 2
            combined.paste(img, (x_offset, y_offset))
            y_offset += img.height
            img.close()
        buffer = BytesIO()
        combined.save(buffer, format="JPEG", quality=85)
        result_base64.append(base64.b64encode(buffer.getvalue()).decode("utf8"))
        combined.close()

    return result_base64


def _img_to_data_url(img_path):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf8")
    return f"data:image/jpeg;base64,{b64}"


def _build_messages(ind, mode, questions, types, images):
    """Construct the chat-style messages payload for a single question."""
    if mode == "photo":
        prompt = [
            {"type": "text", "text": system_prompt + "\n\nThe question is shown in the image."},
            {"type": "image_url", "image_url": {"url": _img_to_data_url(questions[ind])}},
        ]

    elif mode == "exam":
        prompt = [
            {"type": "text", "text": system_prompt + f"\n\nPlease answer question #{ind + 1} from the paper."}
        ]
        max_images = 5
        if len(images) > max_images:
            for img_b64 in concatenate_images(images, max_images=max_images):
                prompt.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })
        else:
            for img in images:
                prompt.append({"type": "image_url", "image_url": {"url": _img_to_data_url(img)}})

    elif mode == "e2e":
        q, t = questions[ind], types[ind]
        prompt = [
            {"type": "text", "text": system_prompt + f"\n\nQuestion type: \"{t}\"\n\nQuestion: \"{q}\""}
        ]
        per_q_images = images[ind]
        if per_q_images:
            if not isinstance(per_q_images, list):
                per_q_images = [per_q_images]
            for img in per_q_images:
                if not img.startswith("/"):
                    img = os.path.join(paper_dir, img)
                prompt.append({"type": "image_url", "image_url": {"url": _img_to_data_url(img)}})

    else:
        raise ValueError(f"Unknown solving mode: {mode!r}")

    return [{"role": "user", "content": prompt}]


def process_single_question(ind, mode, questions, types=None, images=None, model=None, max_tokens=32768):
    """Solve a single question with retries. Returns (ind, response_text, metadata)."""
    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"Retrying question {ind}, attempt {attempt}/{max_retries}")
            else:
                print(f"Solving question: {ind}")

            messages = _build_messages(ind, mode, questions, types, images)
            response, metadata = call_llm(messages=messages, model=model, max_tokens=max_tokens)

            if response and response.strip():
                if attempt > 0:
                    print(f"Question {ind} succeeded on attempt {attempt + 1}")
                return ind, response, metadata
            elif attempt < max_retries:
                print(f"Question {ind} returned empty response, retrying ({attempt + 1}/{max_retries + 1})")
                time.sleep(1)
                continue
            else:
                print(f"Question {ind} failed after {max_retries + 1} attempts; saving empty result")
                return ind, "", metadata

        except Exception as e:
            print(f"Error processing question {ind} on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                time.sleep(1)
                continue
            else:
                return ind, "", {"usage": {}, "total_time": 0, "tokens": 0}


def predict(
    *,
    run_id,
    model,
    max_tokens=32768,
    max_workers=5,
    mode="e2e",
    split=None,
    subject=None,
    json_path=None,
    limit=None,
    ids=None,
):
    """Solve all questions and save predictions to ``predictions/<model>/<run_id>.json``.

    Question source is one of:
        - HuggingFace dataset (``split`` and optionally ``subject``)
        - Local JSON produced by ``analyze/analyze.py`` (``json_path``)
        - Photo / exam mode legacy local-paper layout (``run_id`` resolves to
          ``<paper_dir>/<run_id>``).

    Args:
        run_id:      Stable identifier used as the prediction filename stem,
                     e.g. ``"en_2603__math"`` or a paper name.
        model:       Model identifier passed to call_llm.
        max_tokens:  Output token cap.
        max_workers: Thread pool size for parallel solving.
        mode:        One of "e2e" / "photo" / "exam".
        split:       HuggingFace split (e.g. ``"en_2603"``).
        subject:     Optional subject filter for HF split.
        json_path:   Local JSON file produced by analyze.py.
        limit / ids: Optional row filters (forwarded to the dataset loader).
    """
    questions, types, images = _load_questions(
        run_id=run_id,
        mode=mode,
        split=split,
        subject=subject,
        json_path=json_path,
        limit=limit,
        ids=ids,
    )

    os.makedirs(os.path.join(prediction_dir, model), exist_ok=True)
    prediction_path = os.path.join(prediction_dir, model, f"{run_id}.json")

    predictions = {}
    inds = list(range(len(questions)))
    if os.path.exists(prediction_path):
        with open(prediction_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        for ind in list(predictions.keys()):
            if predictions[ind].get("prediction"):
                inds.remove(int(ind))

    file_lock = threading.Lock()
    print(f"Starting parallel processing with {max_workers} workers for {len(inds)} questions")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ind = {
            executor.submit(
                process_single_question,
                ind, mode, questions, types, images, model, max_tokens
            ): ind for ind in inds
        }
        for future in as_completed(future_to_ind):
            try:
                ind, response, metadata = future.result()
                with file_lock:
                    predictions[str(ind)] = {"prediction": response, "metadata": metadata}
                    with open(prediction_path, "w", encoding="utf-8") as f:
                        json.dump(predictions, f, ensure_ascii=False, indent=4)
                    if response:
                        remaining = len([fu for fu in future_to_ind if not fu.done()])
                        print(f"Question {ind} done. Remaining: {remaining}")
                    else:
                        print(f"Question {ind} completed but no response")
            except Exception as e:
                ind = future_to_ind[future]
                print(f"Question {ind} raised: {e}")
                with file_lock:
                    predictions[str(ind)] = {"prediction": "", "metadata": {"usage": {}, "total_time": 0, "tokens": 0}}
                    with open(prediction_path, "w", encoding="utf-8") as f:
                        json.dump(predictions, f, ensure_ascii=False, indent=4)

    print(f"All questions processed. Results saved to {prediction_path}")


def _load_questions(
    *,
    run_id,
    mode,
    split=None,
    subject=None,
    json_path=None,
    limit=None,
    ids=None,
):
    """Resolve the question source for a given run.

    For ``e2e`` mode we go through ``util.dataset_loader`` so HF datasets
    and analyze.py JSON share one normalised schema. ``photo`` and
    ``exam`` modes still walk a local image directory keyed by ``run_id``
    (these modes are designed for raw OCR screenshots that do not live in
    the HF dataset; place the screenshots under ``<paper_dir>/<run_id>/``).
    """
    if mode == "photo":
        # photo mode: per-question screenshot in <paper_dir>/<run_id>/*.png
        files = os.listdir(os.path.join(paper_dir, run_id))
        files = [q for q in files if ".png" in q and "page_" not in q]
        files = sorted(files, key=lambda x: int(re.search(r"\d+", x).group()))
        questions = [os.path.join(paper_dir, run_id, q) for q in files]
        return questions, None, None

    if mode == "exam":
        # exam mode: full-paper screenshots + structured questions.
        records = _resolve_records(
            run_id=run_id,
            split=split,
            subject=subject,
            json_path=json_path,
            limit=limit,
            ids=ids,
        )
        questions = [r["question"] for r in records]
        # Whole-paper page screenshots live next to <run_id>/page_*.png on disk.
        files = os.listdir(os.path.join(paper_dir, run_id))
        files = [q for q in files if ".png" in q and "page_" in q]
        files = sorted(files, key=lambda x: int(x.split("page_")[1].split(".png")[0]))
        images = [os.path.join(paper_dir, run_id, q) for q in files]
        return questions, None, images

    if mode == "e2e":
        records = _resolve_records(
            run_id=run_id,
            split=split,
            subject=subject,
            json_path=json_path,
            limit=limit,
            ids=ids,
        )
        questions = [r["question"] for r in records]
        types = [r["question_type"] for r in records]
        images = [r["images"] for r in records]
        return questions, types, images

    raise ValueError(f"Unknown solving mode: {mode!r}")


def _resolve_records(*, run_id, split, subject, json_path, limit, ids):
    """Pick one of the two question sources and return normalised records.

    Resolution order:
        1. Explicit ``split`` (HuggingFace dataset)
        2. Explicit ``json_path`` (local file from analyze.py)
    """
    if split:
        return _loader_load(
            split=split, subject=subject, limit=limit, ids=ids,
        )
    if json_path:
        return _loader_load(
            json_path=json_path, subject=subject or "", limit=limit,
        )
    raise FileNotFoundError(
        f"No question source provided. Pass --split or --json."
    )


if __name__ == "__main__":
    # Example invocations:
    #     # HuggingFace dataset slice
    #     python solve.py --split en_2603 --subject math --model gpt-5-mini --limit 10
    #     # Local JSON produced by analyze/analyze.py
    #     python solve.py --json analyze/analyzed_json/sample/foo.json --model gpt-5
    # Set OPENAI_API_KEY (and friends) before running.
    import argparse

    parser = argparse.ArgumentParser(description="Run a model over a paper / dataset slice.")
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

    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--mode", default=mode_cfg, choices=["e2e", "photo", "exam"])
    args = parser.parse_args()

    if args.run_id:
        run_id = args.run_id
    elif args.split:
        run_id = split_subject_to_run_id(args.split, args.subject)
    else:
        run_id = os.path.splitext(os.path.basename(args.json_path))[0]

    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None

    predict(
        run_id=run_id,
        model=args.model,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        mode=args.mode,
        split=args.split,
        subject=args.subject,
        json_path=args.json_path,
        limit=args.limit,
        ids=ids,
    )
