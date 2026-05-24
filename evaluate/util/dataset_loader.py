"""Dataset loader for LiveK12Bench.

Provides a single entry point :func:`load_questions` that returns a list
of normalised per-question dicts regardless of source (HuggingFace
``Shawn-wxh/livek12bench`` or a local JSON file produced by
``analyze/analyze.py``).

Normalised question schema (every field is guaranteed to exist):

    {
        "id":               str,            # stable id (HF: dataset id; local: f"{paper}_{idx:04d}")
        "set":              str,            # "2603" / "2605" / paper name
        "subject":          str,            # "math" / "physics" / "chemistry" / "biology" / ""
        "question_type":    str,            # one of QUESTION_TYPES values (canonical enum)
        "question_type_raw": str,           # original string from the dataset (Chinese or English)
        "point_value":      int,            # default 0 if missing
        "question":         str,
        "answer":           list[str],      # always a list (single-answer wrapped in a 1-elt list)
        "solution":         str,
        "knowledge_points": str,
        "images":           list[str],      # local file-system paths (PIL objects are cached to disk)
    }

Local-JSON inputs may use either the new English field names or the
legacy Chinese field names emitted by ``analyze/configs/chinese_k12_exam.py``;
both are accepted transparently.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Canonical question_type enum
# ---------------------------------------------------------------------------
# We map every observed surface form (Chinese in zh_* splits, English in
# en_* splits, plus the strings emitted by analyze.py) to a canonical
# enum value so downstream graders never have to branch on language.

MULTIPLE_CHOICE = "multiple_choice"
FILL_IN_BLANK = "fill_in_blank"
OPEN_ENDED = "open_ended"
PROVING = "proving"
UNKNOWN = "unknown"

QUESTION_TYPES = {MULTIPLE_CHOICE, FILL_IN_BLANK, OPEN_ENDED, PROVING, UNKNOWN}

_TYPE_ALIASES: Dict[str, str] = {
    # Chinese (zh splits + analyze.py output)
    "选择题": MULTIPLE_CHOICE,
    "填空题": FILL_IN_BLANK,
    "解答题": OPEN_ENDED,
    "证明题": PROVING,
    "未知": UNKNOWN,
    # English (en splits)
    "multiple choice": MULTIPLE_CHOICE,
    "multiple-choice": MULTIPLE_CHOICE,
    "mcq": MULTIPLE_CHOICE,
    "fill in the blank": FILL_IN_BLANK,
    "fill-in-the-blank": FILL_IN_BLANK,
    "fill in blank": FILL_IN_BLANK,
    "open-ended": OPEN_ENDED,
    "open ended": OPEN_ENDED,
    "free response": OPEN_ENDED,
    "proof": PROVING,
    "proving": PROVING,
}


def normalise_question_type(raw: Optional[str]) -> str:
    """Map any surface form of question_type to the canonical enum.

    Falls back to ``UNKNOWN`` for unseen strings (and logs the offender
    once via stderr to ease dataset onboarding).
    """
    if not raw:
        return UNKNOWN
    key = str(raw).strip().lower()
    if key in _TYPE_ALIASES:
        return _TYPE_ALIASES[key]
    # Also try the original Chinese form (case-insensitive doesn't help
    # for CJK but the dict key is the canonical form).
    if str(raw).strip() in _TYPE_ALIASES:
        return _TYPE_ALIASES[str(raw).strip()]
    return UNKNOWN


# ---------------------------------------------------------------------------
# Image cache: PIL → local path
# ---------------------------------------------------------------------------
# HuggingFace returns images as PIL objects. solve.py expects local
# file-system paths it can ``open(..., "rb")``. We cache decoded images
# under ``<cache_dir>/<id>_<idx>.<ext>`` so subsequent runs (or a
# parallel solver invocation) can reuse them without re-decoding.

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/livek12bench/images")


def _ensure_cache_dir(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)


def _save_pil_to_cache(pil_image, cache_dir: str, q_id: str, idx: int) -> str:
    """Save a PIL.Image to cache_dir if not already present, return path."""
    fmt = (getattr(pil_image, "format", None) or "PNG").upper()
    ext = "jpg" if fmt in ("JPEG", "JPG") else fmt.lower()
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", q_id)
    path = os.path.join(cache_dir, f"{safe_id}_{idx}.{ext}")
    if not os.path.exists(path):
        # Convert mode RGBA→RGB for JPEG to avoid decoder errors downstream.
        save_image = pil_image
        if ext == "jpg" and save_image.mode != "RGB":
            save_image = save_image.convert("RGB")
        save_image.save(path)
    return path


def _materialise_images(images: Any, q_id: str, cache_dir: str) -> List[str]:
    """Coerce the ``images`` field into a list of local file-system paths.

    Accepts: ``None``, ``[]``, list of PIL.Image, list of str paths, or
    list of dicts with a ``path`` / ``bytes`` key (HF datasets sometimes
    expose this when ``decode=False``).
    """
    if not images:
        return []
    out: List[str] = []
    _ensure_cache_dir(cache_dir)
    for idx, item in enumerate(images):
        if item is None:
            continue
        # Already a path string
        if isinstance(item, str):
            out.append(item)
            continue
        # HF dict-form
        if isinstance(item, dict):
            if item.get("path") and os.path.exists(item["path"]):
                out.append(item["path"])
                continue
            if item.get("bytes"):
                # Hash-name and write to cache.
                h = hashlib.md5(item["bytes"]).hexdigest()[:12]
                path = os.path.join(cache_dir, f"{q_id}_{idx}_{h}.png")
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        f.write(item["bytes"])
                out.append(path)
                continue
        # PIL.Image (don't import PIL eagerly; duck-type)
        if hasattr(item, "save") and hasattr(item, "format"):
            out.append(_save_pil_to_cache(item, cache_dir, q_id, idx))
            continue
        # Fallback: stringify
        out.append(str(item))
    return out


# ---------------------------------------------------------------------------
# Field normalisation (Chinese ↔ English)
# ---------------------------------------------------------------------------
# analyze/analyze.py emits the legacy Chinese keys; the HF dataset uses
# English keys. We accept both and emit a single English schema.

_FIELD_ALIASES: Dict[str, str] = {
    # legacy Chinese → English
    "题型": "question_type",
    "分值": "point_value",
    "题目": "question",
    "答案": "answer",
    "解答": "solution",
    "图像": "images",
    "考点": "knowledge_points",
}


def _normalise_record(
    rec: Dict[str, Any],
    *,
    fallback_id: str,
    fallback_set: str,
    fallback_subject: str,
    cache_dir: str,
) -> Dict[str, Any]:
    """Convert a raw dataset record into the canonical schema."""
    # First make a shallow copy with English keys.
    src: Dict[str, Any] = {}
    for k, v in rec.items():
        if k in _FIELD_ALIASES:
            src[_FIELD_ALIASES[k]] = v
        else:
            src[k] = v

    # answer: coerce str → [str]
    answer = src.get("answer")
    if answer is None:
        answer = []
    elif isinstance(answer, str):
        answer = [answer]
    elif isinstance(answer, (list, tuple)):
        answer = [str(a) for a in answer]
    else:
        answer = [str(answer)]

    # point_value: coerce to int
    try:
        point_value = int(src.get("point_value") or 0)
    except (TypeError, ValueError):
        point_value = 0

    qtype_raw = src.get("question_type") or ""
    qtype = normalise_question_type(qtype_raw)

    q_id = str(src.get("id") or fallback_id)
    images = _materialise_images(src.get("images"), q_id=q_id, cache_dir=cache_dir)

    rec: Dict[str, Any] = {
        "id": q_id,
        "set": str(src.get("set") or fallback_set),
        "subject": str(src.get("subject") or fallback_subject),
        "question_type": qtype,
        "question_type_raw": str(qtype_raw),
        "point_value": point_value,
        "question": str(src.get("question") or ""),
        "answer": answer,
        "solution": str(src.get("solution") or ""),
        "knowledge_points": str(src.get("knowledge_points") or ""),
        "images": images,
    }

    # Pass through any extra fields users put on their questions (e.g. custom
    # boolean tags consumed by `metric.py --subset`). We never overwrite the
    # canonical fields above, but anything else is forwarded verbatim.
    for k, v in src.items():
        if k not in rec:
            rec[k] = v

    return rec


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------
HF_DATASET_NAME = "Shawn-wxh/livek12bench"


def load_split(
    split: str,
    *,
    subject: Optional[str] = None,
    limit: Optional[int] = None,
    ids: Optional[Sequence[str]] = None,
    dataset_name: str = HF_DATASET_NAME,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> List[Dict[str, Any]]:
    """Load one HuggingFace split, optionally filtered by subject / ids / limit.

    Args:
        split: e.g. ``"en_2603"``, ``"zh_2605"``.
        subject: filter to a single subject (``"math"`` / ``"physics"`` / ...).
        limit: take only the first N rows after filtering (handy for smoke tests).
        ids: restrict to specific question ids (set semantics).
        dataset_name: override HF dataset name (defaults to the public release).
        cache_dir: where to materialise PIL images.

    Returns:
        A list of normalised records.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "datasets is not installed; `pip install datasets` "
            "or use load_local_json() for offline JSON files."
        ) from e

    ds = load_dataset(dataset_name, split=split)

    id_set = set(ids) if ids else None
    out: List[Dict[str, Any]] = []
    # Try to pull a 'set' value out of the split name (e.g. en_2603 → 2603).
    inferred_set = ""
    m = re.match(r"^(?:zh|en)_(.+)$", split)
    if m:
        inferred_set = m.group(1)

    for idx, row in enumerate(ds):
        if subject and str(row.get("subject", "")).lower() != subject.lower():
            continue
        if id_set is not None and str(row.get("id", "")) not in id_set:
            continue
        rec = _normalise_record(
            row,
            fallback_id=f"{split}_{idx:04d}",
            fallback_set=inferred_set,
            fallback_subject=str(row.get("subject", "")),
            cache_dir=cache_dir,
        )
        out.append(rec)
        if limit is not None and len(out) >= limit:
            break
    return out


def load_local_json(
    json_path: str,
    *,
    set_name: str = "",
    subject: str = "",
    limit: Optional[int] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> List[Dict[str, Any]]:
    """Load a single JSON file produced by analyze/analyze.py.

    Accepts both legacy Chinese-key and new English-key schemas.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{json_path}: expected a top-level JSON array.")

    paper_stem = os.path.splitext(os.path.basename(json_path))[0]
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        rec = _normalise_record(
            row,
            fallback_id=f"{paper_stem}_{idx:04d}",
            fallback_set=set_name or paper_stem,
            fallback_subject=subject,
            cache_dir=cache_dir,
        )
        out.append(rec)
        if limit is not None and len(out) >= limit:
            break
    return out


def load_questions(
    *,
    split: Optional[str] = None,
    subject: Optional[str] = None,
    json_path: Optional[str] = None,
    limit: Optional[int] = None,
    ids: Optional[Sequence[str]] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> List[Dict[str, Any]]:
    """Unified entry point.

    Exactly one of ``split`` (HuggingFace) or ``json_path`` (local file)
    must be provided.
    """
    if (split is None) == (json_path is None):
        raise ValueError("Provide exactly one of `split` or `json_path`.")
    if split is not None:
        return load_split(
            split,
            subject=subject,
            limit=limit,
            ids=ids,
            cache_dir=cache_dir,
        )
    return load_local_json(
        json_path,
        subject=subject or "",
        limit=limit,
        cache_dir=cache_dir,
    )


def split_subject_to_run_id(split: str, subject: Optional[str]) -> str:
    """Compose the canonical ``predictions/<model>/<run_id>.json`` stem."""
    if subject:
        return f"{split}__{subject}"
    return split
