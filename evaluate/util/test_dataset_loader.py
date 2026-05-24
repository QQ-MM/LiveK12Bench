"""Sanity tests for evaluate.util.dataset_loader.

These tests exercise the offline / pure-logic paths (no HuggingFace
network calls) so they can run in any environment with just stdlib +
the package itself. Run with:

    python -m unittest evaluate.util.test_dataset_loader

or simply:

    python -m unittest discover -s evaluate -p 'test_*.py'
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

# Allow running both as `python -m unittest evaluate.util.test_dataset_loader`
# and as a script (`python evaluate/util/test_dataset_loader.py`).
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluate.util import dataset_loader as dl


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SMOKE_JSON = os.path.join(REPO_ROOT, "data", "chinese_k12", "bio", "smoke_test_paper.json")


class TestNormaliseQuestionType(unittest.TestCase):
    """Type-alias mapping should cover both Chinese and English surface forms."""

    def test_chinese_aliases(self):
        self.assertEqual(dl.normalise_question_type("选择题"), dl.MULTIPLE_CHOICE)
        self.assertEqual(dl.normalise_question_type("填空题"), dl.FILL_IN_BLANK)
        self.assertEqual(dl.normalise_question_type("解答题"), dl.OPEN_ENDED)
        self.assertEqual(dl.normalise_question_type("证明题"), dl.PROVING)

    def test_english_aliases(self):
        self.assertEqual(dl.normalise_question_type("multiple choice"), dl.MULTIPLE_CHOICE)
        self.assertEqual(dl.normalise_question_type("Multiple-Choice"), dl.MULTIPLE_CHOICE)
        self.assertEqual(dl.normalise_question_type("MCQ"), dl.MULTIPLE_CHOICE)
        self.assertEqual(dl.normalise_question_type("fill in the blank"), dl.FILL_IN_BLANK)
        self.assertEqual(dl.normalise_question_type("open-ended"), dl.OPEN_ENDED)
        self.assertEqual(dl.normalise_question_type("Free Response"), dl.OPEN_ENDED)
        self.assertEqual(dl.normalise_question_type("proof"), dl.PROVING)

    def test_falls_back_to_unknown(self):
        self.assertEqual(dl.normalise_question_type(None), dl.UNKNOWN)
        self.assertEqual(dl.normalise_question_type(""), dl.UNKNOWN)
        self.assertEqual(dl.normalise_question_type("brand-new-type"), dl.UNKNOWN)


class TestNormaliseRecord(unittest.TestCase):
    """Both legacy Chinese keys and modern English keys must produce the same canonical schema."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp(prefix="livek12bench_test_")

    def _normalise(self, raw):
        return dl._normalise_record(
            raw,
            fallback_id="fallback_0001",
            fallback_set="testset",
            fallback_subject="math",
            cache_dir=self.cache_dir,
        )

    def test_legacy_chinese_record(self):
        raw = {
            "题型": "选择题",
            "分值": 5,
            "题目": "1+1=?",
            "答案": ["2"],
            "解答": "Trivially 2.",
            "图像": None,
        }
        rec = self._normalise(raw)
        self.assertEqual(rec["question_type"], dl.MULTIPLE_CHOICE)
        self.assertEqual(rec["question_type_raw"], "选择题")
        self.assertEqual(rec["point_value"], 5)
        self.assertEqual(rec["question"], "1+1=?")
        self.assertEqual(rec["answer"], ["2"])
        self.assertEqual(rec["solution"], "Trivially 2.")
        self.assertEqual(rec["images"], [])
        # Fallback metadata used when source omits the fields
        self.assertEqual(rec["id"], "fallback_0001")
        self.assertEqual(rec["set"], "testset")
        self.assertEqual(rec["subject"], "math")

    def test_english_record(self):
        raw = {
            "id": "en_2603_q42",
            "set": "2603",
            "subject": "physics",
            "question_type": "Open-Ended",
            "point_value": "12",          # string is OK -> coerced to int
            "question": "Derive Newton's second law.",
            "answer": "F = ma",            # bare string -> wrapped to list
            "solution": "By definition.",
            "knowledge_points": "mechanics",
        }
        rec = self._normalise(raw)
        self.assertEqual(rec["id"], "en_2603_q42")
        self.assertEqual(rec["set"], "2603")
        self.assertEqual(rec["subject"], "physics")
        self.assertEqual(rec["question_type"], dl.OPEN_ENDED)
        self.assertEqual(rec["question_type_raw"], "Open-Ended")
        self.assertEqual(rec["point_value"], 12)
        self.assertEqual(rec["answer"], ["F = ma"])
        self.assertEqual(rec["knowledge_points"], "mechanics")

    def test_missing_or_invalid_point_value_defaults_to_zero(self):
        rec = self._normalise({"题型": "选择题", "题目": "x"})
        self.assertEqual(rec["point_value"], 0)
        rec = self._normalise({"题型": "选择题", "分值": "not-a-number", "题目": "x"})
        self.assertEqual(rec["point_value"], 0)

    def test_unknown_question_type_preserves_raw_string(self):
        rec = self._normalise({"题型": "趣味题", "题目": "x"})
        self.assertEqual(rec["question_type"], dl.UNKNOWN)
        self.assertEqual(rec["question_type_raw"], "趣味题")

    def test_string_image_paths_pass_through(self):
        rec = self._normalise({
            "题型": "选择题", "题目": "x",
            "图像": ["/tmp/foo.png", "/tmp/bar.jpg"],
        })
        self.assertEqual(rec["images"], ["/tmp/foo.png", "/tmp/bar.jpg"])

    def test_extra_fields_passthrough(self):
        """Custom fields (e.g. subset boolean tags) must be forwarded verbatim."""
        rec = self._normalise({
            "题型": "选择题", "题目": "x",
            "answer_hack": True,
            "complex_layout": False,
            "long_reasoning": True,
            "my_custom_tag": "foo",
        })
        self.assertTrue(rec["answer_hack"])
        self.assertFalse(rec["complex_layout"])
        self.assertTrue(rec["long_reasoning"])
        self.assertEqual(rec["my_custom_tag"], "foo")

    def test_subset_list_field_passthrough(self):
        """The canonical `subset` field (list[str]) must be forwarded as-is."""
        rec = self._normalise({
            "题型": "选择题", "题目": "x",
            "subset": ["complex_layout", "long_reason"],
        })
        self.assertEqual(rec["subset"], ["complex_layout", "long_reason"])
        self.assertIsInstance(rec["subset"], list)

    def test_extras_never_clobber_canonical_fields(self):
        """A clash with a canonical key (e.g. answer) must not break normalisation."""
        rec = self._normalise({
            "题型": "选择题", "题目": "x",
            "answer": "A",     # legacy mapping handles this
        })
        # answer should still be the wrapped list, not silently replaced.
        self.assertEqual(rec["answer"], ["A"])


class TestLoadLocalJson(unittest.TestCase):
    """End-to-end smoke test against the committed fixture."""

    def test_smoke_fixture_loads(self):
        self.assertTrue(
            os.path.exists(SMOKE_JSON),
            f"smoke fixture missing: {SMOKE_JSON}"
        )
        records = dl.load_local_json(
            SMOKE_JSON, set_name="smoke", subject="math", limit=2,
        )
        self.assertEqual(len(records), 2)
        for rec in records:
            # Canonical schema fields must all be present
            for key in (
                "id", "set", "subject", "question_type", "question_type_raw",
                "point_value", "question", "answer", "solution",
                "knowledge_points", "images",
            ):
                self.assertIn(key, rec, f"missing field: {key}")
            self.assertIn(rec["question_type"], dl.QUESTION_TYPES)
            self.assertIsInstance(rec["answer"], list)
            self.assertIsInstance(rec["images"], list)
            self.assertEqual(rec["set"], "smoke")
            self.assertEqual(rec["subject"], "math")

    def test_load_local_json_respects_limit(self):
        all_records = dl.load_local_json(SMOKE_JSON)
        limited = dl.load_local_json(SMOKE_JSON, limit=1)
        self.assertEqual(len(limited), 1)
        self.assertGreaterEqual(len(all_records), len(limited))


class TestLoadQuestionsDispatch(unittest.TestCase):
    """The unified entry point routes split/json correctly and rejects bad calls."""

    def test_requires_exactly_one_source(self):
        with self.assertRaises(ValueError):
            dl.load_questions()  # neither
        with self.assertRaises(ValueError):
            dl.load_questions(split="en_2603", json_path="/tmp/x.json")  # both

    def test_json_path_route(self):
        records = dl.load_questions(json_path=SMOKE_JSON, limit=1)
        self.assertEqual(len(records), 1)
        self.assertIn("question_type", records[0])


class TestSplitSubjectToRunId(unittest.TestCase):
    def test_with_subject(self):
        self.assertEqual(
            dl.split_subject_to_run_id("en_2603", "math"),
            "en_2603__math",
        )

    def test_without_subject(self):
        self.assertEqual(
            dl.split_subject_to_run_id("en_2603", None),
            "en_2603",
        )
        self.assertEqual(
            dl.split_subject_to_run_id("en_2603", ""),
            "en_2603",
        )


if __name__ == "__main__":
    unittest.main()
