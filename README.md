# LiveK12Bench

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-Shawn--wxh%2Flivek12bench-yellow)](https://huggingface.co/datasets/Shawn-wxh/livek12bench)
[![arXiv](https://img.shields.io/badge/arXiv-2605.26781-b31b1b.svg)](https://arxiv.org/abs/2605.26781)

> **Have Large Multimodal Models Truly Conquered High School-level Examinations?**

Official code release for the paper *LiveK12Bench: Have Large Multimodal
Models Truly Conquered High School-level Examinations?* ([Arxiv](https://arxiv.org/abs/2605.26781), [HuggingFace
dataset](https://huggingface.co/datasets/Shawn-wxh/livek12bench)).

## TL;DR

Large multimodal models (LMMs) routinely score near-perfect on benchmarks
like MATH and AIME, yet they have never been put through a real student's
day: a full exam paper, end-to-end, under timing and process-rigor
constraints, with figures and questions laid out together on the page.
**LiveK12Bench** asks whether they can.

The benchmark contains **2,000+ verified questions** spanning Mathematics,
Physics, Chemistry, and Biology, sourced from the latest (2026) authentic
Chinese high-school exam papers, distributed in both Chinese and English,
and designed to **grow over time** so it can resist data contamination.
Its evaluation goes beyond final-answer accuracy and grades models the
way a teacher grades a student:

- **Outcome** — final-answer accuracy (Pass@1).
- **Reasoning process** — three error categories (Condition Interpretation,
  Logical Assumption, Deductive Reasoning) penalised against the
  reference solution.
- **Reasoning efficiency** — accuracy reweighted by response length and
  accuracy under a hard token budget.
- **Exam performance** — a holistic, weighted *Mock Exam* score that
  mirrors human grading.

## What's in this repository

- **`evaluate/`** — solver and grader for the benchmark. Runs over
  [`Shawn-wxh/livek12bench`](https://huggingface.co/datasets/Shawn-wxh/livek12bench)
  (or any local JSON in the same schema), and routes every model call
  through [LiteLLM](https://github.com/BerriAI/litellm) so you can plug
  in any provider behind one configuration knob.
- **`analyze/`** — OCR-markdown → structured-JSON parsing pipeline. Use
  this if you want to add your own exam papers to the benchmark.
- **`data/`** — exam-paper inputs. The default `paper_dir` is
  `data/chinese_k12/bio/`. A two-question smoke-test fixture
  (`data/chinese_k12/bio/smoke_test_paper.json` and
  `data/ocr_input/smoke_test_paper.md`) is committed so you can verify
  your environment without HuggingFace access; everything else is
  user-supplied and gitignored.
- **`predictions/`** — solver outputs. Each run lands in
  `predictions/<model>/<run-id>.json` (gitignored).
- **`metrics/`** — aggregated Excel workbooks produced by `metric.py`
  (gitignored).

---

## Installation

```bash
bash setup.sh
```

…or manually:

```bash
pip install -U -r requirements.txt
```

Python 3.10+ recommended.

---

## Dataset

The benchmark lives on the HuggingFace Hub:

```python
from datasets import load_dataset

ds = load_dataset("Shawn-wxh/livek12bench")
# DatasetDict with four splits:
#   zh_2603, zh_2605    — original Chinese papers
#   en_2603, en_2605    — English translations of the same papers
```

Each row has the following schema:

| field              | type        | meaning                                                                |
|--------------------|-------------|------------------------------------------------------------------------|
| `id`               | `str`       | stable id, e.g. `"en_2603_math_0001"`                                  |
| `set`              | `str`       | release set (e.g. `"2603"`, `"2605"`)                                  |
| `subject`          | `str`       | one of `math`, `physics`, `chemistry`, `biology`                       |
| `question_type`    | `str`       | language-dependent: `选择题` / `Multiple Choice`, etc.                  |
| `point_value`      | `int`       | exam-board point value                                                 |
| `question`         | `str`       | question text with LaTeX                                               |
| `answer`           | `list[str]` | reference answer(s); multi-select like `["ACD"]`                       |
| `solution`         | `str`       | full reference solution                                                |
| `knowledge_points` | `str`       | knowledge points (semicolon-separated)                                 |
| `images`           | `list[Image]` | per-question reference images (PIL objects via HuggingFace `datasets`) |

The split-naming convention is `{lang}_{set}` where `lang` ∈ {`zh`, `en`}.

The evaluation tooling normalises every record to a canonical English
schema regardless of source. In particular:

- `question_type` is mapped to a canonical enum
  (`multiple_choice`, `fill_in_blank`, `open_ended`, `proving`,
  `unknown`) so graders never branch on the natural-language form.
- `images` (PIL) are cached to `~/.cache/livek12bench/images/` and the
  in-memory record exposes their on-disk paths.

---

## LLM Configuration

LiveK12Bench routes every model call through
[LiteLLM](https://github.com/BerriAI/litellm), so the same code can
drive OpenAI, Anthropic, Gemini, and self-hosted (vLLM / SGLang / TGI)
models behind a single function `evaluate.util.llm.call_llm`.

### Hosted providers (env vars)

Set the API key for whichever provider you use; LiteLLM auto-detects
the provider from the model name.

```bash
# OpenAI / Azure-style endpoints
export OPENAI_API_KEY=sk-...
# Optional: route OpenAI-compatible traffic to a proxy
# export OPENAI_BASE_URL=https://your-proxy.example.com/v1

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
export GEMINI_API_KEY=...
```

The model name you pass to `call_llm(model="...")` follows LiteLLM's
naming convention. The 12 models exercised in the paper map to:

| paper name                       | LiteLLM model id                       |
|----------------------------------|----------------------------------------|
| `gpt-5`                          | `gpt-5`                                |
| `gpt-5-mini`                     | `gpt-5-mini`                           |
| `gpt-4o-2024-11-20`              | `gpt-4o-2024-11-20`                    |
| `claude-opus-4-6`                | `anthropic/claude-opus-4-6`            |
| `claude-sonnet-4-6`              | `anthropic/claude-sonnet-4-6`          |
| `gemini-3-pro`                   | `gemini/gemini-3-pro`                  |
| `gemini-3-flash`                 | `gemini/gemini-3-flash`                |
| `kimi-k2.5`                      | `moonshot/kimi-k2.5`                   |
| `glm-5`                          | `zhipuai/glm-5`                        |
| `qwen3-vl-235b-a22b-thinking`    | self-hosted (see below)                |
| `qwen3-vl-8b`                    | self-hosted (see below)                |
| `qwen3-vl-32b`                   | self-hosted (see below)                |

You can list any model you like — the LiteLLM
[Supported Models](https://docs.litellm.ai/docs/providers) page is the
canonical reference.

### Self-hosted / vLLM endpoints

For models you serve yourself (vLLM, SGLang, TGI, llama.cpp, etc.),
add them to the `vllm_models` dict in `evaluate/constants.py`:

```python
vllm_models = {
    "qwen3-vl-8b":  "http://your-vllm-host:8000/v1",
    "qwen3-vl-32b": "http://your-vllm-host:8001/v1",
    # ...
}
```

`call_llm` looks up the model name in this dict; if it's there, the
endpoint is used and the call is forwarded as OpenAI-compatible.

### Prompt language

Default prompts are in English (`evaluate/prompts/en.py`) and
explicitly handle Chinese-content exams (they instruct the grader to
work in the question's native language). A verbatim Chinese version of
the original prompts is archived in `evaluate/prompts/zh.py`. Switch
between them by editing `PROMPT_LANG` in `evaluate/constants.py`:

```python
PROMPT_LANG = "en"   # or "zh"
```

---

## Project layout

```
.
├── analyze/                        Exam paper parsing pipeline (optional)
│   ├── analyze.py                  OCR markdown → structured per-question JSON
│   └── configs/
│       └── chinese_k12_exam.py     Extraction schema (legacy Chinese keys)
│
├── evaluate/                       Solver + grader framework
│   ├── constants.py                ⚠️ Project-wide config (paths, model lists)
│   ├── solve.py                    Run a solver across one dataset slice
│   ├── evaluate.py                 Grade solutions (parallel verifier voting)
│   ├── evaluate_process.py         Per-step process error classification
│   ├── metric.py                   Aggregate per-model scores into Excel
│   ├── prompts/
│   │   ├── __init__.py             Lang selector (PROMPT_LANG)
│   │   ├── en.py                   Default English prompts
│   │   └── zh.py                   Archived Chinese prompts
│   └── util/
│       ├── llm.py                  LiteLLM-backed call_llm entry point
│       ├── dataset_loader.py       Unified HF + local-JSON loader
│       ├── average_metrics.py      Cross-paper averaging into summary sheet
│       └── ...
│
├── data/                           Exam paper inputs
│   ├── chinese_k12/bio/            Default `paper_dir` (smoke fixture committed)
│   └── ocr_input/                  Raw OCR markdown drop-zone (smoke fixture committed)
│
├── predictions/                    Solver outputs land here (gitignored)
├── metrics/                        Aggregated Excel reports (gitignored)
├── requirements.txt
└── setup.sh
```

---

## Usage

All four entry points (`solve.py`, `evaluate.py`, `evaluate_process.py`,
`metric.py`) accept the same source-selection flags:

| flag           | source                                                                    |
|----------------|---------------------------------------------------------------------------|
| `--split`      | a HuggingFace split (e.g. `en_2603`, `zh_2605`)                           |
| `--subject`    | optional subject filter (`math` / `physics` / `chemistry` / `biology`)    |
| `--json`       | a local JSON file produced by `analyze/analyze.py`                        |
| `--limit`      | take only the first N questions after filtering (handy for smoke tests)  |
| `--ids`        | comma-separated list of question ids to keep                              |
| `--run-id`     | override the prediction filename stem (default: `<split>__<subject>`)     |

`metric.py` additionally accepts a `--paper` shortcut (run-id stem under
`predictions/`) so you can aggregate results that were produced with a
custom `--run-id`.

Predictions land in `predictions/<model>/<run-id>.json` and the
aggregated metrics workbook lives at `metrics/metrics.xlsx`
(configurable via `evaluate.constants.metrics_path`).

### 1. Run a solver

```bash
cd evaluate

# Smoke-test: 10 math questions from the English split
python solve.py \
    --split en_2603 --subject math --limit 10 \
    --model gpt-5-mini

# Full sweep over one subject, parallel
python solve.py \
    --split en_2603 --subject math \
    --model gpt-5 \
    --max-workers 8

# Run a model on the committed smoke-test fixture (no HuggingFace needed)
python solve.py \
    --json ../data/chinese_k12/bio/smoke_test_paper.json \
    --model gpt-5-mini
```

Available solving modes (`--mode`):

| mode    | input                                                                  |
|---------|------------------------------------------------------------------------|
| `e2e`   | question text + reference images go to the model (default)             |
| `photo` | per-question screenshot only (legacy local-paper directory layout)     |
| `exam`  | full-paper screenshots + per-question instructions                     |

### 2. Grade

```bash
# Single model
python evaluate.py \
    --split en_2603 --subject math --limit 10 \
    --model gpt-5-mini

# All solvers configured in constants.solvers, in parallel
python evaluate.py --split en_2603 --subject math
```

Verifiers vote across multiple solver outputs; results are written back
into the prediction JSON under `metrics.*`.

### 3. Process-level evaluation (CIE / LAE / DRE / PES / OES)

```bash
python evaluate_process.py \
    --split en_2603 --subject math \
    --models gpt-5 claude-opus-4-6
```

### 4. Aggregate into Excel

```bash
# Per (split, subject) sheet
python metric.py \
    --split en_2603 --subject math \
    --models gpt-5 claude-opus-4-6 gemini-3-pro

# Subset mode: aggregate metrics over questions belonging to one
# challenging subset. Each question carries a `subset` field of type
# list[str] (e.g. ["complex_layout", "long_reason"]); pass the subset
# name you want to slice on:
python metric.py --subset --subset-field complex_layout \
    --papers paper_a paper_b paper_c
python metric.py --subset --subset-field rigorous_process --papers ...
python metric.py --subset --subset-field long_reason --papers ...
```

*A question with `subset = ["complex_layout", "rigorous_process"]` will
be counted in both the `complex_layout` and `rigorous_process` slices.
For backward compatibility, a top-level boolean field named after the
subset is also accepted.*

To produce a cross-paper summary sheet on top of the workbook:

```bash
python util/average_metrics.py --xlsx ../metrics/your_run.xlsx
```

---

## Metrics

All metrics in the paper are computed by `evaluate.py` / `evaluate_process.py`
and aggregated into the workbook by `metric.py`.

| metric  | meaning                                                                  |
|---------|--------------------------------------------------------------------------|
| **ACC**       | Pass@1 final-answer accuracy. Proportion of questions whose extracted answer matches the ground truth. |
| **ARL**       | *Accuracy Reweighted by Length.* Acc reweighted by a log-ratio of the average response length to the model's actual length — rewards concise correct solutions. |
| **Acc≤r**     | Accuracy when the total generation budget (including thinking tokens) is hard-capped at ratio *r* of the context window — simulates a time/length constraint. The default knob in this repo is `accuracy_within_16k_tokens`. |
| **OCS**       | *Outcome Exam Score.* Per-paper score derived purely from final-answer correctness, distributed across correctly answered (sub-)parts. |
| **PES**       | *Process Exam Score.* Per-paper score that penalises three reasoning-process error types: Condition Interpretation Error (CIE), Logical Assumption Error (LAE), Deductive Reasoning Error (DRE). |
| **OES**       | *Overall Exam Score.* Weighted combination of OCS and PES (weight *w_p*) normalised to a 100-point scale — the headline "Mock Exam" number. |

The per-question score `ES` decomposes into outcome (`ES_O`) and process
(`ES_P`) components combined by the same `w_p`. See the paper for the
full definitions.

---

## Adding new papers (optional)

If you have OCR output (from MinerU or similar) and want to parse new
papers into the same schema, use `analyze/analyze.py`:

```bash
python analyze/analyze.py \
    --ocr-dir path/to/ocr_results \
    --save-dir analyze/analyzed_json/my_papers \
    --model gpt-5
```

The script asks the LLM to extract per-question fields defined in
`analyze/configs/chinese_k12_exam.py` and writes one JSON per paper. The output
uses the legacy Chinese field names (`题型`, `分值`, `题目`, `答案`,
`解答`, `图像`); `evaluate/util/dataset_loader.py` accepts both that
schema and the new English schema transparently.

---

## Citation

If LiveK12Bench is useful to you, please consider citing it:

```bibtex
@misc{livek12bench2026,
  title  = {LiveK12Bench},
  author = {Wang, Xiaohan and Yin, Mingze and Zhao, Yilin and Sinbadliu and Li, Dian},
  year   = {2026},
  url    = {https://github.com/QQ-MM/LiveK12Bench}
}
@misc{wang2026livek12bench,
      title={LiveK12Bench: Have Large Multimodal Models Truly Conquered High School-level Examinations?}, 
      author={Xiaohan Wang and Mingze Yin and Yilin Zhao and Gang Liu and Dian Li},
      year={2026},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2605.26781}, 
}
```

(GitHub's "Cite this repository" sidebar button is also available; it
reads from [`CITATION.cff`](CITATION.cff).)

---

## License

The code in this repository is released under the [Apache License 2.0](LICENSE).
The accompanying dataset on HuggingFace
([`Shawn-wxh/livek12bench`](https://huggingface.co/datasets/Shawn-wxh/livek12bench))
is released under CC BY-NC 4.0 — see the dataset card for details.
