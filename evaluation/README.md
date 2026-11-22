# Evaluation

Offline retrieval evaluation (no LLM calls, no API key): the same golden
questions run through every retrieval mode so the effect of BM25, MMR and the
reranker is measurable.

```powershell
python evaluation/evaluate.py                # all six modes, k=5
python evaluation/evaluate.py --k 10
python evaluation/evaluate.py --modes dense mmr mmr-rerank
```

Metrics are section-level, computed over the 13 answerable questions in
[`golden_questions.json`](golden_questions.json):

- **Hit@k** — correct section appears in the top k
- **Recall@k** — fraction of expected sections found (comparison questions
  expect two sections)
- **MRR** — 1 / rank of the first correct section
- **nDCG@k** — binary-gain ranking quality
- **sec@k** — mean number of distinct sections in the top k (diversity)

The three unanswerable questions (pricing, quota, step-by-step how-to) are
excluded from ranking metrics and reported separately with their top-1 rerank
score. That near-zero score is an offline diagnostic: the answering prompt
never sees scores, so refusal comes from the model judging the retrieved text
against rule 3 of the system prompt.

## Answer-level checks

Retrieval metrics say nothing about whether the generated answer is honest, so
`answer_eval.py` calls the real chain and checks what came back:

```powershell
python evaluation/answer_eval.py                    # all 16 questions
python evaluation/answer_eval.py --ids q01 q14      # spot check
python evaluation/answer_eval.py --show-answers     # print each answer
```

For every citation in an answer it requires both:

1. the cited page range overlaps a chunk that was actually retrieved, and
2. the cited label (service or section name) matches that chunk's section path,
   title, service or opening text.

Refusal accuracy is checked with a phrase heuristic ("does not contain",
"not covered", "no information", ...) against the three unanswerable questions,
and false refusals are reported for the answerable ones.

This costs one LLM call per question and paces itself (1.5 s between calls by
default) to stay inside free-tier rate limits. Citation precision measures
that a citation points at real retrieved context — it does **not** prove the
sentence is fully entailed by that page.

### Answer-level results

| Metric | Result |
|---|---|
| Citations checked | 55 |
| Invalid citations | 0 (precision 1.000) |
| Answerable questions with citations | 13/13 |
| Answerable answers fully supported | 13/13 |
| Refusals on unanswerable questions | 3/3 |
| False refusals | 0 |

The first run of this check exposed a false refusal on "What is Amazon
SageMaker AI?": reranking returned five child sections and never the parent
overview, so the model declined. Parent-section expansion in the retriever
(`EXPAND_PARENT_SECTIONS`) fixed it.

Gemini's free tier returned occasional 503/504s during the run, so the check
was executed in small batches; the numbers above aggregate those batches. The
script prints each question as it finishes and continues after errors, so a
provider hiccup never discards earlier results.

## Results (k=5, `RERANK_FETCH_K=10`)

| Mode | Hit@5 | Recall@5 | MRR | nDCG@5 | sec@5 | ms/query |
|---|---|---|---|---|---|---|
| dense | 1.000 | 1.000 | 0.962 | 0.962 | 4.5 | 108 |
| + MMR | 1.000 | 1.000 | 0.949 | 0.962 | 4.8 | 214 |
| + BM25 | 1.000 | 1.000 | 0.885 | 0.916 | 4.5 | 120 |
| + rerank | 1.000 | 1.000 | **1.000** | **0.994** | 4.6 | 4508 |
| + MMR + rerank | 1.000 | 1.000 | **1.000** | **0.994** | 4.8 | 4968 |
| + BM25 + rerank | 1.000 | 1.000 | **1.000** | **0.994** | 4.6 | 4228 |

Dense search already retrieves the right section in the top 5 for all 13
answerable questions, so the golden set measures ranking quality more than
recall. BM25 alone lowers MRR because category and comparison sections repeat
service names; the cross-encoder repairs that and reaches a perfect MRR.
Unanswerable questions score 0.002–0.004 after reranking, versus 0.89–0.99 for
answerable ones.

MMR adds no measurable quality: with reranking it leaves MRR/nDCG unchanged and
only raises section diversity from 4.6 to 4.8 of 5 — section-based chunks are
already nearly all distinct — while costing roughly 0.5 s per question. Without
reranking it slightly lowers MRR (0.962 → 0.949). **`USE_MMR` stays off by
default.**

Timings are CPU-only and vary by run: reranking measured 3.3–4.5 s per question
across runs on the same machine. `RERANK_FETCH_K` is the real latency knob —
20 gives the same metrics at ~6.8 s, 10 gives ~3.3 s, and 5 gives ~1.6 s but
lets the reranker only reorder the first pass:

```powershell
python evaluation/evaluate.py --modes dense-rerank hybrid-rerank --fetch-k 20
```

Section matching is exact by default. Questions 03 (EC2) and 09 (SageMaker AI)
use `"match": "subtree"`, because their service sections have child sections
that legitimately answer the question.
