# Evaluation Harness

This repo is a RAG pipeline built on pretrained, frozen models (`mistral`,
`nomic-embed-text` via Ollama) — there's no model training or fine-tuning
step, so nothing to add there. What *was* missing was a way to answer "how
good is the retrieval, and how good are the answers?" This folder adds that.

## What gets measured

**Retrieval** (is the vector search finding the right chunks?)
- `precision_at_k` — of the k chunks retrieved, what fraction were actually relevant
- `recall_at_k` — of the expected relevant sources, what fraction got retrieved
- `mrr` — how high up the first relevant chunk ranked (1.0 = first result, 0 = never)
- `hit_rate` — did at least one relevant chunk get retrieved at all

**Generation** (is the LLM's answer any good?)
- `answer_relevance` — embedding similarity between the question and the answer (off-topic answers score low)
- `answer_correctness` — embedding similarity between the answer and a hand-written ground-truth answer (only computed if you supply one)
- `faithfulness` — LLM-as-judge score (0-1) for whether the answer's claims are actually supported by the retrieved context, i.e. a hallucination check

**Efficiency**
- Latency broken out per stage: embedding the query, vector search, LLM generation, and total

## Setup

1. Make sure the app's normal prerequisites are met: Ollama running (`ollama serve`) with `mistral` and `nomic-embed-text` pulled, and Qdrant credentials in `.env`.
2. Ingest a fixed set of documents under a stable session ID, so your eval results are reproducible instead of depending on whatever's currently in a browser session:
   ```bash
   python -m eval.ingest_eval_docs path/to/your_document.pdf
   ```
   This tags the chunks with `session_id="eval"` by default.
3. Edit `eval/eval_dataset.json`:
   - `question`: something you'd actually ask about the ingested doc(s)
   - `expected_sources`: filename(s) that should be retrieved for this question (used for precision/recall/MRR)
   - `ground_truth_answer`: the correct answer, written by hand — this is your "gold" answer to compare against
   - For a question that should *not* be answerable from your docs, leave `expected_sources` empty — this checks the system correctly says "I don't know" instead of hallucinating.

## Running

```bash
python -m eval.run_eval                 # default k=5, includes faithfulness judge
python -m eval.run_eval --k 3           # retrieve fewer chunks
python -m eval.run_eval --no-judge      # skip the LLM-judge call (faster, no faithfulness score)
```

Each run prints a per-question summary to the console and writes a full JSON
report to `eval/results/eval_report_<timestamp>.json`, including every
question's retrieved sources, generated answer, and all metric scores — useful
for a presentation appendix or to eyeball specific failures.

## Reading the results

- Low `recall_at_k` but high `precision_at_k` → your chunk size/overlap or `k` might be too small; relevant content exists but isn't all getting pulled in.
- High retrieval scores but low `faithfulness` → retrieval is working, but the LLM is adding claims not in the context (prompt or model issue, not a retrieval issue).
- Low `answer_relevance` → the model is answering a different question than what was asked, often a symptom of noisy/irrelevant retrieved context.
- The "no answer expected" case scoring `hit_rate=0` and a low-confidence-sounding answer is a *pass*, not a failure — it means the system isn't hallucinating from nothing.

## Growing this further

- Add more questions to `eval_dataset.json` as you add documents — aim for at least 15-20 for numbers that mean something.
- If you want to compare configurations (e.g. `k=3` vs `k=8`, or chunk_size 500 vs 1000), just re-run with different flags/settings and diff the `summary` blocks across two report files.