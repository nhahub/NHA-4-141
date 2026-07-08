"""
Step 1: Build an instruction-tuning dataset (train.jsonl) directly from your
own project documents.

How it works:
  1. Reuses the project's existing `src/document_processor.py` so chunking is
     100% consistent with what your RAG pipeline already does (same chunk
     size/overlap, same file types: pdf, txt, csv, docx, pptx, xlsx, md).
  2. For each chunk, asks your already-running Ollama model to generate a
     handful of question/answer pairs grounded in that chunk (same idea as
     "augment each document into instruction-style training instances" used
     in parametric-knowledge-injection literature) — this way you don't have
     to hand-write a dataset.
  3. Writes everything to finetune/data/train.jsonl in the format the
     training script (2_train_qlora.py) expects.

Usage:
  1. Put your documents in finetune/source_docs/  (any type the app already
     supports: pdf, docx, pptx, xlsx, csv, txt, md)
  2. Make sure Ollama is running (`ollama serve`) with GENERATOR_MODEL pulled
  3. Run: python finetune/1_prepare_dataset.py
"""
import os
import sys
import json
import time
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.document_processor import DocumentProcessor  # noqa: E402
from finetune import config  # noqa: E402

QA_GENERATION_PROMPT = """You will be given a passage of text. Generate exactly 3 diverse \
question-answer pairs that can be answered using only the information in the passage.

Respond with ONLY a JSON array, no other text, in this exact format:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

Passage:
{chunk}
"""


def call_ollama_generate(prompt: str, model: str, base_url: str, retries: int = 2) -> str:
    """Call Ollama's /api/generate endpoint and return the raw text response."""
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4},
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            if attempt == retries:
                print(f"  Warning: Ollama call failed after {retries + 1} attempts: {e}")
                return ""
            time.sleep(2)
    return ""


def extract_json_array(text: str):
    """Best-effort extraction of a JSON array from a model response that may
    contain extra commentary or markdown code fences around the JSON."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []


def build_training_example(question: str, answer: str) -> dict:
    """Format a QA pair as an instruction-tuning example matching the chat
    template of Qwen2.5-Instruct models."""
    text = (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )
    return {"text": text}


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    source_dir = os.path.abspath(os.path.join(PROJECT_ROOT, config.SOURCE_DOCS_DIR))

    if not os.path.isdir(source_dir) or not os.listdir(source_dir):
        print(f"No documents found in {source_dir}")
        print("Add some pdf/docx/pptx/xlsx/csv/txt/md files there first, then rerun this script.")
        return

    processor = DocumentProcessor()
    examples = []

    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    print(f"Found {len(files)} file(s) in {source_dir}")

    for filename in files:
        file_path = os.path.join(source_dir, filename)
        try:
            chunks = processor.process_document(file_path, filename)
        except ValueError as e:
            print(f"  Skipping {filename}: {e}")
            continue

        print(f"  {filename}: {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue  # skip near-empty chunks

            prompt = QA_GENERATION_PROMPT.format(chunk=chunk)
            raw_response = call_ollama_generate(prompt, config.GENERATOR_MODEL, config.OLLAMA_BASE_URL)
            qa_pairs = extract_json_array(raw_response)

            if not qa_pairs:
                print(f"    chunk {i}: no valid QA pairs generated, skipping")
                continue

            for pair in qa_pairs:
                q, a = pair.get("question", "").strip(), pair.get("answer", "").strip()
                if q and a:
                    examples.append(build_training_example(q, a))

            print(f"    chunk {i}: +{len(qa_pairs)} QA pairs (total so far: {len(examples)})")

    if not examples:
        print("No training examples were generated. Check that Ollama is running "
              "and GENERATOR_MODEL is pulled (ollama pull " + config.GENERATOR_MODEL + ").")
        return

    out_path = os.path.abspath(os.path.join(PROJECT_ROOT, config.DATASET_PATH))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(examples)} training examples to {out_path}")


if __name__ == "__main__":
    main()