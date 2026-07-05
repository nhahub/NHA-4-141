"""
eval/run_eval.py

Runs the RAG pipeline against eval_dataset.json and reports retrieval +
generation metrics, plus per-stage latency.

Prerequisites:
  1. Ollama running locally (`ollama serve`) with `mistral` and
     `nomic-embed-text` pulled.
  2. Qdrant credentials set in your .env (same as the main app).
  3. The documents referenced in eval_dataset.json already ingested under
     session_id == the "session_id" field in eval_dataset.json (default: "eval").
     Easiest way: run the Streamlit app, upload the doc(s), and note the
     session ID shown in the sidebar -- or write a tiny ingestion script
     using RAGPipeline.add_documents() with a fixed session_id="eval".

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --k 3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline import RAGPipeline
from eval.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    hit_rate,
    answer_relevance,
    answer_correctness,
    faithfulness,
    summarize,
)


def run(dataset_path: str, k: int = 5, judge: bool = True):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    session_id = dataset.get("session_id", "eval")
    cases = dataset["cases"]

    pipeline = RAGPipeline()
    results = []

    print(f"Running {len(cases)} eval case(s) against session '{session_id}' (k={k})\n")

    for case in cases:
        qid = case["id"]
        question = case["question"]
        expected_sources = case.get("expected_sources", [])
        ground_truth = case.get("ground_truth_answer", "")

        print(f"[{qid}] {question}")

        # --- Stage 1: embed the query -----------------------------------
        t0 = time.perf_counter()
        query_embedding = pipeline.embedding_client.embed_query(question)
        t1 = time.perf_counter()

        # --- Stage 2: retrieve -------------------------------------------
        relevant_docs = pipeline.vector_store.similarity_search(
            query_embedding, k=k, filter_dict={"session_id": session_id}
        )
        t2 = time.perf_counter()

        retrieved_sources = [
            doc.metadata.get("source_name", "unknown") for doc in relevant_docs
        ]
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        # --- Stage 3: generate ---------------------------------------------
        if relevant_docs:
            answer = pipeline.llm_client.generate_response(question, context)
        else:
            answer = "I don't have any relevant information to answer your question. Please upload some documents first."
        t3 = time.perf_counter()

        # --- Metrics -------------------------------------------------------
        row = {
            "id": qid,
            "question": question,
            "answer": answer,
            "retrieved_sources": retrieved_sources,
            "embed_latency_s": round(t1 - t0, 3),
            "retrieval_latency_s": round(t2 - t1, 3),
            "generation_latency_s": round(t3 - t2, 3),
            "total_latency_s": round(t3 - t0, 3),
            "precision_at_k": precision_at_k(retrieved_sources, expected_sources),
            "recall_at_k": recall_at_k(retrieved_sources, expected_sources),
            "mrr": mean_reciprocal_rank(retrieved_sources, expected_sources),
            "hit_rate": hit_rate(retrieved_sources, expected_sources),
            "answer_relevance": answer_relevance(question, answer, pipeline.embedding_client),
        }

        correctness = answer_correctness(answer, ground_truth, pipeline.embedding_client)
        if correctness is not None:
            row["answer_correctness"] = correctness

        if judge and context:
            score = faithfulness(context, answer)
            if score is not None:
                row["faithfulness"] = score

        results.append(row)
        print(f"    precision@{k}={row['precision_at_k']:.2f}  recall@{k}={row['recall_at_k']:.2f}  "
              f"mrr={row['mrr']:.2f}  relevance={row['answer_relevance']:.2f}  "
              f"total_latency={row['total_latency_s']}s\n")

    summary = summarize(results)

    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "k": k,
        "num_cases": len(cases),
        "summary": summary,
        "results": results,
    }

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    out_path = os.path.join(
        os.path.dirname(__file__), "results",
        f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("SUMMARY (averaged across all cases)")
    print("=" * 60)
    for k_metric, v in summary.items():
        print(f"  {k_metric:<24} {v}")
    print(f"\nFull report saved to: {out_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG assistant.")
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "eval_dataset.json"))
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve per query")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM-as-judge faithfulness check (faster)")
    args = parser.parse_args()

    run(args.dataset, k=args.k, judge=not args.no_judge)