"""
eval/ingest_eval_docs.py

Ingests one or more files into the vector store under a FIXED session_id
("eval" by default), so that eval_dataset.json can reliably query against
the same content every time (instead of relying on whatever the last
Streamlit browser session happened to upload).

Usage:
    python -m eval.ingest_eval_docs path/to/doc1.pdf path/to/doc2.txt
    python -m eval.ingest_eval_docs --session-id eval_v2 path/to/doc.pdf
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor


def main():
    parser = argparse.ArgumentParser(description="Ingest documents for a fixed eval session.")
    parser.add_argument("files", nargs="+", help="Paths to PDF/TXT/CSV files to ingest")
    parser.add_argument("--session-id", default="eval", help="Session ID to tag these documents with")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    processor = DocumentProcessor()

    for path in args.files:
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        print(f"Processing {path} ...")
        chunks = processor.process_document(path, os.path.basename(path))

        ok = pipeline.add_documents(
            documents=chunks,
            source_type="document",
            source_name=os.path.basename(path),
            session_id=args.session_id,
        )
        status = "OK" if ok else "FAILED"
        print(f"  -> {status}: {len(chunks)} chunks under session_id='{args.session_id}'")


if __name__ == "__main__":
    main()