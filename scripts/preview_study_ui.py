"""Offline browser fixture: synthetic content and model, no remote services.

Run python -m scripts.preview_study_ui. Never use this as the production
launcher; python -m app.ui uses real configured services instead.
"""
import argparse
import json
import tempfile
from pathlib import Path

import pymupdf

from app.documents import DocumentStore
from app.indexing import CorpusIndexer
from app.qna import QnaOrchestrator
from app.quiz_model import VisionQuizGenerator
from app.quiz_provider import QuizProvider
from app.quiz_replay import QuizReplayProvider
from app.retrieval import RetrievalService
from app.ui import LOOKUP_CSS, add_file, build_ui


class SyntheticModel:
    def chat(self, messages):
        if "generate multiple-choice questions" in messages[0]["content"]:
            count = json.loads(messages[1]["content"])["count"]
            return json.dumps({"questions": [{
                "prompt": f"Which method matches query terms? (Example {i + 1})",
                "options": ["BM25", "Image compression", "Speech synthesis", "Video encoding"],
                "correct_index": 0, "explanation": "BM25 matches query terms in documents.",
                "evidence_id": "E1",
            } for i in range(count)]})
        return json.dumps({"answer": "BM25 retrieves documents by matching query terms.",
            "grounded": True, "sources": [{"document": "Synthetic UI test.pdf", "page": 1,
            "excerpt": "BM25 retrieves documents by matching query terms."}]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="study-ui-fixture-") as directory:
        root = Path(directory)
        pdf = root / "Synthetic UI test.pdf"
        with pymupdf.open() as document:
            page = document.new_page(width=640, height=420)
            page.insert_text((40, 55), "Synthetic course notes: keyword retrieval", fontsize=22)
            page.insert_text((40, 100), "BM25 retrieves documents by matching query terms.", fontsize=18)
            page.draw_rect(pymupdf.Rect(40, 150, 210, 230), color=(0.1, 0.3, 0.7), fill=(0.9, 0.94, 1))
            page.insert_text((65, 195), "Query terms", fontsize=20)
            page.draw_line((215, 190), (350, 190), color=(0.1, 0.3, 0.7), width=3)
            page.draw_rect(pymupdf.Rect(355, 150, 600, 230), color=(0.1, 0.3, 0.7), fill=(0.9, 0.94, 1))
            page.insert_text((370, 195), "Matching documents", fontsize=18)
            page.insert_text((40, 300), "Offline interface fixture: synthetic content and injected model.", fontsize=13)
            document.save(pdf)
        store = DocumentStore(root / "data")
        indexer = CorpusIndexer(store, persist_dir=root / "index")
        add_file(store, str(pdf), indexer)
        # Reconcile before binding services; build_ui reconciles once more.
        # The fixture service follows the current index through this proxy.
        class CurrentKeyword:
            def search(self, *args, **kwargs):
                return indexer.keyword.search(*args, **kwargs)
        service = RetrievalService(CurrentKeyword(), store=store)
        model = SyntheticModel()
        provider = QuizReplayProvider(store, QuizProvider(service, VisionQuizGenerator(model)))
        build_ui(store, indexer, quiz_provider=provider, retrieval=service,
                 qna=QnaOrchestrator(model)).launch(
                     server_name="127.0.0.1", server_port=args.port, inbrowser=False, css=LOOKUP_CSS)


if __name__ == "__main__":
    main()
