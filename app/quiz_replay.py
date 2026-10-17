"""Instance-local replay of generated quizzes (never student sessions)."""
import re
import hashlib
import json
from pathlib import Path
import stat
from collections import OrderedDict
from threading import RLock

from app.quiz_generation import QuizRequest
from app.quiz_provider import MAX_QUESTIONS, QuizProviderError
from app.quiz import EvidenceRef, Question, Quiz


class QuizReplayProvider:
    """Bounded, memory-only cache scoped to this wrapper instance.

    Exact request fields (including order and topic whitespace/case) are keys.
    A global per-instance lock serializes even unrelated generation requests:
    simple duplicate coalescing, at the cost of throughput. Byte snapshots are
    checked before/after generation, not atomic with external store mutation;
    changes after the final check remain possible. This validates structure
    and artifact ownership, not factual grounding (the wrapped provider's job).
    Call clear() after changing model or retrieval configuration.
    """
    def __init__(self, store, provider, *, max_entries=128):
        if type(max_entries) is not int or max_entries < 1:
            raise QuizProviderError("Replay cache capacity must be a positive integer.")
        self._store = store
        self._provider = provider
        self._max_entries = max_entries
        self._cache = OrderedDict()
        self._lock = RLock()

    def clear(self):
        """Invalidate replay after changing model/retrieval configuration."""
        with self._lock:
            self._cache.clear()

    def __call__(self, request):
        with self._lock:
            return self._replay(request)

    def _replay(self, request):
        if (type(request) is not QuizRequest
                or type(request.materials) is not tuple or not request.materials
                or any(type(doc_id) is not str or not re.fullmatch(r"[0-9a-f]{16}", doc_id)
                       for doc_id in request.materials)
                or len(set(request.materials)) != len(request.materials)
                or type(request.count) is not int or not 1 <= request.count <= MAX_QUESTIONS
                or type(request.seed) is not int
                or (request.topic is not None and (type(request.topic) is not str
                                                   or len(request.topic) > 2000))):
            raise QuizProviderError("Invalid quiz request.")
        snapshot = {}
        for doc_id in request.materials:
            try:
                snapshot.update(self._snapshot((doc_id,)))
            except Exception:
                self._invalidate((doc_id,))
                raise QuizProviderError("Selected quiz materials are unavailable.") from None
            for key, (old, _) in list(self._cache.items()):
                if doc_id in old and old[doc_id] != snapshot[doc_id]:
                    del self._cache[key]
        if request not in self._cache:
            try:
                quiz = self._provider(request)
                self._validate_quiz(quiz, request)
                if self._snapshot(request.materials) != snapshot:
                    raise ValueError
            except Exception:
                self._invalidate(request.materials)
                raise QuizProviderError("Unable to replay a valid quiz; please try again.") from None
            self._cache[request] = (snapshot, quiz)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        self._cache.move_to_end(request)
        return self._cache[request][1]

    def _invalidate(self, materials):
        for key in list(self._cache):
            if set(key.materials).intersection(materials):
                del self._cache[key]

    def _snapshot(self, materials):
        snapshots = {}
        for doc_id in materials:
            record = self._store.library_dir / f"{doc_id}.json"
            self._safe_path(record, self._store.root)
            record_bytes = record.read_bytes()
            doc = self._store.get(doc_id)
            if (doc.doc_id != doc_id or type(doc.filename) is not str
                    or not doc.filename or doc.filename in (".", "..")
                    or any(c in doc.filename for c in ("/", "\\", ":"))):
                raise ValueError
            corpus = self._store.corpus_dir / doc_id
            artifacts = self._store.artifacts_dir / doc_id
            self._safe_path(corpus / doc.filename, self._store.root)
            if not (corpus / doc.filename).is_file():
                raise ValueError
            files = [("record", hashlib.sha256(record_bytes).hexdigest())]
            for label, directory in (("corpus", corpus), ("artifacts", artifacts)):
                self._safe_path(directory, self._store.root)
                paths = []
                def visit(folder):
                    for path in sorted(folder.iterdir()):
                        self._safe_path(path, self._store.root)
                        if path.is_dir():
                            visit(path)
                        elif path.is_file():
                            paths.append(path)
                        else:
                            raise ValueError
                visit(directory)
                if not paths:
                    raise ValueError
                for path in paths:
                    files.append((f"{label}/{path.relative_to(directory).as_posix()}",
                                  hashlib.sha256(path.read_bytes()).hexdigest()))
            manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
            if (manifest["doc_id"] != doc_id or type(manifest["pages"]) is not list
                    or not manifest["pages"]):
                raise ValueError
            numbers = set()
            for page in manifest["pages"]:
                if (type(page["number"]) is not int or page["number"] < 1
                        or page["number"] in numbers or type(page["text"]) is not str
                        or type(page["image_path"]) is not str or not page["image_path"]):
                    raise ValueError
                numbers.add(page["number"])
                image = Path(page["image_path"])
                self._safe_path(image, artifacts)
                if not image.is_file():
                    raise ValueError
            snapshots[doc_id] = tuple(files)
        return snapshots

    def _validate_quiz(self, quiz, request):
        if (type(quiz) is not Quiz or type(quiz.seed) is not int or quiz.seed != request.seed
                or type(quiz.questions) is not tuple or len(quiz.questions) != request.count):
            raise ValueError
        for question in quiz.questions:
            if (type(question) is not Question
                    or type(question.prompt) is not str or type(question.explanation) is not str
                    or type(question.options) is not tuple
                    or any(type(option) is not str for option in question.options)
                    or type(question.correct_index) is not int):
                raise ValueError
            Question.__post_init__(question)
            ref = question.evidence
            if (type(ref) is not EvidenceRef or type(ref.document) is not str
                    or type(ref.page) is not int or type(ref.doc_id) is not str
                    or ref.doc_id not in request.materials
                    or (ref.excerpt is not None and type(ref.excerpt) is not str)
                    or (ref.image_path is not None and type(ref.image_path) is not str)):
                raise ValueError
            EvidenceRef.__post_init__(ref)
            doc = self._store.get(ref.doc_id)
            if ref.document != doc.filename:
                raise ValueError
            artifacts = self._store.artifacts_dir / ref.doc_id
            manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
            pages = [page for page in manifest["pages"] if page["number"] == ref.page]
            if len(pages) != 1:
                raise ValueError
            if ref.image_path is not None:
                self._safe_path(Path(ref.image_path), artifacts)
                if Path(ref.image_path).resolve() != Path(pages[0]["image_path"]).resolve():
                    raise ValueError

    @staticmethod
    def _safe_path(path, root):
        path, root = Path(path).absolute(), Path(root).absolute()
        path.relative_to(root)
        # Check every lexical ancestor, including the configured store root.
        for entry in (path, *path.parents):
            info = entry.lstat()
            if (entry.is_symlink() or
                    getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
                raise ValueError
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
