"""Offline replay contract over disposable, synthetic stored materials."""
import json
from dataclasses import replace

import pytest

from app.documents import DocumentStore
from app.quiz import EvidenceRef, Question, Quiz, QuizSession
from app.quiz_generation import QuizRequest
from app.quiz_provider import QuizProviderError

pytestmark = pytest.mark.unit


@pytest.fixture
def material(tmp_path):
    store = DocumentStore(tmp_path / "data")
    source = tmp_path / "Lecture.pdf"
    source.write_bytes(b"synthetic corpus")
    doc = store.add(source)
    artifacts = store.artifacts_dir / doc.doc_id
    artifacts.mkdir()
    image = artifacts / "page.png"
    image.write_bytes(b"original image")
    manifest = {"doc_id": doc.doc_id, "format": ".pdf", "converted_from": None,
                "pages": [{"number": 1, "text": "BM25 matches terms", "image_path": str(image),
                           "embedded_images": 0}]}
    (artifacts / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return store, doc, artifacts, image


def quiz_for(request, doc, image, serial=0):
    ref = EvidenceRef(doc.filename, 1, "BM25 matches terms", str(image), doc.doc_id)
    question = Question(f"Method? {serial}", ("BM25", "OCR", "ASR", "RNN"),
                        serial % 4, "BM25 matches terms", ref)
    return Quiz((question,) * request.count, request.seed)


def test_replays_exact_quiz_from_nondeterministic_provider(material):
    from app.quiz_replay import QuizReplayProvider
    store, doc, _, image = material
    calls = []

    def provider(request):
        calls.append(request)
        return quiz_for(request, doc, image, len(calls))

    replay = QuizReplayProvider(store, provider)
    request = QuizRequest((doc.doc_id,), None, 1, 42)
    first = replay(request)
    assert replay(request) is first
    assert replay(request).answer_key == first.answer_key
    assert calls == [request]
    one, two = QuizSession(first), QuizSession(replay(request))
    one.reveal()
    assert not two.student_view()["completed"]


@pytest.mark.parametrize("changes", [
    {"materials": []}, {"materials": ()}, {"materials": ("../escape",)},
    {"materials": ("C:\\escape",)}, {"materials": ("a" * 16 + "/x",)},
    {"materials": ("g" * 16,)}, {"materials": ("A" * 16,)},
    {"materials": ("a" * 16, "a" * 16)}, {"count": True}, {"count": 0},
    {"count": 6}, {"seed": False}, {"seed": "42"}, {"topic": []},
    {"topic": "x" * 2001},
])
def test_invalid_request_precedes_filesystem_and_provider(changes):
    from app.quiz_replay import QuizReplayProvider
    request = object.__new__(QuizRequest)
    for field, value in dict(materials=("a" * 16,), topic=None, count=1, seed=42,
                             ).items():
        object.__setattr__(request, field, changes.get(field, value))

    class NoAccess:
        def __getattr__(self, name):
            pytest.fail("invalid request touched the store")

    def no_provider(request):
        pytest.fail("invalid request reached provider")

    with pytest.raises(QuizProviderError):
        QuizReplayProvider(NoAccess(), no_provider)(request)


def test_wrong_request_type_is_safe():
    from app.quiz_replay import QuizReplayProvider
    with pytest.raises(QuizProviderError):
        QuizReplayProvider(None, None)(None)


@pytest.mark.parametrize("target", ["corpus", "record", "manifest", "image", "extra"])
def test_byte_changes_invalidate_replay_even_with_same_size_and_mtime(material, target):
    import os
    from app.quiz_replay import QuizReplayProvider
    store, doc, artifacts, image = material
    extra = artifacts / "nested" / "text.txt"
    extra.parent.mkdir()
    extra.write_bytes(b"text one")
    paths = {"corpus": store.corpus_path(doc.doc_id),
             "record": store.library_dir / f"{doc.doc_id}.json",
             "manifest": artifacts / "manifest.json", "image": image, "extra": extra}
    calls = []

    def provider(request):
        calls.append(request)
        return quiz_for(request, doc, image, len(calls))

    replay = QuizReplayProvider(store, provider)
    request = QuizRequest((doc.doc_id,), None, 1, 42)
    first = replay(request)
    path = paths[target]
    old, stat = path.read_bytes(), path.stat()
    if target == "record":
        new = old.replace(b'"size_bytes": 16', b'"size_bytes": 17')
    elif target == "manifest":
        new = old.replace(b"matches", b"MATCHES")
    else:
        new = bytes([old[0] ^ 1]) + old[1:]
    assert new != old and len(new) == len(old)
    path.write_bytes(new)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = replay(request)
    assert second is not first
    assert replay(request) is second
    assert len(calls) == 2
    path.write_bytes(old)
    assert replay(request) is not first  # old fingerprint cannot resurrect old quiz
    assert len(calls) == 3


@pytest.mark.parametrize("target", ["record", "corpus", "manifest", "image"])
def test_missing_selected_material_invalidates_hits_and_restoration(material, target):
    from app.quiz_replay import QuizReplayProvider
    store, doc, artifacts, image = material
    paths = {"record": store.library_dir / f"{doc.doc_id}.json",
             "corpus": store.corpus_path(doc.doc_id),
             "manifest": artifacts / "manifest.json", "image": image}
    calls = []

    def provider(request):
        calls.append(request)
        return quiz_for(request, doc, image, len(calls))

    replay = QuizReplayProvider(store, provider)
    request = QuizRequest((doc.doc_id,), None, 1, 42)
    first = replay(request)
    path = paths[target]
    old = path.read_bytes()
    path.unlink()
    with pytest.raises(QuizProviderError):
        replay(request)
    assert len(calls) == 1
    path.write_bytes(old)
    assert replay(request) is not first
    assert len(calls) == 2


@pytest.mark.parametrize("attack", ["record", "manifest", "image", "filename", "symlink"])
def test_unsafe_or_mismatched_material_is_rejected_before_provider(material, tmp_path, attack):
    from app.quiz_replay import QuizReplayProvider
    store, doc, artifacts, image = material
    record = store.library_dir / f"{doc.doc_id}.json"
    manifest = artifacts / "manifest.json"
    if attack in ("record", "filename"):
        data = json.loads(record.read_text())
        data["doc_id" if attack == "record" else "filename"] = (
            "f" * 16 if attack == "record" else "../escape.pdf")
        record.write_text(json.dumps(data))
    elif attack in ("manifest", "image"):
        data = json.loads(manifest.read_text())
        if attack == "manifest":
            data["doc_id"] = "f" * 16
        else:
            outside = tmp_path / "outside.png"
            outside.write_bytes(b"outside")
            data["pages"][0]["image_path"] = str(outside)
        manifest.write_text(json.dumps(data))
    else:
        outside = tmp_path / "outside.txt"
        outside.write_text("private synthetic marker")
        try:
            (artifacts / "linked.txt").symlink_to(outside)
        except OSError:
            pytest.skip("host does not permit symlink creation")

    def no_provider(request):
        pytest.fail("unsafe material reached provider")

    with pytest.raises(QuizProviderError) as error:
        QuizReplayProvider(store, no_provider)(QuizRequest((doc.doc_id,), None, 1, 42))
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize("mutation", ["change", "delete", "provider-error"])
def test_generation_failure_is_safe_and_never_cached(material, mutation):
    from app.quiz_replay import QuizReplayProvider
    store, doc, artifacts, image = material
    calls = []
    old = image.read_bytes()

    def provider(request):
        calls.append(request)
        if len(calls) == 1:
            if mutation == "change":
                image.write_bytes(b"changed image")
            elif mutation == "delete":
                image.unlink()
            else:
                raise RuntimeError("synthetic-private-upstream-marker")
        return quiz_for(request, doc, image, len(calls))

    replay = QuizReplayProvider(store, provider)
    request = QuizRequest((doc.doc_id,), None, 1, 42)
    with pytest.raises(QuizProviderError) as error:
        replay(request)
    assert "synthetic-private-upstream-marker" not in str(error.value)
    assert error.value.__suppress_context__
    image.write_bytes(old)
    result = replay(request)
    assert len(calls) == 2
    assert replay(request) is result


@pytest.mark.parametrize("bad", [
    "not-quiz", "session", "questions-list", "seed", "count", "options-list",
    "prompt", "explanation", "options", "correct-index", "evidence", "document",
    "page", "excerpt", "image-type", "foreign-doc", "foreign-image", "wrong-page",
])
def test_malformed_or_mutable_provider_result_is_not_cached(material, tmp_path, bad):
    from app.quiz_replay import QuizReplayProvider
    store, doc, artifacts, image = material
    request = QuizRequest((doc.doc_id,), None, 1, 42)
    good = quiz_for(request, doc, image)
    question, ref = good.questions[0], good.questions[0].evidence

    def forged(value, field, replacement):
        copy = replace(value)
        object.__setattr__(copy, field, replacement)
        return copy

    if bad == "not-quiz":
        invalid = None
    elif bad == "session":
        invalid = QuizSession(good)
    elif bad in ("questions-list", "seed", "count"):
        field, value = {"questions-list": ("questions", list(good.questions)),
                        "seed": ("seed", 9), "count": ("questions", ())}[bad]
        invalid = forged(good, field, value)
    elif bad in ("options-list", "prompt", "explanation", "options", "correct-index", "evidence"):
        field, value = {"options-list": ("options", list(question.options)),
                        "prompt": ("prompt", []), "explanation": ("explanation", " "),
                        "options": ("options", ("x", "x", "y", "z")),
                        "correct-index": ("correct_index", True),
                        "evidence": ("evidence", {})}[bad]
        invalid = replace(good, questions=(forged(question, field, value),))
    else:
        field, value = {"document": ("document", "Other.pdf"), "page": ("page", True),
                        "excerpt": ("excerpt", []), "image-type": ("image_path", []),
                        "foreign-doc": ("doc_id", "f" * 16),
                        "foreign-image": ("image_path", str(tmp_path / "foreign.png")),
                        "wrong-page": ("page", 2)}[bad]
        invalid = replace(good, questions=(replace(question, evidence=forged(ref, field, value)),))
    calls = []

    def provider(request):
        calls.append(request)
        return invalid if len(calls) == 1 else good

    replay = QuizReplayProvider(store, provider)
    with pytest.raises(QuizProviderError):
        replay(request)
    assert replay(request) is good
    assert len(calls) == 2


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "2", None])
def test_cache_capacity_must_be_positive_integer(limit):
    from app.quiz_replay import QuizReplayProvider
    with pytest.raises(ValueError):
        QuizReplayProvider(None, None, max_entries=limit)


def test_lru_eviction_and_explicit_clear(material):
    from app.quiz_replay import QuizReplayProvider
    store, doc, _, image = material
    calls = []

    def provider(request):
        calls.append(request)
        return quiz_for(request, doc, image, len(calls))

    replay = QuizReplayProvider(store, provider, max_entries=2)
    a, b, c = [QuizRequest((doc.doc_id,), None, 1, seed) for seed in range(3)]
    first, second = replay(a), replay(b)
    assert replay(a) is first  # touch makes b least-recently used
    replay(c)
    assert replay(a) is first
    assert replay(b) is not second
    assert len(calls) == 4
    replay.clear()
    assert replay(a) is not first
    assert len(calls) == 5


def test_default_capacity_is_128(material):
    from app.quiz_replay import QuizReplayProvider
    store, doc, _, image = material
    calls = []

    def provider(request):
        calls.append(request)
        return quiz_for(request, doc, image, len(calls))

    replay = QuizReplayProvider(store, provider)
    first_request = QuizRequest((doc.doc_id,), None, 1, 0)
    first = replay(first_request)
    for seed in range(1, 128):
        replay(replace(first_request, seed=seed))
    assert replay(first_request) is first
    replay(replace(first_request, seed=128))
    replay(replace(first_request, seed=1))
    assert len(calls) == 130


def test_concurrent_duplicate_requests_coalesce(material):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event, Lock
    from app.quiz_replay import QuizReplayProvider
    store, doc, _, image = material
    start, entered, duplicate, release = Barrier(3), Event(), Event(), Event()
    calls, guard = [], Lock()

    def provider(request):
        with guard:
            calls.append(request)
            serial = len(calls)
            if serial > 1:
                duplicate.set()
        entered.set()
        assert release.wait(5), "test did not release provider"
        return quiz_for(request, doc, image, serial)

    replay = QuizReplayProvider(store, provider)
    request = QuizRequest((doc.doc_id,), None, 1, 42)

    def invoke():
        start.wait(timeout=5)
        return replay(request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(invoke) for _ in range(2)]
        try:
            start.wait(timeout=5)
            assert entered.wait(5)
            duplicate.wait(0.3)
        finally:
            release.set()
        results = [future.result(timeout=5) for future in futures]
    assert len(calls) == 1
    assert results[0] is results[1]


@pytest.mark.parametrize("damage", ["missing-corpus", "page-bool", "duplicate-page", "no-text", "no-image"])
def test_incomplete_parsed_material_fails_closed(material, damage):
    from app.quiz_replay import QuizReplayProvider
    store, doc, artifacts, image = material
    manifest_path = artifacts / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "missing-corpus":
        store.corpus_path(doc.doc_id).unlink()
        (store.corpus_dir / doc.doc_id / "unrelated.txt").write_text("not original")
    elif damage == "page-bool":
        manifest["pages"][0]["number"] = True
    elif damage == "duplicate-page":
        manifest["pages"].append(manifest["pages"][0])
    elif damage == "no-text":
        del manifest["pages"][0]["text"]
    else:
        del manifest["pages"][0]["image_path"]
    manifest_path.write_text(json.dumps(manifest))

    def provider(request):
        pytest.fail("incomplete materials reached provider")

    with pytest.raises(QuizProviderError):
        QuizReplayProvider(store, provider)(QuizRequest((doc.doc_id,), None, 1, 42))


def test_exact_request_fields_and_wrapper_instances_are_separate(material):
    from app.quiz_replay import QuizReplayProvider
    store, doc, artifacts, image = material
    source = artifacts.parent.parent.parent / "Other.pdf"
    source.write_bytes(b"other synthetic corpus")
    other = store.add(source)
    other_artifacts = store.artifacts_dir / other.doc_id
    other_artifacts.mkdir()
    other_image = other_artifacts / "page.png"
    other_image.write_bytes(b"other original image")
    manifest = {"doc_id": other.doc_id, "pages": [{"number": 1, "text": "BM25",
                 "image_path": str(other_image)}]}
    (other_artifacts / "manifest.json").write_text(json.dumps(manifest))
    calls = []

    def provider(request):
        calls.append(request)
        return quiz_for(request, doc, image, len(calls))

    replay = QuizReplayProvider(store, provider)
    base = QuizRequest((doc.doc_id,), None, 1, 42)
    requests = [base, replace(base, topic=""), replace(base, topic="BM25"),
                replace(base, topic="bm25"), replace(base, topic=" BM25 "),
                replace(base, count=2), replace(base, seed=43),
                replace(base, materials=(doc.doc_id, other.doc_id)),
                replace(base, materials=(other.doc_id, doc.doc_id))]
    results = [replay(request) for request in requests]
    assert len(calls) == len(requests)
    for request, result in zip(requests, results):
        assert replay(request) is result
    assert QuizReplayProvider(store, provider)(base) is not results[0]
    # Removing even an uncited selected document must block an existing hit.
    store.remove(other.doc_id)
    with pytest.raises(QuizProviderError):
        replay(requests[-1])
    assert replay(base) is results[0]


def test_reparse_points_are_rejected_without_following_them(material, monkeypatch):
    from app.quiz_replay import QuizReplayProvider
    from pathlib import Path
    from types import SimpleNamespace
    store, doc, artifacts, image = material
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        if path == image:
            return SimpleNamespace(st_file_attributes=0x400)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)

    def provider(request):
        pytest.fail("reparse point reached provider")

    with pytest.raises(QuizProviderError):
        QuizReplayProvider(store, provider)(QuizRequest((doc.doc_id,), None, 1, 42))


def test_observed_change_invalidates_even_when_later_selected_doc_is_missing(material):
    from app.quiz_replay import QuizReplayProvider
    store, doc, _, image = material
    calls = []

    def provider(request):
        calls.append(request)
        return quiz_for(request, doc, image, len(calls))

    replay = QuizReplayProvider(store, provider)
    request = QuizRequest((doc.doc_id,), None, 1, 42)
    first = replay(request)
    original = image.read_bytes()
    image.write_bytes(b"changed original image")
    with pytest.raises(QuizProviderError):
        replay(replace(request, materials=(doc.doc_id, "f" * 16)))
    image.write_bytes(original)
    assert replay(request) is not first
