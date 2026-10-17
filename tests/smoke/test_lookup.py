"""Lookup uses real PDF parsing and BM25; never calls class services."""

import pytest

from app.documents import DocumentStore
from app.indexing import CorpusIndexer
from app.ui import add_file

pytestmark = pytest.mark.smoke


@pytest.fixture
def library(tmp_path):
    store = DocumentStore(root=tmp_path / 'data')
    indexer = CorpusIndexer(store, embedder=None, persist_dir=tmp_path / 'indexes')
    indexer.register_hooks()
    return store, indexer


def test_question_returns_pdf_passages_with_filename_and_page(library, tiny_pdf):
    from app.lookup import lookup_question

    store, indexer = library
    add_file(store, str(tiny_pdf), indexer)
    result = lookup_question(store, 'What is retrieval augmented generation?', indexer)
    assert 'mini.pdf' in result
    assert 'Page 1' in result
    assert 'retrieval augmented generation quiz prep slides' in result
    assert 'Matching passages' in result


@pytest.mark.parametrize('question', [None, '', '   ', 123, True])
def test_invalid_question_is_helpful(library, question):
    from app.lookup import lookup_question

    store, indexer = library
    assert 'Enter a question' in lookup_question(store, question, indexer)


def test_overlong_question_is_rejected(library):
    from app.lookup import lookup_question

    store, indexer = library
    assert '2,000 characters' in lookup_question(store, 'x' * 2001, indexer)


def test_empty_library_prompts_for_pdf(library):
    from app.lookup import lookup_question

    store, indexer = library
    assert 'Add a PDF' in lookup_question(store, 'retrieval', indexer)


def test_missing_indexer_does_not_crash(library, tiny_pdf):
    from app.lookup import lookup_question

    store, _ = library
    add_file(store, str(tiny_pdf))
    assert 'unavailable' in lookup_question(store, 'retrieval')


@pytest.mark.parametrize('question', ['zzzzunmatched', 'the and or'])
def test_no_match_is_not_presented_as_an_answer(library, tiny_pdf, question):
    from app.lookup import lookup_question

    store, indexer = library
    add_file(store, str(tiny_pdf), indexer)
    result = lookup_question(store, question, indexer)
    assert 'No matching passages' in result
    assert '<article>' not in result


def test_search_failure_hides_internal_details(library, tiny_pdf, monkeypatch, caplog):
    from app.lookup import lookup_question

    store, indexer = library
    add_file(store, str(tiny_pdf), indexer)

    def fail(*args, **kwargs):
        raise RuntimeError('sensitive-internal-detail')

    monkeypatch.setattr(indexer.keyword, 'search', fail)
    result = lookup_question(store, 'retrieval', indexer)
    assert 'temporarily unavailable' in result
    assert 'sensitive-internal-detail' not in result + caplog.text


def test_deleted_document_cannot_return_from_stale_index(library, tiny_pdf):
    from app.lookup import lookup_question

    store, indexer = library
    add_file(store, str(tiny_pdf), indexer)
    # Use an independent store instance without index hooks to simulate stale search data.
    DocumentStore(root=store.root).remove(store.list()[0].doc_id)
    result = lookup_question(store, 'retrieval', indexer)
    assert 'mini.pdf' not in result
    assert '<article>' not in result


def test_pdf_text_and_filename_are_escaped(library, tiny_pdf):
    from app.chunking import Chunk
    from app.lookup import lookup_question

    store, indexer = library
    safe_path = tiny_pdf.rename(tiny_pdf.with_name('study&notes.pdf'))
    add_file(store, str(safe_path), indexer)
    doc = store.list()[0]
    indexer.keyword.replace_all([Chunk('unsafe', doc.doc_id, doc.filename, '.pdf', 1, 0,
                                    'retrieval <script>alert(1)</script> **not bold**')])
    result = lookup_question(store, 'retrieval', indexer)
    assert 'study&amp;notes.pdf' in result
    assert '<script>' not in result
    assert '&lt;script&gt;' in result
    assert '**not bold**' in result


def test_lookup_does_not_modify_library_files(library, tiny_pdf):
    from app.lookup import lookup_question

    store, indexer = library
    add_file(store, str(tiny_pdf), indexer)
    before = {p.relative_to(store.root): p.read_bytes() for p in store.root.rglob('*') if p.is_file()}
    lookup_question(store, 'retrieval', indexer)
    after = {p.relative_to(store.root): p.read_bytes() for p in store.root.rglob('*') if p.is_file()}
    assert before == after


def test_lookup_is_below_library_with_click_and_enter_actions(library):
    from app.ui import build_ui

    store, indexer = library
    demo = build_ui(store, indexer)
    config = demo.get_config_file()
    components = config['components']
    labels = {component['props'].get('label'): component['id'] for component in components}
    assert 'Question' in labels
    question_id = labels['Question']
    assert labels['Library'] < labels['Select a document to remove'] < question_id
    assert any(c['type'] == 'markdown' and c['props'].get('value') == '## Lookup' for c in components)
    lookup_events = [d for d in config['dependencies'] if question_id in d['inputs']]
    assert len(lookup_events) == 2
    assert {target[1] for d in lookup_events for target in d['targets']} == {'click', 'submit'}
    assert all(d['outputs'] for d in lookup_events)
    assert any(c['props'].get('elem_id') == 'lookup-heading' for c in components)
    assert any(c['props'].get('elem_id') == 'lookup-button' for c in components)


def test_non_pdf_documents_are_excluded_even_if_indexed(library, tmp_path, tiny_pdf):
    from app.chunking import Chunk
    from app.lookup import lookup_question

    store, indexer = library
    add_file(store, str(tiny_pdf), indexer)
    slides = tmp_path / 'slides.pptx'
    slides.write_bytes(b'lookup-only test document; not a real slide deck')
    doc = store.add(slides)
    indexer.keyword.add([Chunk('slide', doc.doc_id, doc.filename, '.pptx', 1, 0, 'retrieval secret-slide-marker')])
    result = lookup_question(store, 'secret-slide-marker', indexer)
    assert 'slides.pptx' not in result
    assert 'secret-slide-marker' not in result


def test_results_are_bounded_to_five_passages(library, tiny_pdf):
    from app.chunking import Chunk
    from app.lookup import lookup_question

    store, indexer = library
    add_file(store, str(tiny_pdf), indexer)
    doc = store.list()[0]
    indexer.keyword.replace_all([Chunk(str(i), doc.doc_id, doc.filename, '.pdf', i+1, 0, 'retrieval') for i in range(8)])
    assert lookup_question(store, 'retrieval', indexer).count('<article>') == 5


def test_removal_during_search_drops_already_retrieved_passages(library, tiny_pdf, monkeypatch):
    from app.lookup import lookup_question

    store, indexer = library
    add_file(store, str(tiny_pdf), indexer)
    original_search = indexer.keyword.search

    def remove_after_search(*args, **kwargs):
        hits = original_search(*args, **kwargs)
        store.remove(store.list()[0].doc_id)
        return hits

    monkeypatch.setattr(indexer.keyword, 'search', remove_after_search)
    result = lookup_question(store, 'retrieval', indexer)
    assert '<article>' not in result
    assert 'mini.pdf' not in result
