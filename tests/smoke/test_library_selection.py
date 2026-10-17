"""Row selection and targeted removal, using generated PDFs only."""
import asyncio
import json

import gradio as gr
import pymupdf
import pytest
from gradio.state_holder import SessionState

from app.documents import DocumentStore
from app.ui import add_file, build_ui

pytestmark = pytest.mark.smoke


@pytest.fixture
def library(tmp_path):
    store = DocumentStore(root=tmp_path / 'data')
    for name, text in [('alpha.pdf', 'alpha astronomy'), ('beta.pdf', 'beta biology'),
                       ('gamma.pdf', 'gamma geography')]:
        path = tmp_path / name
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), text)
            pdf.save(path)
        add_file(store, str(path))
    return store


def test_row_selection_uses_stable_id(library):
    from app.ui import select_library_document

    documents = library.list()
    ids = [doc.doc_id for doc in documents]
    event = gr.SelectData(None, {'index': [1, 0], 'value': documents[1].filename})
    assert select_library_document(library, ids, event)['value'] == ids[1]
    assert [doc.doc_id for doc in library.list()] == ids


def test_dropdown_options_and_row_ids_are_refreshed_together(library):
    from app.ui import _choices, _library_updates

    frame, update, message, ids = _library_updates(library, 'Ready.')
    assert update['__type__'] == 'update'
    assert update['value'] is None
    assert update['choices'] == _choices(library)
    assert ids == [doc.doc_id for doc in library.list()]
    assert frame['filename'].tolist() == [name for name, _ in update['choices']]
    assert message == 'Ready.'


def test_actual_gradio_selection_removes_only_target_and_refreshes_quiz(library, tmp_path):
    from app.indexing import CorpusIndexer
    from tests.fake_embeddings import FakeEmbedder

    indexer = CorpusIndexer(library, embedder=FakeEmbedder(), persist_dir=tmp_path / 'indexes')
    demo = build_ui(library, indexer)
    session = SessionState(demo)
    config = demo.get_config_file()
    components = {c['props'].get('label'): c['id'] for c in config['components']}
    functions = {fn.api_name: number for number, fn in demo.fns.items()}
    select_events = [d for d in config['dependencies']
                     if (components['Library'], 'select') in [tuple(t) for t in d['targets']]]
    assert len(select_events) == 1
    target = library.list()[1]
    survivors = [doc for doc in library.list() if doc.doc_id != target.doc_id]
    original_bytes = {doc.doc_id: library.corpus_path(doc.doc_id).read_bytes() for doc in survivors}

    async def exercise():
        loaded = await demo.process_api(functions['load_library'], [], state=session)
        assert loaded['data'][1]['__type__'] == 'update'
        assert len(loaded['data'][1]['choices']) == 3
        event = gr.SelectData(demo.blocks[components['Library']], {'index': [1, 0], 'value': target.filename})
        selected = await demo.process_api(select_events[0]['id'], [None], state=session, event_data=event)
        assert selected['data'][0]['value'] == target.doc_id
        removed = await demo.process_api(functions['remove_selected'], [target.doc_id], state=session)
        assert removed['data'][1]['value'] is None
        assert {value for _, value in removed['data'][1]['choices']} == {d.doc_id for d in survivors}
        assert len(removed['data'][0]['data']) == 2
        refresh = next(d for d in config['dependencies'] if d.get('trigger_after') == functions['remove_selected'])
        assert refresh['outputs'] == [components['Course materials']]
        quiz_choices = await demo.process_api(refresh['id'], [], state=session)
        assert {value for _, value in quiz_choices['data'][0]['choices']} == {d.doc_id for d in survivors}
        assert quiz_choices['data'][0]['value'] == []

    asyncio.run(exercise())
    assert {doc.doc_id for doc in library.list()} == {doc.doc_id for doc in survivors}
    assert target.doc_id not in {c.doc_id for c in indexer.keyword.chunks()}
    assert target.doc_id not in indexer.vector.stored_doc_ids()
    assert not (library.corpus_dir / target.doc_id).exists()
    assert not (library.artifacts_dir / target.doc_id).exists()
    for doc in survivors:
        assert library.corpus_path(doc.doc_id).read_bytes() == original_bytes[doc.doc_id]
        term = doc.filename.split('.')[0]
        assert indexer.keyword.search(term, doc_ids=[doc.doc_id])
        assert indexer.vector.search(term, doc_ids=[doc.doc_id])


@pytest.mark.parametrize('choice', ['../outside', 'alpha.pdf', 'invalid-id', 123, ['bad']])
def test_invalid_removal_ids_do_not_access_store(library, monkeypatch, choice):
    from app.ui import remove_selected

    calls = []
    original_get = library.get
    def observed_get(doc_id):
        calls.append(doc_id)
        return original_get(doc_id)
    monkeypatch.setattr(library, 'get', observed_get)
    before = [d.doc_id for d in library.list()]
    _, _, message = remove_selected(library, choice)
    assert calls == []
    assert 'Select a valid document' in message
    assert [d.doc_id for d in library.list()] == before


@pytest.mark.parametrize('index', [None, -1, [], [-1, 0], [999, 0], [True, 0], ['1', 0]])
def test_invalid_row_events_clear_selection(library, index):
    from app.ui import select_library_document

    ids = [d.doc_id for d in library.list()]
    event = gr.SelectData(None, {'index': index, 'value': None})
    assert select_library_document(library, ids, event)['value'] is None
    assert [d.doc_id for d in library.list()] == ids


def test_deselection_clears_choice(library):
    from app.ui import select_library_document

    event = gr.SelectData(None, {'index': [0, 0], 'value': None, 'selected': False})
    assert select_library_document(library, [d.doc_id for d in library.list()], event)['value'] is None


def test_deleted_row_does_not_shift_to_another_document(library):
    from app.ui import select_library_document

    ids = [d.doc_id for d in library.list()]
    library.remove(ids[0])
    first = gr.SelectData(None, {'index': [0, 0], 'value': None})
    second = gr.SelectData(None, {'index': [1, 0], 'value': None})
    assert select_library_document(library, ids, first)['value'] is None
    assert select_library_document(library, ids, second)['value'] == ids[1]


def test_duplicate_filenames_keep_distinct_document_ids(library, tmp_path):
    from app.ui import select_library_document

    for number in range(2):
        source = tmp_path / str(number) / 'same.pdf'
        source.parent.mkdir()
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), f'Distinct document {number}')
            pdf.save(source)
        add_file(library, str(source))
    documents = library.list()
    ids = [d.doc_id for d in documents]
    duplicates = [(i, d.doc_id) for i, d in enumerate(documents) if d.filename == 'same.pdf']
    assert len(duplicates) == 2
    for row, doc_id in duplicates:
        event = gr.SelectData(None, {'index': [row, 0], 'value': 'same.pdf'})
        assert select_library_document(library, ids, event)['value'] == doc_id


def test_empty_snapshot_does_not_fall_back_to_another_listing(tmp_path, monkeypatch):
    from app.ui import TABLE_COLUMNS, _library_updates

    store = DocumentStore(root=tmp_path / 'empty')
    calls = []

    def list_once():
        calls.append(True)
        assert len(calls) == 1, 'Empty captured snapshots must not trigger another read'
        return []

    monkeypatch.setattr(store, 'list', list_once)
    frame, dropdown, _, row_ids = _library_updates(store, 'Ready.')
    assert frame.empty
    assert list(frame.columns) == TABLE_COLUMNS
    assert dropdown['choices'] == []
    assert dropdown['value'] is None
    assert row_ids == []
    assert len(calls) == 1


def test_load_uses_one_snapshot_for_duplicate_filename_identities(tmp_path, monkeypatch):
    """A second listing has identical names but opposite document identities."""
    store = DocumentStore(root=tmp_path / 'data')
    for number in (1, 2):
        source = tmp_path / str(number) / 'same.pdf'
        source.parent.mkdir()
        with pymupdf.open() as pdf:
            for page_number in range(number):
                pdf.new_page().insert_text((72, 72), f'Document {number}, page {page_number}')
            pdf.save(source)
        add_file(store, str(source))
    documents = store.list()
    assert documents[0].filename == documents[1].filename
    assert documents[0].doc_id != documents[1].doc_id
    expected_ids = [doc.doc_id for doc in documents]
    expected_pages = [len(json.loads((store.artifacts_dir / doc.doc_id / 'manifest.json')
                                    .read_text(encoding='utf-8'))['pages']) for doc in documents]
    demo = build_ui(store)
    session = SessionState(demo)
    load_id = next(number for number, fn in demo.fns.items() if fn.api_name == 'load_library')
    load_event = next(event for event in demo.config['dependencies'] if event['api_name'] == 'load_library')
    calls = []

    def changing_list():
        calls.append(True)
        return list(documents) if len(calls) == 1 else list(reversed(documents))

    with monkeypatch.context() as patch:
        patch.setattr(store, 'list', changing_list)
        loaded = asyncio.run(demo.process_api(load_id, [], state=session))
    frame, dropdown = loaded['data'][:2]
    assert [row[2] for row in frame['data']] == expected_pages
    assert session[load_event['outputs'][-1]] == expected_ids
    assert [doc_id for _, doc_id in dropdown['choices']] == expected_ids
    assert len(calls) == 1

    # Row zero must still select/remove the ID whose metadata is displayed there.
    from app.ui import select_library_document, remove_selected
    event = gr.SelectData(None, {'index': [0, 0], 'value': 'same.pdf'})
    selected = select_library_document(store, session[load_event['outputs'][-1]], event)
    assert selected['value'] == expected_ids[0]
    remove_selected(store, selected['value'])
    assert [doc.doc_id for doc in store.list()] == expected_ids[1:]


@pytest.mark.parametrize('action', ['add_file', 'remove_selected'])
def test_mutation_refresh_rebuilds_one_snapshot_not_returned_views(library, monkeypatch, action):
    """Legacy handler views cannot supply mismatched identity/order to the UI."""
    import app.ui as ui

    documents = library.list()
    frame = ui._library_df(library)
    expected_rows = frame.values.tolist()
    stale_frame = frame.iloc[::-1].reset_index(drop=True)
    wrong_choices = list(reversed(ui._choices(library)))
    demo = build_ui(library)
    session = SessionState(demo)
    event_id = next(number for number, fn in demo.fns.items() if fn.api_name == action)
    event = next(d for d in demo.config['dependencies'] if d['api_name'] == action)
    calls = []

    def list_once():
        calls.append(True)
        assert len(calls) == 1
        return list(documents)

    monkeypatch.setattr(ui, action, lambda *args: (stale_frame, wrong_choices, 'Operation complete.'))
    monkeypatch.setattr(library, 'list', list_once)
    inputs = [None] if action == 'add_file' else [documents[0].doc_id]
    result = asyncio.run(demo.process_api(event_id, inputs, state=session))
    expected_ids = [doc.doc_id for doc in documents]
    assert result['data'][0]['data'] == expected_rows
    assert [value for _, value in result['data'][1]['choices']] == expected_ids
    assert session[event['outputs'][-1]] == expected_ids
    assert result['data'][2] == 'Operation complete.'
    assert len(calls) == 1
