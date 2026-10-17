"""Unit tests for the endpoint-configuration gates (offline, no network).

Regression for the visual e2e skip gate: endpoint URL + API key without a
nonempty model name must count as unconfigured, and template placeholder
model values must not trigger live calls.
"""

import pytest

import app.config as config

pytestmark = pytest.mark.unit

REAL_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def _set_visual(monkeypatch, *, url="http://class.test:9003/v1", key="k", model=REAL_MODEL):
    monkeypatch.setattr(config, "VISUAL_EMBED_BASE_URL", url)
    monkeypatch.setattr(config, "VISUAL_EMBED_API_KEY", key)
    monkeypatch.setattr(config, "VISUAL_EMBED_MODEL", model)


def test_visual_endpoint_configured_when_complete(monkeypatch):
    _set_visual(monkeypatch)
    assert config.visual_embed_endpoint_configured()


def test_visual_endpoint_unconfigured_without_api_key(monkeypatch):
    _set_visual(monkeypatch, key="")
    assert not config.visual_embed_endpoint_configured()


def test_visual_endpoint_unconfigured_without_base_url(monkeypatch):
    _set_visual(monkeypatch, url="")
    assert not config.visual_embed_endpoint_configured()


def test_visual_endpoint_unconfigured_with_empty_model(monkeypatch):
    # Regression: endpoint + key valid but model empty must fail the gate,
    # otherwise e2e would attempt live calls instead of skipping.
    _set_visual(monkeypatch, model="")
    assert not config.visual_embed_endpoint_configured()


def test_visual_endpoint_unconfigured_with_blank_model(monkeypatch):
    _set_visual(monkeypatch, model="   ")
    assert not config.visual_embed_endpoint_configured()


def test_visual_endpoint_unconfigured_with_dummy_model(monkeypatch):
    _set_visual(monkeypatch, model="dummy-visual-model")
    assert not config.visual_embed_endpoint_configured()


def test_visual_endpoint_unconfigured_with_placeholder_model(monkeypatch):
    _set_visual(monkeypatch, model="replace-with-visual-model")
    assert not config.visual_embed_endpoint_configured()
