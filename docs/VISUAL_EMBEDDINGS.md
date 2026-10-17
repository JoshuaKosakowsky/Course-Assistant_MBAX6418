# Visual embeddings (class endpoint, port 9003)

The visual index embeds page/slide images with the class visual-embedding
endpoint (`BASE_URL_VISUAL_EMBED`, default `http://dobolyi.com:9003/v1`,
model `VISUAL_EMBED_MODEL`, key `CLASS_API_KEY`) through
`app.visual_embeddings.ClassEmbeddingClient`.

## Verified request format

Images are sent as the documented **multimodal `messages` shape**, not as a
data-URI string under `input`:

```json
{
  "model": "Qwen/Qwen3-VL-Embedding-2B",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ]
}
```

Why: a plain data-URI `input` string is tokenized as text. A realistic
396x512 page render produced HTTP 400 ("your prompt contains at least 8193
input tokens", exceeding the model's 8192-token context); the identical
image through the `messages`/`image_url` shape returned HTTP 200 with a
2048-dimension vector. The client therefore sends only the multimodal
shape for images.

Text queries (`embed_text_visual`) continue to use the same `input` list
shape; both use the visual model/endpoint so image and text vectors remain
comparable.

## Image sizing

Each embedding request uses a copy downscaled to at most 512px on the
longest edge (`resize_for_embedding`). The caller's original file is never
modified; stored original paths remain the citation authority.

## Tests

Unit (offline, default suite):

```bash
python -m pytest tests/unit/test_visual_embeddings.py -q
```

These assert the exact outgoing request body: modal model selection,
multimodal `messages`/`image_url` shape, no `input` key for images, 512px
downscaling of the embedded copy, original preservation, reserved-port
rejection, and missing-config errors.

Live (opt-in, real class endpoint; needs `.env` keys):

```bash
python -m pytest tests/e2e/test_visual_index_e2e.py -m e2e -q
```

The e2e uses a realistic 512x512 synthetic page image — not a tiny
fixture that could succeed under either encoding — and covers embedding,
indexing, and image-to-image query roundtrip. It skips automatically when
the endpoint is not configured and is excluded from the default/CI run.

## Limits

- Service availability and exact model IDs are live facts; reconfirm with
  authenticated `/models` output when behavior changes.
- Dimension equality does not prove semantic quality or cross-model
  compatibility; the same model produces image and text vectors.
- The client only encodes images for the request; it does not validate
  file ownership. Trusted callers supply manifest-vetted original paths.
- Port 9000 remains reserved and is rejected by the client.
- Errors are generic (`Visual embedding service request failed.`) with
  suppressed detail; credentials never appear in requests or messages.
