import pytest


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "model_loaded" in data
    assert "version" in data
    # device info
    assert "device" in data
    device = data["device"]
    assert device["device"] in ("cpu", "cuda", "unknown")
    assert isinstance(device["cuda_available"], bool)
    assert device["device_config"] in ("auto", "cpu", "cuda")


@pytest.mark.anyio
async def test_translate_single(client):
    resp = await client.post("/translate", json={"text": "Que voulez-vous donc?"})
    if resp.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    assert resp.status_code == 200
    data = resp.json()
    assert data["original"] == "Que voulez-vous donc?"
    assert isinstance(data["translated"], str)
    assert len(data["translated"]) > 0
    assert "model_version" in data


@pytest.mark.anyio
async def test_translate_empty_text(client):
    resp = await client.post("/translate", json={"text": ""})
    assert resp.status_code == 422  # validation error: min_length=1


@pytest.mark.anyio
async def test_translate_batch(client):
    texts = [
        "Que voulez-vous donc?",
        "Il estoit une fois un homme.",
        "Je ne sçay pas.",
    ]
    resp = await client.post("/translate/batch", json={"texts": texts, "batch_size": 2})
    if resp.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert len(data["translations"]) == 3
    assert data["processing_time"] > 0


@pytest.mark.anyio
async def test_translate_batch_empty_list(client):
    resp = await client.post("/translate/batch", json={"texts": []})
    assert resp.status_code == 422  # validation error: min_length=1


@pytest.mark.anyio
async def test_translate_special_characters(client):
    resp = await client.post("/translate", json={"text": "L'amour & la vertu, c'est-à-dire..."})
    if resp.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["translated"], str)


@pytest.mark.anyio
async def test_metrics_endpoint(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "translation_requests_total" in resp.text


@pytest.mark.anyio
async def test_request_id_header(client):
    resp = await client.get("/health", headers={"X-Request-ID": "test-123"})
    assert resp.headers.get("X-Request-ID") == "test-123"


@pytest.mark.anyio
async def test_translate_stream(client):
    texts = ["Que voulez-vous donc?", "Il estoit une fois."]
    resp = await client.post("/translate/stream", json={"texts": texts})
    if resp.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
