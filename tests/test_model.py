import pytest

from app.translator import FairseqTranslator


@pytest.fixture(scope="module")
def model():
    t = FairseqTranslator()
    try:
        t.load()
    except Exception:
        pytest.skip("Model files not available in test environment")
    return t


def test_model_is_loaded(model):
    assert model.is_loaded is True


def test_translate_single_sentence(model):
    result = model.translate("Que voulez-vous donc?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_translate_empty_string(model):
    result = model.translate("")
    assert result == ""


def test_translate_whitespace_normalization(model):
    result = model.translate("  Que   voulez-vous   donc?  ")
    assert isinstance(result, str)
    assert len(result) > 0


def test_translate_batch(model):
    texts = [
        "Que voulez-vous donc?",
        "Il estoit une fois un homme.",
        "Je ne sçay pas.",
    ]
    results = model.translate_batch(texts, batch_size=2)
    assert len(results) == 3
    assert all(isinstance(r, str) and len(r) > 0 for r in results)


def test_translate_long_text(model):
    long_text = "Je suis un homme de bien. " * 30
    result = model.translate(long_text)
    assert isinstance(result, str)


def test_singleton_pattern():
    a = FairseqTranslator()
    b = FairseqTranslator()
    assert a is b


def test_device_info(model):
    info = model.device_info
    assert info["device"] in ("cpu", "cuda")
    assert isinstance(info["cuda_available"], bool)
    assert info["device_config"] in ("auto", "cpu", "cuda")
    if info["cuda_available"] and info["device"] == "cuda":
        assert "gpu_name" in info
        assert info["gpu_memory_mb"] > 0


def test_device_matches_config(model):
    import torch
    from app import config

    if config.DEVICE == "cpu":
        assert model.device == "cpu"
    elif config.DEVICE == "cuda":
        assert model.device == "cuda"
    else:  # auto
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert model.device == expected
