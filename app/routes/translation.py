import asyncio
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .. import config
from ..metrics import (
    BATCH_SIZE_PROCESSED,
    TRANSLATION_DURATION,
    TRANSLATION_REQUESTS,
    logger,
)
from ..schemas import (
    BatchTranslateRequest,
    BatchTranslateResponse,
    TranslateRequest,
    TranslateResponse,
)
from ..translator import translator

router = APIRouter(tags=["translation"])


@router.post("/translate", response_model=TranslateResponse)
async def translate_single(body: TranslateRequest, request: Request):
    if not translator.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    loop = asyncio.get_event_loop()
    start = time.perf_counter()

    try:
        translated = await asyncio.wait_for(
            loop.run_in_executor(None, translator.translate, body.text),
            timeout=config.TRANSLATE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        TRANSLATION_REQUESTS.labels(endpoint="/translate", status="timeout").inc()
        raise HTTPException(status_code=504, detail="Translation timed out")
    except Exception as exc:
        TRANSLATION_REQUESTS.labels(endpoint="/translate", status="error").inc()
        logger.exception("translation failed", extra={"request_id": request.state.request_id})
        raise HTTPException(status_code=500, detail=str(exc))

    duration = time.perf_counter() - start
    TRANSLATION_DURATION.labels(endpoint="/translate").observe(duration)
    TRANSLATION_REQUESTS.labels(endpoint="/translate", status="success").inc()

    logger.info(
        "translation complete",
        extra={
            "request_id": request.state.request_id,
            "endpoint": "/translate",
            "input_length": len(body.text),
            "output_length": len(translated),
            "inference_time_ms": round(duration * 1000, 2),
        },
    )

    return TranslateResponse(
        original=body.text,
        translated=translated,
        model_version=config.MODEL_VERSION,
    )


@router.post("/translate/batch", response_model=BatchTranslateResponse)
async def translate_batch(body: BatchTranslateRequest, request: Request):
    if not translator.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    loop = asyncio.get_event_loop()
    start = time.perf_counter()

    try:
        translations = await asyncio.wait_for(
            loop.run_in_executor(
                None, translator.translate_batch, body.texts, body.batch_size
            ),
            timeout=config.BATCH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        TRANSLATION_REQUESTS.labels(endpoint="/translate/batch", status="timeout").inc()
        raise HTTPException(status_code=504, detail="Batch translation timed out")
    except Exception as exc:
        TRANSLATION_REQUESTS.labels(endpoint="/translate/batch", status="error").inc()
        logger.exception("batch translation failed", extra={"request_id": request.state.request_id})
        raise HTTPException(status_code=500, detail=str(exc))

    duration = time.perf_counter() - start
    TRANSLATION_DURATION.labels(endpoint="/translate/batch").observe(duration)
    TRANSLATION_REQUESTS.labels(endpoint="/translate/batch", status="success").inc()
    BATCH_SIZE_PROCESSED.set(len(body.texts))

    logger.info(
        "batch translation complete",
        extra={
            "request_id": request.state.request_id,
            "endpoint": "/translate/batch",
            "count": len(body.texts),
            "inference_time_ms": round(duration * 1000, 2),
        },
    )

    return BatchTranslateResponse(
        translations=translations,
        processing_time=round(duration, 4),
        count=len(translations),
    )


@router.post("/translate/stream")
async def translate_stream(body: BatchTranslateRequest):
    if not translator.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    async def event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        for text in body.texts:
            try:
                translated = await loop.run_in_executor(
                    None, translator.translate, text
                )
                yield f"data: {translated}\n\n"
            except Exception as exc:
                yield f"event: error\ndata: {str(exc)}\n\n"
                break
        yield "event: done\ndata: stream complete\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
