from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .. import config
from ..schemas import HealthResponse
from ..translator import translator

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if translator.is_loaded else "degraded",
        model_loaded=translator.is_loaded,
        version=config.MODEL_VERSION,
        device=translator.device_info,
        inference=translator.inference_info,
    )


@router.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
