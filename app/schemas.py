from typing import List, Optional

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text in early modern French")


class TranslateResponse(BaseModel):
    original: str
    translated: str
    model_version: str


class BatchTranslateRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts to translate")
    batch_size: int = Field(
        default=32,
        ge=1,
        description=(
            "Kept for compatibility; batches are sized by the token budget. "
            "1 selects the line-by-line reference path."
        ),
    )


class BatchTranslateResponse(BaseModel):
    translations: List[str]
    processing_time: float
    count: int


class DeviceInfo(BaseModel):
    device: str
    cuda_available: bool
    device_config: str
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[int] = None


class InferenceInfo(BaseModel):
    token_budget: Optional[int] = None
    token_budget_mode: Optional[str] = None
    oom_cap: Optional[int] = None
    fp16: Optional[bool] = None
    beam_size: Optional[int] = None
    coalesce_requests: Optional[bool] = None
    split_over_tokens: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    device: DeviceInfo
    inference: Optional[InferenceInfo] = None


class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None
