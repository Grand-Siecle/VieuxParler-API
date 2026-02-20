import asyncio

from fastapi import FastAPI

from app import config
from app.middleware import RequestIdMiddleware
from app.routes import system, translation
from app.translator import translator

app = FastAPI(
    title="VieuxParler API",
    description="Translate early modern French (17th c.) to contemporary French.",
    version=config.APP_VERSION,
)

app.add_middleware(RequestIdMiddleware)
app.include_router(system.router)
app.include_router(translation.router)


@app.on_event("startup")
async def startup_load_model():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, translator.load)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        log_level=config.LOG_LEVEL,
    )
