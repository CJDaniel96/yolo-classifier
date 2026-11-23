"""Read-only health/status HTTP endpoints; inference stays off the event loop."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from .config import Config
from .inference import Predictor
from .service import InferenceBackend, Service


def create_app(config: Config,
               predictor_factory: Callable[[Config], InferenceBackend] = Predictor,
               retry_failed: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # No process may start scanning until model loading/validation succeeds.
        service = Service(config, lambda: predictor_factory(config))
        app.state.service = service
        if retry_failed:
            service.state.retry_failed(service.clock())
        service.start()
        try:
            yield
        finally:
            await asyncio.to_thread(service.close)

    app = FastAPI(title="AOI YOLO OBB Service", version="0.3.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        status = app.state.service.status()
        healthy = status["worker_alive"] and status["last_error"] is None
        return JSONResponse({"status": "ok" if healthy else "unhealthy", **status},
                            status_code=200 if healthy else 503)

    @app.get("/status")
    def status():
        return app.state.service.status()

    @app.get("/jobs")
    def jobs(limit: int = Query(default=50, ge=1, le=500)):
        records = app.state.service.state.recent(limit)
        return [{"id": j["id"], "xml": j["source"], "status": j["status"],
                 "attempts": j["attempts"], "error": j["error"], "updated": j["updated"],
                 "output": j["payload"].get("output"),
                 "archive_dir": j["payload"].get("archive_dir"),
                 "component_selection": j["payload"].get("component_selection"),
                 "images": j["payload"].get("images", {})}
                for j in records]

    return app
