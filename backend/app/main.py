"""
APEX PitWall Intelligence — FastAPI Application Entry Point.

Architecture decisions encoded here:
  - Factory function pattern: create_application() allows multiple instances for testing.
  - Lifespan context manager (FastAPI 0.95+): replaces deprecated @app.on_event.
  - Models are NOT loaded at startup — lazy loading via Model Registry (Module 01).
  - CORS is permissive by default; tighten CORS_ORIGINS in production.
  - All API routes are versioned under /api/v1/.
  - Gzip compression applied to responses > 1 KB.
  - Process-time header on every response for debugging.
  - Global exception handler prevents raw tracebacks reaching clients.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import get_settings
from app.models.schemas import ErrorResponse
from app.utils.logging_config import configure_logging


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan: startup → serve → shutdown.

    Startup sequence (in order):
      1. Configure logging
      2. Apply environment variables (HF cache dirs, tokens)
      3. Log system configuration
      4. Initialise app state placeholders
      5. Yield (server begins accepting requests)

    Shutdown sequence:
      6. Log uptime
      7. (Model memory released by Python GC)

    Models are NOT loaded here.  The Model Registry (Module 01) loads them
    lazily on the first request that needs them.
    """
    settings = get_settings()

    # 1. Logging must come first — everything after this can emit log records
    configure_logging(settings)

    logger.info("━" * 60)
    logger.info("  APEX PitWall Intelligence  v{}", settings.APP_VERSION)
    logger.info("━" * 60)

    # 2. Environment already applied by get_settings() → setup_environment()
    #    But log it for transparency.
    logger.info("HF cache → {}", settings.HF_CACHE_DIR)

    # 3. System diagnostics
    sys_info = settings.get_system_info()
    logger.info(
        "Hardware │ device={} │ cuda={} │ ram_total={} GB │ cpu_cores={}",
        sys_info["resolved_device"],
        sys_info["cuda_available"],
        sys_info.get("ram_total_gb", "?"),
        sys_info["cpu_count"],
    )
    if sys_info.get("cuda_device_name"):
        logger.info("GPU │ {}", sys_info["cuda_device_name"])
    else:
        logger.info("GPU │ Not available — running CPU-only inference")

    logger.info(
        "Config  │ device={} │ dtype={} │ max_ram={} GB │ vlm={} ({}) │ weather={}",
        settings.resolved_device,
        settings.effective_torch_dtype,
        settings.MAX_RAM_GB,
        settings.VLM_PROVIDER.value,
        "✓" if settings.vlm_is_functional else "structured fallback",
        "✓" if settings.WEATHER_ENABLED else "disabled",
    )

    if not settings.HF_TOKEN:
        logger.warning(
            "HF_TOKEN not set — VLM explanations will use structured fallback. "
            "Set HF_TOKEN=hf_... in backend/.env to enable LLM explanations."
        )

    # 4. Initialise app state
    app.state.settings = settings
    app.state.startup_time = time.time()
    app.state.session_store = None      # Populated by Module 12 on first request

    # 5. Initialise Model Registry (lazy — no models loaded yet, ~0 MB RAM)
    from app.models.registry import get_registry  # noqa: PLC0415
    registry = get_registry(settings)
    app.state.model_registry = registry
    logger.info(
        "Registry│ initialised | device={} | dtype={} | budget={:.1f} GB | 0 models loaded",
        registry.device,
        settings.effective_torch_dtype,
        settings.MAX_RAM_GB,
    )

    logger.info("Server  │ http://{}:{}", settings.HOST, settings.PORT)
    logger.info("API Docs│ http://{}:{}/docs", settings.HOST, settings.PORT)
    logger.info("Models  │ Lazy loading active — will load on first analysis request")
    logger.info("Demo    │ {}", "enabled" if settings.DEMO_MODE_ENABLED else "disabled")
    logger.info("━" * 60)
    logger.info("APEX is ready to accept requests")

    # ── Serve ──────────────────────────────────────────────────────────────
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────

    uptime = time.time() - app.state.startup_time

    # Unload all models to release RAM (safely with timeout)
    registry = getattr(app.state, "model_registry", None)
    if registry is not None:
        import asyncio  # noqa: PLC0415
        try:
            count = await asyncio.wait_for(asyncio.to_thread(registry.unload_all), timeout=5.0)
            logger.info("Unloaded {} model(s) during shutdown", count)
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Shutdown registry unload timed out or failed: {}", exc)

    logger.info("APEX shutting down after {:.1f} s uptime", uptime)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_application() -> FastAPI:
    """
    Build and return the configured FastAPI application.

    Using a factory function (rather than a module-level instance) allows
    the test suite to call create_application() to get fresh instances
    with overridden settings or state.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Middleware (applied in reverse order — last added = outermost)
    # ------------------------------------------------------------------

    # CORS — must be added before any other middleware that reads headers
    cors_origins = ["*"] if settings.CORS_ALLOW_ALL else settings.CORS_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=not settings.CORS_ALLOW_ALL,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time-Ms", "X-Session-Id"],
    )

    # Gzip — compress large JSON responses (analysis results can be 20–80 KB)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Process-time header for frontend latency monitoring
    @app.middleware("http")
    async def _process_time_header(request: Request, call_next):  # noqa: ANN001, ANN202
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    # ------------------------------------------------------------------
    # Global exception handlers
    # ------------------------------------------------------------------

    @app.exception_handler(Exception)
    async def _global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all for unhandled exceptions.
        Never exposes internal detail to clients in production.
        """
        logger.exception(
            "Unhandled exception on {} {}",
            request.method,
            str(request.url),
        )
        app_settings = getattr(request.app.state, "settings", settings)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="An unexpected error occurred.",
                error_code="INTERNAL_SERVER_ERROR",
                detail=str(exc) if app_settings.DEBUG else None,
            ).model_dump(),
        )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    # Only health routes are registered here.
    # Analysis, session, WebSocket, and demo routes are registered in
    # Modules 13–14 when those modules are integrated.

    from app.api.routes.health import router as health_router  # noqa: PLC0415
    from app.api.routes.analysis import router as analysis_router  # noqa: PLC0415
    app.include_router(health_router)
    app.include_router(analysis_router)

    return app


# ---------------------------------------------------------------------------
# WSGI/ASGI entry point
# ---------------------------------------------------------------------------

# The application instance used by uvicorn.
# uvicorn command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
app = create_application()
