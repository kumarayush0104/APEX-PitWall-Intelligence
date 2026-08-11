"""
Health check endpoints.

GET /health         — Quick liveness probe (for load balancers, Docker health checks).
GET /api/v1/health  — Detailed system diagnostics (for frontend status panel).
POST /api/v1/warmup — Trigger lazy model initialisation ahead of first analysis.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from loguru import logger

from app.api.dependencies import SettingsDep
from app.config import get_settings
from app.models.schemas import (
    HealthResponse,
    ModelProvider,
    ModelStatus,
    ServiceStatus,
    SystemInfo,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_model_statuses(app_state: Any) -> dict[str, ModelStatus]:
    """
    Construct per-model status dict.

    If the model registry exists and exposes a get_status() method
    (implemented in Module 01), use live data. Otherwise return the
    pre-load 'not_loaded' state derived from config.
    """
    settings = get_settings()

    # Default statuses — shown before Module 01 is integrated
    statuses: dict[str, ModelStatus] = {
        "dinov2": ModelStatus(
            loaded=False,
            provider=ModelProvider.NOT_LOADED,
            model_id=settings.DINOV2_MODEL_ID,
        ),
        "segformer": ModelStatus(
            loaded=False,
            provider=ModelProvider.NOT_LOADED,
            model_id=settings.SEGFORMER_MODEL_ID,
        ),
        "clip": ModelStatus(
            loaded=False,
            provider=ModelProvider.NOT_LOADED,
            model_id=settings.CLIP_MODEL_ID,
        ),
        "vlm": ModelStatus(
            loaded=False,
            provider=(
                ModelProvider.DISABLED
                if not settings.VLM_ENABLED
                else (
                    ModelProvider.HF_API
                    if settings.VLM_PROVIDER.value == "api"
                    else ModelProvider.NOT_LOADED
                )
            ),
            model_id=settings.VLM_MODEL_ID,
        ),
    }

    # Override with live data from the model registry (Module 01)
    registry = getattr(app_state, "model_registry", None)
    if registry is not None and callable(getattr(registry, "get_status", None)):
        try:
            live = registry.get_status()
            statuses.update(live)
        except Exception as exc:
            logger.warning("Model registry status unavailable: {}", exc)

    return statuses


def _build_service_statuses(app_state: Any) -> dict[str, ServiceStatus]:
    """Build per-service status dict."""
    settings = get_settings()

    statuses: dict[str, ServiceStatus] = {
        "session_store": ServiceStatus(
            status="ready",
            detail="In-memory (initialised on first request)",
        ),
        "weather": ServiceStatus(
            status="disabled" if not settings.WEATHER_ENABLED else "ready",
            detail=(
                None
                if not settings.WEATHER_ENABLED
                else "OpenWeatherMap One-Call API 3.0"
            ),
        ),
        "demo_mode": ServiceStatus(
            status="ready" if settings.DEMO_MODE_ENABLED else "disabled",
            detail=str(settings.DEMO_CLIPS_DIR) if settings.DEMO_MODE_ENABLED else None,
        ),
        "vlm": ServiceStatus(
            status=(
                "disabled"
                if not settings.VLM_ENABLED
                else ("ready" if settings.vlm_is_functional else "degraded")
            ),
            detail=(
                "Structured fallback active (HF_TOKEN not set)"
                if settings.VLM_ENABLED and not settings.vlm_is_functional
                else settings.VLM_PROVIDER.value
            ),
        ),
    }

    # Live session store check (Module 12)
    session_store = getattr(app_state, "session_store", None)
    if session_store is not None:
        session_count = getattr(session_store, "active_session_count", 0)
        statuses["session_store"] = ServiceStatus(
            status="ready",
            detail=f"{session_count} active sessions",
        )

    return statuses


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    summary="Liveness probe",
    description=(
        "Ultra-lightweight liveness check. Returns 200 immediately. "
        "Suitable for Docker HEALTHCHECK and load balancer probes."
    ),
    tags=["Health"],
)
async def liveness() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "healthy",
        "service": "pitwall-intelligence",
        "version": settings.APP_VERSION,
    }


@router.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Detailed system health",
    description=(
        "Full system diagnostics: hardware info, per-model load status, "
        "per-service health, and configuration summary. "
        "Used by the frontend status panel."
    ),
    tags=["Health"],
)
async def detailed_health(request: Request, settings: SettingsDep) -> HealthResponse:
    """Return comprehensive system health information."""
    app_state = request.app.state

    # Collect system information (no torch required)
    raw = settings.get_system_info()
    system = SystemInfo(
        python_version=raw["python_version"],
        torch_version=raw.get("torch_version"),
        platform=raw["platform"],
        platform_release=raw.get("platform_release", ""),
        cpu_count=raw["cpu_count"],
        resolved_device=raw["resolved_device"],
        cuda_available=raw["cuda_available"],
        cuda_device_name=raw.get("cuda_device_name"),
        cuda_device_memory_gb=raw.get("cuda_device_memory_gb"),
        ram_total_gb=raw.get("ram_total_gb"),
        ram_available_gb=raw.get("ram_available_gb"),
    )

    models = _build_model_statuses(app_state)
    services = _build_service_statuses(app_state)

    # Overall status: healthy if all non-disabled services are ready
    non_disabled = [s for s in services.values() if s.status != "disabled"]
    if all(s.status == "ready" for s in non_disabled):
        overall = "healthy"
    elif any(s.status == "error" for s in non_disabled):
        overall = "unhealthy"
    else:
        overall = "degraded"

    # Config summary — never expose tokens
    config_summary = settings.summary()
    config_summary["uptime_seconds"] = round(
        time.time() - getattr(app_state, "startup_time", time.time()), 1
    )

    logger.debug("Health check | overall={}", overall)

    return HealthResponse(
        status=overall,
        system=system,
        models=models,
        services=services,
        config=config_summary,
    )


@router.post(
    "/api/v1/warmup",
    summary="Trigger model pre-loading",
    description=(
        "Instruct the server to pre-load models into memory ahead of the first "
        "analysis request. Eliminates cold-start latency during demos. "
        "Returns immediately; monitor /api/v1/health for load progress."
    ),
    tags=["Health"],
)
async def warmup(request: Request) -> dict[str, str]:
    """
    Trigger lazy model initialisation.

    The actual loading is delegated to the model registry (Module 01).
    Before Module 01 is integrated, this endpoint is a no-op that returns
    an informational message.
    """
    registry = getattr(request.app.state, "model_registry", None)
    if registry is not None and callable(getattr(registry, "warmup", None)):
        # Module 01 provides this — triggers background pre-loading
        import asyncio  # noqa: PLC0415
        task = asyncio.create_task(registry.warmup())  # type: ignore[arg-type]
        task.add_done_callback(
            lambda t: t.exception() and logger.error("Background warmup failed: {}", t.exception())
        )
        return {"status": "warmup_started", "message": "Models are loading in the background."}

    return {
        "status": "not_applicable",
        "message": (
            "Model registry not yet integrated (Module 01). "
            "Models will load lazily on first analysis request."
        ),
    }
