"""
FastAPI dependency injection functions.

All resource acquisition (settings, model registry, session store) is centralised
here. Routes never import these resources directly — they receive them via
FastAPI's Depends() injection. This enables:
  - Clean testability (mock any dependency)
  - Lazy model loading (models not imported at route-module level)
  - Consistent error handling (503 before models are ready)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings, get_settings

if TYPE_CHECKING:
    from app.models.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def settings_dependency(request: Request) -> Settings:
    """
    Provide the Settings instance from app state.

    Using app.state rather than calling get_settings() directly ensures
    tests can override settings by replacing app.state.settings.
    """
    settings: Settings = getattr(request.app.state, "settings", None)
    if settings is None:
        # Fallback — should never happen after lifespan runs
        return get_settings()
    return settings


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

def model_registry_or_none(request: Request):
    """
    Return the model registry if initialised, otherwise None.

    Use for endpoints that can provide partial responses without models.
    """
    return getattr(request.app.state, "model_registry", None)


def require_model_registry(request: Request):
    """
    Return the model registry, raising HTTP 503 if not yet initialised.

    Use for endpoints that MUST have models available to function.
    The frontend should poll /api/v1/health and wait for models to report
    'loaded' before sending analysis requests.
    """
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Model registry not yet initialised.",
                "error_code": "MODELS_NOT_READY",
                "hint": (
                    "Models load lazily on first use. "
                    "Trigger initialisation by calling POST /api/v1/warmup, "
                    "or wait for the first analysis request."
                ),
            },
        )
    return registry


# ---------------------------------------------------------------------------
# Session Store
# ---------------------------------------------------------------------------

def get_session_store(request: Request):
    """
    Return the session store if initialised, otherwise None.

    The session store is initialised lazily in Module 12.
    Before Module 12 is integrated, this returns None.
    """
    return getattr(request.app.state, "session_store", None)


def require_session_store(request: Request):
    """Return the session store, raising HTTP 503 if not available."""
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Session store not initialised.",
                "error_code": "SESSION_STORE_NOT_READY",
            },
        )
    return store


# ---------------------------------------------------------------------------
# Type aliases — clean route signatures
# ---------------------------------------------------------------------------

SettingsDep       = Annotated[Settings, Depends(settings_dependency)]
OptionalRegistry  = Annotated["ModelRegistry | None", Depends(model_registry_or_none)]
RequiredRegistry  = Annotated["ModelRegistry", Depends(require_model_registry)]
OptionalSession   = Annotated[object | None, Depends(get_session_store)]
RequiredSession   = Annotated[object, Depends(require_session_store)]
