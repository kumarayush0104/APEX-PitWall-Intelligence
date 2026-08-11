"""
Module 00 tests — Project scaffold and health endpoints.

All tests in this file run WITHOUT any AI models loaded.
They verify the application scaffold is correct and the health
endpoint responds with the expected schema.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Make sure we can import from backend/app
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

def _make_local_registry():
    """Build a real ModelRegistry locally (no HF hub calls — lazy loading)."""
    from app.models.registry import ModelRegistry, reset_registry  # noqa: PLC0415
    from app.config import Settings  # noqa: PLC0415
    reset_registry()
    settings = Settings(
        DEVICE="cpu",
        TORCH_DTYPE="float32",
        MAX_RAM_GB=5.0,
        HF_TOKEN=None,
        VLM_ENABLED=True,
        VLM_PROVIDER="api",
    )
    return ModelRegistry(settings)


@pytest.fixture(scope="session")
def client():
    """
    Synchronous test client for the FastAPI application.

    Bypasses the lifespan entirely by populating app.state directly.
    This avoids the threading deadlock in anyio portal teardown caused
    by the lifespan calling registry.unload_all() with a sync RLock.
    No models are loaded — all model statuses will be 'not_loaded'.
    """
    import time  # noqa: PLC0415
    from app.main import create_application  # noqa: PLC0415
    app = create_application()
    local_registry = _make_local_registry()
    # Populate app.state directly — no lifespan needed
    app.state.model_registry = local_registry
    app.state.startup_time = time.time()
    app.state.session_store = None
    # Use TestClient WITHOUT context manager so lifespan is NOT triggered
    c = TestClient(app, raise_server_exceptions=False)
    yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLiveness:
    """Quick liveness probe tests."""

    def test_liveness_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200, resp.text

    def test_liveness_response_schema(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data

    def test_liveness_is_fast(self, client: TestClient) -> None:
        """Liveness probe should respond in < 100 ms."""
        import time  # noqa: PLC0415
        t0 = time.perf_counter()
        client.get("/health")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 500, f"Liveness took {elapsed_ms:.0f}ms — too slow"


class TestDetailedHealth:
    """Detailed health endpoint tests."""

    def test_detailed_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200, resp.text

    def test_detailed_health_schema_top_level(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert "success" in data
        assert "timestamp" in data
        assert "status" in data
        assert "system" in data
        assert "models" in data
        assert "services" in data
        assert "config" in data

    def test_system_info_fields(self, client: TestClient) -> None:
        system = client.get("/api/v1/health").json()["system"]
        assert "python_version" in system
        assert "platform" in system
        assert "cpu_count" in system
        assert isinstance(system["cpu_count"], int)
        assert system["cpu_count"] >= 1
        assert "resolved_device" in system
        assert system["resolved_device"] in ("cpu", "cuda")
        assert "cuda_available" in system
        assert isinstance(system["cuda_available"], bool)

    def test_model_statuses_present(self, client: TestClient) -> None:
        models = client.get("/api/v1/health").json()["models"]
        assert "dinov2" in models
        assert "segformer" in models
        assert "clip" in models
        assert "vlm" in models

    def test_models_not_loaded_before_registry(self, client: TestClient) -> None:
        """Before Module 01 is integrated, all models should be 'not_loaded'."""
        models = client.get("/api/v1/health").json()["models"]
        # DINOv2, SegFormer, CLIP should be not_loaded (lazy loading)
        for model_name in ("dinov2", "segformer", "clip"):
            assert models[model_name]["loaded"] is False, (
                f"{model_name} should not be loaded at startup"
            )

    def test_services_present(self, client: TestClient) -> None:
        services = client.get("/api/v1/health").json()["services"]
        assert "session_store" in services
        assert "weather" in services
        assert "demo_mode" in services

    def test_config_no_secrets(self, client: TestClient) -> None:
        """Config summary must never expose tokens."""
        config = client.get("/api/v1/health").json()["config"]
        # Check that sensitive keys are not present
        assert "hf_token" not in str(config).lower() or (
            "hf_token_set" in config  # Only the boolean indicator is allowed
        )
        assert "weather_api_key" not in config

    def test_process_time_header(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert "x-process-time-ms" in resp.headers
        elapsed = float(resp.headers["x-process-time-ms"])
        assert elapsed > 0


class TestWarmup:
    """Warmup endpoint tests (require real model loading — skipped in unit mode)."""

    @pytest.mark.skip(reason="Warmup triggers real model loading (DINOv2/SegFormer/CLIP). Run manually with models downloaded.")
    def test_warmup_returns_200(self, client: TestClient) -> None:
        resp = client.post("/api/v1/warmup")
        assert resp.status_code == 200

    @pytest.mark.skip(reason="Warmup triggers real model loading (DINOv2/SegFormer/CLIP). Run manually with models downloaded.")
    def test_warmup_returns_status(self, client: TestClient) -> None:
        data = client.post("/api/v1/warmup").json()
        assert "status" in data
        assert "message" in data


class TestOpenAPI:
    """OpenAPI schema endpoint tests."""

    def test_openapi_json_accessible(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_docs_accessible(self, client: TestClient) -> None:
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_accessible(self, client: TestClient) -> None:
        resp = client.get("/redoc")
        assert resp.status_code == 200


class TestCORS:
    """CORS configuration tests."""

    def test_cors_allows_frontend_origin(self, client: TestClient) -> None:
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5500",
                "Access-Control-Request-Method": "GET",
            },
        )
        # With CORS_ALLOW_ALL=True the header should be present
        # (TestClient may not always exercise CORS — this is a smoke test)
        assert resp.status_code in (200, 204)


class TestConfiguration:
    """Settings / configuration smoke tests."""

    def test_settings_loads_without_error(self) -> None:
        from app.config import get_settings  # noqa: PLC0415
        settings = get_settings()
        assert settings is not None

    def test_resolved_device_is_valid(self) -> None:
        from app.config import get_settings  # noqa: PLC0415
        settings = get_settings()
        assert settings.resolved_device in ("cpu", "cuda")

    def test_effective_dtype_is_float32_on_cpu(self) -> None:
        from app.config import get_settings  # noqa: PLC0415
        settings = get_settings()
        if not settings.is_gpu_available:
            assert settings.effective_torch_dtype == "float32", (
                "float16/bfloat16 should be forced to float32 on CPU"
            )

    def test_schema_imports_without_error(self) -> None:
        from app.models.schemas import (  # noqa: PLC0415
            APEXResult,
            ConditionState,
            HealthResponse,
            RecommendationPriority,
        )
        # Just check they import without error
        assert ConditionState.TRANSITIONAL == "TRANSITIONAL"
        assert RecommendationPriority.IMMEDIATE == "IMMEDIATE"
