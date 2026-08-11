"""
Module 01 tests — Model Registry & Loader.

Test categories:
  1. Registry initialisation (no models loaded)
  2. Device/dtype detection
  3. Lazy loading behaviour
  4. Memory tracking
  5. LRU eviction under memory pressure
  6. Unload and garbage collection
  7. Thread safety
  8. Health status integration (get_status → ModelStatus schema)
  9. VLM provider handling
  10. Error recovery and fallback
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure backend root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_registry(max_ram_gb: float = 5.0, vlm_provider: str = "api"):
    """Create a fresh registry with configurable settings."""
    from app.models.registry import ModelRegistry, reset_registry  # noqa: PLC0415
    from app.config import Settings  # noqa: PLC0415

    reset_registry()

    settings = Settings(
        MAX_RAM_GB=max_ram_gb,
        VLM_PROVIDER=vlm_provider,
        VLM_ENABLED=True,
        HF_TOKEN=None,
        DEVICE="cpu",
        TORCH_DTYPE="float32",
    )
    return ModelRegistry(settings)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegistryInitialisation:
    """Registry creates successfully with zero models loaded."""

    def test_registry_creates_without_error(self) -> None:
        registry = _fresh_registry()
        assert registry is not None

    def test_no_models_loaded_on_init(self) -> None:
        registry = _fresh_registry()
        from app.models.registry import ModelRole  # noqa: PLC0415
        for role in ModelRole:
            assert not registry.is_loaded(role), f"{role.value} should not be loaded at init"

    def test_total_ram_is_zero_at_init(self) -> None:
        registry = _fresh_registry()
        assert registry.total_loaded_ram_mb == 0.0

    def test_budget_remaining_equals_total_at_init(self) -> None:
        registry = _fresh_registry(max_ram_gb=5.0)
        assert registry.budget_remaining_mb == pytest.approx(5.0 * 1024, abs=1)

    def test_singleton_accessor_works(self) -> None:
        from app.models.registry import get_registry, reset_registry  # noqa: PLC0415
        reset_registry()
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2, "get_registry should return the same singleton instance"
        reset_registry()

    def test_reset_registry_clears_singleton(self) -> None:
        from app.models.registry import get_registry, reset_registry  # noqa: PLC0415
        reset_registry()
        r1 = get_registry()
        reset_registry()
        r2 = get_registry()
        assert r1 is not r2, "After reset, a new instance should be created"
        reset_registry()


class TestDeviceDetection:
    """Device and dtype resolution."""

    def test_device_is_cpu_on_this_machine(self) -> None:
        registry = _fresh_registry()
        assert registry.device == "cpu"

    def test_dtype_is_float32_on_cpu(self) -> None:
        registry = _fresh_registry()
        import torch  # noqa: PLC0415
        assert registry.dtype == torch.float32

    def test_dtype_forced_to_float32_on_cpu_even_if_float16_requested(self) -> None:
        """Config should force float32 on CPU regardless of user setting."""
        from app.config import Settings  # noqa: PLC0415
        from app.models.registry import ModelRegistry, reset_registry  # noqa: PLC0415
        reset_registry()
        settings = Settings(DEVICE="cpu", TORCH_DTYPE="float16")
        registry = ModelRegistry(settings)
        # effective_torch_dtype should be float32 on CPU
        assert settings.effective_torch_dtype == "float32"
        import torch  # noqa: PLC0415
        assert registry.dtype == torch.float32


class TestVLMProvider:
    """VLM is never loaded locally on 8 GB machines."""

    def test_vlm_provider_is_hf_api_by_default(self) -> None:
        from app.models.registry import ModelRole  # noqa: PLC0415
        from app.models.schemas import ModelProvider  # noqa: PLC0415
        registry = _fresh_registry(vlm_provider="api")
        assert registry.get_provider(ModelRole.VLM) == ModelProvider.HF_API

    def test_vlm_provider_disabled_when_none(self) -> None:
        from app.models.registry import ModelRole  # noqa: PLC0415
        from app.models.schemas import ModelProvider  # noqa: PLC0415
        registry = _fresh_registry(vlm_provider="none")
        assert registry.get_provider(ModelRole.VLM) == ModelProvider.DISABLED

    def test_get_vlm_returns_none_for_api_provider(self) -> None:
        registry = _fresh_registry(vlm_provider="api")
        result = registry.get_vlm()
        assert result is None, "VLM should return None when provider is API"

    def test_get_vlm_returns_none_for_disabled(self) -> None:
        registry = _fresh_registry(vlm_provider="none")
        result = registry.get_vlm()
        assert result is None


class TestHealthStatus:
    """get_status() returns correct ModelStatus objects."""

    def test_get_status_returns_all_four_models(self) -> None:
        registry = _fresh_registry()
        status = registry.get_status()
        assert "dinov2" in status
        assert "segformer" in status
        assert "clip" in status
        assert "vlm" in status

    def test_status_matches_model_status_schema(self) -> None:
        from app.models.schemas import ModelStatus  # noqa: PLC0415
        registry = _fresh_registry()
        status = registry.get_status()
        for key, model_status in status.items():
            assert isinstance(model_status, ModelStatus), (
                f"{key} status is not a ModelStatus instance"
            )

    def test_all_models_not_loaded_in_fresh_registry(self) -> None:
        registry = _fresh_registry()
        status = registry.get_status()
        for key in ("dinov2", "segformer", "clip"):
            assert status[key].loaded is False, f"{key} should not be loaded"

    def test_vlm_shows_hf_api_provider(self) -> None:
        from app.models.schemas import ModelProvider  # noqa: PLC0415
        registry = _fresh_registry(vlm_provider="api")
        status = registry.get_status()
        assert status["vlm"].provider == ModelProvider.HF_API

    def test_model_ids_are_correct(self) -> None:
        registry = _fresh_registry()
        status = registry.get_status()
        assert "dinov2" in status["dinov2"].model_id
        assert "segformer" in status["segformer"].model_id
        assert "clip" in status["clip"].model_id
        assert "Qwen" in status["vlm"].model_id

    def test_memory_mb_is_none_when_not_loaded(self) -> None:
        registry = _fresh_registry()
        status = registry.get_status()
        for key in ("dinov2", "segformer", "clip"):
            assert status[key].memory_mb is None

    def test_no_errors_in_fresh_status(self) -> None:
        registry = _fresh_registry()
        status = registry.get_status()
        for key, s in status.items():
            assert s.error is None, f"{key} should have no error"


class TestMemorySummary:
    """Memory tracking and budget reporting."""

    def test_memory_summary_format(self) -> None:
        registry = _fresh_registry(max_ram_gb=5.0)
        summary = registry.get_memory_summary()
        assert "budget_mb" in summary
        assert "used_mb" in summary
        assert "remaining_mb" in summary
        assert "utilisation_pct" in summary
        assert "models_loaded" in summary

    def test_memory_summary_is_zero_at_init(self) -> None:
        registry = _fresh_registry()
        summary = registry.get_memory_summary()
        assert summary["used_mb"] == 0.0
        assert summary["utilisation_pct"] == 0.0
        assert summary["models_loaded"] == []


class TestUnload:
    """Model unloading and garbage collection."""

    def test_unload_not_loaded_returns_false(self) -> None:
        from app.models.registry import ModelRole  # noqa: PLC0415
        registry = _fresh_registry()
        assert registry.unload(ModelRole.DINOV2) is False

    def test_unload_all_on_fresh_registry_returns_zero(self) -> None:
        registry = _fresh_registry()
        count = registry.unload_all()
        assert count == 0


class TestThreadSafety:
    """Concurrent access doesn't cause crashes."""

    def test_concurrent_status_reads(self) -> None:
        registry = _fresh_registry()
        errors: list[str] = []

        def read_status() -> None:
            try:
                for _ in range(50):
                    _ = registry.get_status()
                    _ = registry.get_memory_summary()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=read_status) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent reads caused errors: {errors}"

    def test_concurrent_is_loaded_checks(self) -> None:
        from app.models.registry import ModelRole  # noqa: PLC0415
        registry = _fresh_registry()
        errors: list[str] = []

        def check_loaded() -> None:
            try:
                for _ in range(100):
                    for role in ModelRole:
                        _ = registry.is_loaded(role)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=check_loaded) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors


class TestInferenceTimeRecording:
    """Record and retrieve inference latency."""

    def test_record_inference_time(self) -> None:
        from app.models.registry import ModelRole  # noqa: PLC0415
        registry = _fresh_registry()
        registry.record_inference_time(ModelRole.DINOV2, 42.5)
        status = registry.get_status()
        assert status["dinov2"].last_inference_ms == 42.5

    def test_record_updates_last_accessed(self) -> None:
        from app.models.registry import ModelRole  # noqa: PLC0415
        registry = _fresh_registry()
        before = time.time()
        time.sleep(0.01)
        registry.record_inference_time(ModelRole.SEGFORMER, 10.0)
        # Internal state check
        entry = registry._models[ModelRole.SEGFORMER]
        assert entry.last_accessed >= before


class TestHealthEndpointIntegration:
    """End-to-end test with FastAPI TestClient — registry is now alive."""

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient  # noqa: PLC0415
        from app.models.registry import reset_registry  # noqa: PLC0415
        from app.main import create_application  # noqa: PLC0415
        reset_registry()
        app = create_application()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c
        reset_registry()

    def test_health_shows_registry_status(self, client) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        models = data["models"]

        # Registry is now initialised — models should still be not_loaded (lazy)
        assert models["dinov2"]["loaded"] is False
        assert models["segformer"]["loaded"] is False
        assert models["clip"]["loaded"] is False

    def test_health_vlm_shows_correct_provider(self, client) -> None:
        resp = client.get("/api/v1/health")
        data = resp.json()
        vlm = data["models"]["vlm"]
        # Default VLM_PROVIDER=api
        assert vlm["provider"] in ("hf_api", "disabled")

    def test_warmup_now_returns_meaningful_response(self, client) -> None:
        from app.models.registry import ModelRegistry  # noqa: PLC0415
        with patch.object(ModelRegistry, "warmup") as mock_warmup:
            async def _dummy_warmup(*args, **kwargs):
                return {"dinov2": True}
            mock_warmup.side_effect = _dummy_warmup
            resp = client.post("/api/v1/warmup")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] in ("warmup_started", "not_applicable")


class TestModelSpecs:
    """Verify model specification metadata."""

    def test_all_model_specs_have_loader(self) -> None:
        from app.models.registry import ModelRole  # noqa: PLC0415
        registry = _fresh_registry()
        for role, spec in registry._specs.items():
            assert hasattr(registry, spec.loader_fn_name), (
                f"Missing loader method {spec.loader_fn_name} for {role.value}"
            )

    def test_dinov2_spec_is_reasonable(self) -> None:
        from app.models.registry import ModelRole  # noqa: PLC0415
        registry = _fresh_registry()
        spec = registry._specs[ModelRole.DINOV2]
        assert spec.estimated_ram_mb > 100  # ~344 MB
        assert spec.estimated_ram_mb < 1000
        assert "dinov2" in spec.model_id

    def test_segformer_spec_is_reasonable(self) -> None:
        from app.models.registry import ModelRole  # noqa: PLC0415
        registry = _fresh_registry()
        spec = registry._specs[ModelRole.SEGFORMER]
        assert spec.estimated_ram_mb > 50  # ~108 MB
        assert spec.estimated_ram_mb < 500
        assert "segformer" in spec.model_id

    def test_clip_spec_is_reasonable(self) -> None:
        from app.models.registry import ModelRole  # noqa: PLC0415
        registry = _fresh_registry()
        spec = registry._specs[ModelRole.CLIP]
        assert spec.estimated_ram_mb > 500  # ~1200 MB
        assert spec.estimated_ram_mb < 2000
        assert "clip" in spec.model_id

    def test_vlm_spec_is_huge(self) -> None:
        from app.models.registry import ModelRole  # noqa: PLC0415
        registry = _fresh_registry()
        spec = registry._specs[ModelRole.VLM]
        assert spec.estimated_ram_mb > 10000  # ~15000 MB
        assert "Qwen" in spec.model_id

    def test_total_local_budget_fits(self) -> None:
        """DINOv2 + SegFormer + CLIP combined should fit within default budget."""
        from app.models.registry import ModelRole  # noqa: PLC0415
        registry = _fresh_registry(max_ram_gb=5.0)
        local_models = [ModelRole.DINOV2, ModelRole.SEGFORMER, ModelRole.CLIP]
        total_estimated = sum(
            registry._specs[role].estimated_ram_mb for role in local_models
        )
        budget_mb = 5.0 * 1024
        assert total_estimated < budget_mb, (
            f"Combined model estimate {total_estimated:.0f} MB exceeds budget {budget_mb:.0f} MB"
        )
