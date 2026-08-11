"""
Tests for Module 08: API Analysis Routes.

Covers:
  - POST /api/v1/analyze with image file upload
  - GET  /api/v1/history
  - Error handling when no image input is provided

Uses FastAPI dependency_overrides to inject a mock registry so no models
are downloaded and no lifespan HF calls are made.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import require_model_registry
from app.main import create_application
from app.models.registry import ModelRegistry


def _make_mock_registry() -> MagicMock:
    reg = MagicMock(spec=ModelRegistry)
    reg.device = "cpu"
    reg.torch_dtype = torch.float32

    # Mock DINOv2
    dino_model = MagicMock()
    dino_outputs = MagicMock()
    dino_outputs.last_hidden_state = torch.randn(1, 197, 768)
    dino_outputs.attentions = [torch.rand(1, 12, 197, 197)]
    dino_model.return_value = dino_outputs
    dino_proc = MagicMock()
    dino_proc.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
    reg.get_dinov2.return_value = (dino_model, dino_proc)

    # Mock SegFormer
    seg_model = MagicMock()
    seg_outputs = MagicMock()
    seg_outputs.logits = torch.randn(1, 19, 16, 16)
    seg_model.return_value = seg_outputs
    seg_proc = MagicMock()
    seg_proc.return_value = {"pixel_values": torch.randn(1, 3, 64, 64)}
    reg.get_segformer.return_value = (seg_model, seg_proc)

    # Mock CLIP
    clip_model = MagicMock()
    clip_outputs = MagicMock()
    clip_outputs.logits_per_image = torch.tensor([[5.0, 2.0, 1.0, 0.5, 0.1, 0.0, 0.2, 0.1]])
    clip_model.return_value = clip_outputs
    clip_proc = MagicMock()
    clip_proc.return_value = {
        "pixel_values": torch.randn(1, 3, 224, 224),
        "input_ids": torch.randint(0, 100, (8, 10)),
    }
    reg.get_clip.return_value = (clip_model, clip_proc)

    status_obj = MagicMock()
    status_obj.models = {}
    reg.get_status.return_value = status_obj

    return reg


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient with the model registry dependency overridden."""
    mock_reg = _make_mock_registry()
    app = create_application()
    # Override dependency — no lifespan HF calls, no model downloads
    app.dependency_overrides[require_model_registry] = lambda: mock_reg
    # Also plant in app.state for get_pipeline singleton logic
    app.state.model_registry = mock_reg
    return TestClient(app, raise_server_exceptions=True)


class TestAnalysisEndpoints:
    """Test REST API routes for analysis."""

    def test_analyze_no_input_returns_400(self, client: TestClient) -> None:
        response = client.post("/api/v1/analyze")
        assert response.status_code == 400

    def test_analyze_file_upload_success(self, client: TestClient) -> None:
        img = Image.new("RGB", (64, 64), color="gray")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v1/analyze",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "track_condition" in data
        assert "metrics" in data
        assert "tyre_recommendation" in data
        assert "explainability" in data
        assert "visualization" in data

    def test_get_history_returns_results(self, client: TestClient) -> None:
        response = client.get("/api/v1/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert isinstance(data["history"], list)
