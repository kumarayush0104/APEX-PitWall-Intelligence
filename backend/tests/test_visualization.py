"""
Module 04 tests — Visualization Engine.

Tests for hex_to_rgb, pil_to_base64, render_segmentation_mask,
blend_segmentation_overlay, render_attention_heatmap, and create_visualization_bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Ensure backend root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.perception import PerceptionResult, SurfaceClass
from app.models.schemas import VisualizationBundle
from app.utils.image_utils import PreprocessedFrame
from app.utils.visualization import (
    blend_segmentation_overlay,
    create_visualization_bundle,
    hex_to_rgb,
    pil_to_base64,
    render_attention_heatmap,
    render_segmentation_mask,
)


# ---------------------------------------------------------------------------
# Test Helpers & Conversions
# ---------------------------------------------------------------------------

class TestVisualizationHelpers:
    """Test hex_to_rgb and base64 helpers."""

    def test_hex_to_rgb_valid(self) -> None:
        assert hex_to_rgb("#FF1E00") == (255, 30, 0)
        assert hex_to_rgb("00C853") == (0, 200, 83)

    def test_hex_to_rgb_invalid(self) -> None:
        assert hex_to_rgb("invalid") == (255, 255, 255)

    def test_pil_to_base64(self) -> None:
        img = Image.new("RGB", (10, 10), color="red")
        b64 = pil_to_base64(img, format_name="PNG")
        assert isinstance(b64, str)
        assert b64.startswith("data:image/png;base64,")
        assert len(b64) > 30


# ---------------------------------------------------------------------------
# Test Rendering Functions
# ---------------------------------------------------------------------------

class TestRenderingFunctions:
    """Test mask, overlay, and heatmap rendering."""

    @pytest.fixture
    def dummy_image(self) -> Image.Image:
        arr = np.full((40, 50, 3), 100, dtype=np.uint8)
        return Image.fromarray(arr)

    @pytest.fixture
    def dummy_mask(self) -> np.ndarray:
        mask = np.zeros((40, 50), dtype=np.uint8)
        mask[10:30, 10:40] = SurfaceClass.WET_SURFACE
        mask[15:25, 15:25] = SurfaceClass.PUDDLE
        return mask

    def test_render_segmentation_mask(self, dummy_mask: np.ndarray) -> None:
        img = render_segmentation_mask(dummy_mask)
        assert isinstance(img, Image.Image)
        assert img.size == (50, 40)
        assert img.mode == "RGB"

    def test_blend_segmentation_overlay(self, dummy_image: Image.Image, dummy_mask: np.ndarray) -> None:
        blended = blend_segmentation_overlay(dummy_image, dummy_mask, alpha=0.5)
        assert isinstance(blended, Image.Image)
        assert blended.size == (50, 40)
        assert blended.mode == "RGB"

    def test_render_attention_heatmap(self, dummy_image: Image.Image) -> None:
        attn_grid = np.random.rand(14, 14).astype(np.float32)
        heatmap = render_attention_heatmap(dummy_image, attn_grid, alpha=0.5)
        assert isinstance(heatmap, Image.Image)
        assert heatmap.size == (50, 40)
        assert heatmap.mode == "RGB"


# ---------------------------------------------------------------------------
# Test Master Visualization Bundle
# ---------------------------------------------------------------------------

class TestVisualizationBundle:
    """Test create_visualization_bundle output schema."""

    def test_create_visualization_bundle(self) -> None:
        pil_img = Image.new("RGB", (64, 48), color="gray")
        frame = PreprocessedFrame(
            pil_image=pil_img,
            original_size=(64, 48),
            resized_size=(64, 48),
            aspect_ratio=64 / 48,
            image_hash="hash_123",
        )

        mask = np.zeros((48, 64), dtype=np.uint8)
        perception = PerceptionResult(
            frame_index=0,
            timestamp="2026-08-11T00:00:00Z",
            image_hash="hash_123",
            segmentation_mask=mask,
            class_proportions={"dry_surface": 1.0},
            clip_scores={"dry_evolved": 1.0},
            dinov2_cls_token=np.zeros((768,), dtype=np.float32),
            dinov2_patch_embeddings=np.zeros((196, 768), dtype=np.float32),
            attention_map=np.zeros((14, 14), dtype=np.float32),
            confidence_agreement=1.0,
            processing_time_ms=10.0,
        )

        bundle = create_visualization_bundle(frame, perception)

        assert isinstance(bundle, VisualizationBundle)
        assert bundle.original_width == 64
        assert bundle.original_height == 48
        assert bundle.overlay_b64.startswith("data:image/png;base64,")
        assert bundle.attention_heatmap_b64.startswith("data:image/png;base64,")
        assert bundle.segmentation_b64.startswith("data:image/png;base64,")
        assert len(bundle.class_legend) == 7
