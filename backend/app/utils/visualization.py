"""
APEX Visualization Engine — Module 04.

Generates Base64-encoded visual heatmaps, segmentation overlays, and color masks
for the frontend command center UI.

Outputs:
  - overlay_b64: Original image blended with 7-class surface segmentation mask.
  - attention_heatmap_b64: DINOv2 self-attention heatmap (JET colormap) overlaid on image.
  - segmentation_b64: Colorized raw segmentation mask.
  - class_legend: Dict of surface class name -> Hex color.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from app.core.perception import (
    SURFACE_CLASS_COLORS,
    SURFACE_CLASS_NAMES,
    PerceptionResult,
    SurfaceClass,
)
from app.models.schemas import VisualizationBundle

if TYPE_CHECKING:
    from app.utils.image_utils import PreprocessedFrame


# ---------------------------------------------------------------------------
# Color Parsing & Conversion Helpers
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string ('#FF1E00') to RGB tuple (255, 30, 0)."""
    hex_clean = hex_color.lstrip("#")
    if len(hex_clean) != 6:
        return (255, 255, 255)
    return (
        int(hex_clean[0:2], 16),
        int(hex_clean[2:4], 16),
        int(hex_clean[4:6], 16),
    )


def pil_to_base64(image: Image.Image, format_name: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format_name)
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format_name.lower()};base64,{b64_data}"


# ---------------------------------------------------------------------------
# Rendering Functions
# ---------------------------------------------------------------------------

def render_segmentation_mask(segmentation_mask: np.ndarray) -> Image.Image:
    """
    Render 7-class segmentation mask as a colorized RGB PIL Image.
    """
    h, w = segmentation_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for sc in SurfaceClass:
        name = SURFACE_CLASS_NAMES[sc]
        hex_col = SURFACE_CLASS_COLORS.get(name, "#000000")
        rgb = hex_to_rgb(hex_col)
        color_mask[segmentation_mask == sc] = rgb

    return Image.fromarray(color_mask, mode="RGB")


def blend_segmentation_overlay(
    original_image: Image.Image,
    segmentation_mask: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Blend colorized segmentation mask onto original image with transparency alpha.
    """
    orig_np = np.array(original_image.convert("RGB"))
    color_mask = np.array(render_segmentation_mask(segmentation_mask))

    # Mask out background class from blending so background stays clean
    is_bg = (segmentation_mask == SurfaceClass.BACKGROUND)
    
    blended = (orig_np * (1.0 - alpha) + color_mask * alpha).astype(np.uint8)
    blended[is_bg] = orig_np[is_bg]  # Preserve original background

    return Image.fromarray(blended, mode="RGB")


def render_attention_heatmap(
    original_image: Image.Image,
    attention_map: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlay DINOv2 self-attention map onto original image using JET colormap.
    """
    orig_np = np.array(original_image.convert("RGB"))
    h, w = orig_np.shape[:2]

    # Ensure attention map is 2D float32 [0..1]
    attn = np.asarray(attention_map, dtype=np.float32)
    min_v, max_v = attn.min(), attn.max()
    if max_v > min_v:
        attn_norm = (attn - min_v) / (max_v - min_v)
    else:
        attn_norm = np.zeros_like(attn)

    # Upscale attention grid (e.g. 14x14) to full image size (H, W)
    attn_resized = cv2.resize(attn_norm, (w, h), interpolation=cv2.INTER_CUBIC)
    attn_uint8 = (attn_resized * 255.0).clip(0, 255).astype(np.uint8)

    # Apply JET colormap (returns BGR)
    heatmap_bgr = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend heatmap with original image
    blended = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return Image.fromarray(blended, mode="RGB")


# ---------------------------------------------------------------------------
# Master Bundle Builder
# ---------------------------------------------------------------------------

def create_visualization_bundle(
    frame: PreprocessedFrame,
    perception_result: PerceptionResult,
    overlay_alpha: float = 0.45,
    attention_alpha: float = 0.5,
) -> VisualizationBundle:
    """
    Construct complete VisualizationBundle containing all Base64 images and metadata.
    """
    pil_img = frame.pil_image
    orig_w, orig_h = frame.original_size
    mask = perception_result.segmentation_mask

    # Render images
    color_mask_img = render_segmentation_mask(mask)
    overlay_img = blend_segmentation_overlay(pil_img, mask, alpha=overlay_alpha)
    heatmap_img = render_attention_heatmap(pil_img, perception_result.attention_map, alpha=attention_alpha)

    # Encode to Base64
    seg_b64 = pil_to_base64(color_mask_img, format_name="PNG")
    overlay_b64 = pil_to_base64(overlay_img, format_name="PNG")
    heatmap_b64 = pil_to_base64(heatmap_img, format_name="PNG")

    return VisualizationBundle(
        overlay_b64=overlay_b64,
        attention_heatmap_b64=heatmap_b64,
        segmentation_b64=seg_b64,
        class_legend=SURFACE_CLASS_COLORS,
        original_width=orig_w,
        original_height=orig_h,
    )
