"""
APEX Image Preprocessing Utilities — Module 02.

Normalizes all incoming visual media (raw upload bytes, base64 strings,
local file paths, URLs, PIL Images, or NumPy arrays) into standard RGB PIL
images and model-ready PyTorch tensors.

Engineering features:
  - EXIF orientation auto-correction via ImageOps.exif_transpose.
  - Safe format conversion (RGBA, Palette, Grayscale, CMYK -> RGB).
  - Safety downscaling (max edge cap, e.g. 2048px) to protect 8 GB RAM.
  - Image hash generation (SHA256) for downstream caching.
  - Model-specific tensor preprocessing for DINOv2, SegFormer, and CLIP.
  - Robust exception handling for corrupt bytes, zero-length input, invalid formats.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from loguru import logger
from PIL import Image, ImageFile, ImageOps

# Ensure PIL loads truncated/damaged images if recoverable
ImageFile.LOAD_TRUNCATED_IMAGES = True

if TYPE_CHECKING:
    import torch
    from app.models.registry import ModelRegistry

# Maximum dimension (width or height) allowed before safe downscaling
MAX_IMAGE_DIMENSION = 2048


# ---------------------------------------------------------------------------
# PreprocessedFrame Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedFrame:
    """
    Standardized container holding normalized image media and ready tensors.
    """
    pil_image: Image.Image
    original_size: tuple[int, int]  # (width, height)
    resized_size: tuple[int, int]   # (width, height) after max-dim downscaling
    aspect_ratio: float             # width / height
    image_hash: str                 # SHA-256 hash of RGB bytes for caching
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    frame_index: int = 0

    # Model-specific tensors (populated on demand or via preprocess_frame)
    dinov2_tensor: torch.Tensor | None = None
    segformer_inputs: dict[str, torch.Tensor] | None = None
    clip_inputs: dict[str, torch.Tensor] | None = None

    @property
    def width(self) -> int:
        return self.resized_size[0]

    @property
    def height(self) -> int:
        return self.resized_size[1]


# ---------------------------------------------------------------------------
# Image Loading & Normalization
# ---------------------------------------------------------------------------

def load_image(
    input_data: bytes | str | Path | Image.Image | np.ndarray,
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> Image.Image:
    """
    Load any input format into a normalized, EXIF-corrected RGB PIL Image.

    Supports:
      - Raw bytes (JPEG, PNG, WebP, BMP, TIFF)
      - Base64 encoded string (with or without 'data:image/...;base64,' header)
      - File path string or Path object
      - Existing PIL.Image instance
      - NumPy ndarray (OpenCV BGR/RGB array or grayscale)

    Raises:
      - ValueError: for empty, invalid, or unparseable input data.
    """
    if input_data is None:
        raise ValueError("Input image data cannot be None")

    img: Image.Image | None = None

    # 1. Existing PIL Image
    if isinstance(input_data, Image.Image):
        img = input_data

    # 2. NumPy Array (OpenCV or raw array)
    elif isinstance(input_data, np.ndarray):
        if input_data.size == 0:
            raise ValueError("Input NumPy array is empty")
        
        # Handle grayscale (H, W), BGR/RGB (H, W, 3), RGBA (H, W, 4)
        if input_data.ndim == 2:
            img = Image.fromarray(input_data).convert("RGB")
        elif input_data.ndim == 3:
            if input_data.shape[2] == 4:
                img = Image.fromarray(input_data, mode="RGBA").convert("RGB")
            elif input_data.shape[2] == 3:
                # Default to RGB. If caller passes OpenCV BGR, they can convert beforehand or rely on PIL
                img = Image.fromarray(input_data, mode="RGB")
            else:
                raise ValueError(f"Unsupported NumPy array shape: {input_data.shape}")
        else:
            raise ValueError(f"Unsupported NumPy array dimensions: {input_data.ndim}")

    # 3. Path / File Path String
    elif isinstance(input_data, Path) or (
        isinstance(input_data, str)
        and not input_data.startswith("data:")
        and (
            input_data.startswith(("http://", "https://", "file://"))
            or input_data.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"))
            or "\\" in input_data
            or (len(input_data) < 260 and Path(input_data).is_file())
            or (len(input_data) < 260 and not len(input_data) > 64 and ("/" in input_data and "." in input_data and not input_data.startswith("/9j/")))
        )
    ):
        path = Path(input_data)
        if not path.exists():
            raise ValueError(f"File path does not exist: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"File is empty (0 bytes): {path}")
        try:
            img = Image.open(path)
        except Exception as exc:
            raise ValueError(f"Failed to open image file {path}: {exc}") from exc

    # 4. Base64 String
    elif isinstance(input_data, str):
        cleaned_b64 = input_data.strip()
        if not cleaned_b64:
            raise ValueError("Empty string provided as image input")
        
        # Strip data URI header if present
        if "," in cleaned_b64 and cleaned_b64.startswith("data:"):
            cleaned_b64 = cleaned_b64.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(cleaned_b64)
            if not raw_bytes:
                raise ValueError("Decoded base64 string produced 0 bytes")
            img = Image.open(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise ValueError(f"Failed to decode base64 image data: {exc}") from exc

    # 5. Raw Bytes
    elif isinstance(input_data, bytes):
        if not input_data:
            raise ValueError("Raw image bytes are empty (0 bytes)")
        try:
            img = Image.open(io.BytesIO(input_data))
        except Exception as exc:
            raise ValueError(f"Failed to parse image bytes: {exc}") from exc

    else:
        raise ValueError(f"Unsupported image input type: {type(input_data)}")

    if img is None:
        raise ValueError("Failed to load image from input")

    # Correct EXIF orientation (e.g. photos taken from smartphones/tablets)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception as exc:
        logger.debug("EXIF transpose skipped: {}", exc)

    # Convert to RGB mode (handles RGBA, P, L, CMYK, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Safety check on dimensions & downscaling if exceeding max_dimension
    width, height = img.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    if max(width, height) > max_dimension:
        scale = max_dimension / float(max(width, height))
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        logger.debug("Downscaling image from {}x{} to {}x{} for safety", width, height, new_w, new_h)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def compute_image_hash(image: Image.Image) -> str:
    """Compute a SHA-256 fingerprint for a PIL RGB Image."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


# ---------------------------------------------------------------------------
# Preprocessing Pipeline Function
# ---------------------------------------------------------------------------

def preprocess_frame(
    input_data: bytes | str | Path | Image.Image | np.ndarray,
    registry: ModelRegistry | None = None,
    target_models: list[Literal["dinov2", "segformer", "clip"]] | None = None,
    frame_index: int = 0,
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> PreprocessedFrame:
    """
    Complete preprocessing pipeline for an input frame.

    Args:
        input_data: Raw bytes, base64, path, PIL Image, or numpy array.
        registry: ModelRegistry singleton (optional). If provided, processors
                  are retrieved to prepare PyTorch model input tensors.
        target_models: List of model keys to prepare tensors for ("dinov2", "segformer", "clip").
                       Defaults to all available local models in registry.
        frame_index: Index of frame in sequence (defaults to 0).
        max_dimension: Cap for image edge size.

    Returns:
        PreprocessedFrame containing PIL Image, metadata, and optional model tensors.
    """
    pil_img = load_image(input_data, max_dimension=max_dimension)
    orig_w, orig_h = pil_img.size
    img_hash = compute_image_hash(pil_img)
    aspect_ratio = orig_w / float(orig_h)

    frame = PreprocessedFrame(
        pil_image=pil_img,
        original_size=(orig_w, orig_h),
        resized_size=pil_img.size,
        aspect_ratio=aspect_ratio,
        image_hash=img_hash,
        frame_index=frame_index,
    )

    if registry is None:
        return frame

    if target_models is None:
        target_models = ["dinov2", "segformer", "clip"]

    # Preprocess model tensors using registry processors
    device = registry.device

    # 1. DINOv2 Tensor Preprocessing
    if "dinov2" in target_models:
        try:
            _, processor = registry.get_dinov2()
            if processor is not None:
                inputs = processor(images=pil_img, return_tensors="pt")
                frame.dinov2_tensor = inputs["pixel_values"].to(device)
        except Exception as exc:
            logger.warning("Failed to prepare DINOv2 tensor: {}", exc)

    # 2. SegFormer Tensor Preprocessing
    if "segformer" in target_models:
        try:
            _, processor = registry.get_segformer()
            if processor is not None:
                inputs = processor(images=pil_img, return_tensors="pt")
                frame.segformer_inputs = {
                    k: v.to(device) for k, v in inputs.items()
                }
        except Exception as exc:
            logger.warning("Failed to prepare SegFormer inputs: {}", exc)

    # 3. CLIP Tensor Preprocessing
    if "clip" in target_models:
        try:
            _, processor = registry.get_clip()
            if processor is not None:
                inputs = processor(images=pil_img, return_tensors="pt")
                frame.clip_inputs = {
                    k: v.to(device) for k, v in inputs.items()
                }
        except Exception as exc:
            logger.warning("Failed to prepare CLIP inputs: {}", exc)

    return frame
