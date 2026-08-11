"""
Module 02 tests — Image Preprocessing Utilities.

Tests for load_image, compute_image_hash, preprocess_frame,
and edge cases (corrupt files, empty inputs, EXIF rotation, RGBA/L modes, downscaling).
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

# Ensure backend root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.image_utils import (
    MAX_IMAGE_DIMENSION,
    PreprocessedFrame,
    compute_image_hash,
    load_image,
    preprocess_frame,
)


# ---------------------------------------------------------------------------
# Helper fixtures / image generators
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Create a simple 100x80 RGB PIL image."""
    arr = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_rgba_image() -> Image.Image:
    """Create a 100x80 RGBA PIL image."""
    arr = np.random.randint(0, 255, (80, 100, 4), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def sample_jpeg_bytes(sample_rgb_image: Image.Image) -> bytes:
    """Encode sample_rgb_image into JPEG bytes."""
    buf = io.BytesIO()
    sample_rgb_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_base64_str(sample_jpeg_bytes: bytes) -> str:
    """Encode JPEG bytes as base64 string with header."""
    b64 = base64.b64encode(sample_jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Tests for load_image
# ---------------------------------------------------------------------------

class TestLoadImageValidInputs:
    """Test load_image with various valid input formats."""

    def test_load_from_pil_image(self, sample_rgb_image: Image.Image) -> None:
        img = load_image(sample_rgb_image)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (100, 80)

    def test_load_from_rgba_pil_converts_to_rgb(self, sample_rgba_image: Image.Image) -> None:
        img = load_image(sample_rgba_image)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (100, 80)

    def test_load_from_bytes(self, sample_jpeg_bytes: bytes) -> None:
        img = load_image(sample_jpeg_bytes)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (100, 80)

    def test_load_from_base64_with_header(self, sample_base64_str: str) -> None:
        img = load_image(sample_base64_str)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (100, 80)

    def test_load_from_raw_base64_without_header(self, sample_jpeg_bytes: bytes) -> None:
        b64_plain = base64.b64encode(sample_jpeg_bytes).decode("utf-8")
        img = load_image(b64_plain)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_load_from_numpy_rgb_array(self) -> None:
        arr = np.zeros((60, 80, 3), dtype=np.uint8)
        img = load_image(arr)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (80, 60)

    def test_load_from_numpy_grayscale_array(self) -> None:
        arr = np.zeros((60, 80), dtype=np.uint8)
        img = load_image(arr)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (80, 60)

    def test_load_from_file_path(self, tmp_path: Path, sample_rgb_image: Image.Image) -> None:
        file_path = tmp_path / "test_track.jpg"
        sample_rgb_image.save(file_path, format="JPEG")

        # Test both string and Path object
        img1 = load_image(file_path)
        img2 = load_image(str(file_path))
        assert img1.size == (100, 80)
        assert img2.size == (100, 80)


class TestLoadImageInvalidInputs:
    """Test load_image error handling for corrupt / invalid data."""

    def test_load_none_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be None"):
            load_image(None)  # type: ignore

    def test_load_empty_bytes_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            load_image(b"")

    def test_load_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Empty string"):
            load_image("   ")

    def test_load_invalid_base64_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Failed to decode"):
            load_image("invalid_b64_string_!!!")

    def test_load_non_existent_file_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            load_image("/path/does/not/exist/image.jpg")

    def test_load_empty_file_raises_value_error(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "zero_byte.png"
        empty_file.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            load_image(empty_file)

    def test_load_corrupt_bytes_raises_value_error(self) -> None:
        corrupt_bytes = b"NOT_AN_IMAGE_DATA_HEADER_XYZ"
        with pytest.raises(ValueError, match="Failed to parse"):
            load_image(corrupt_bytes)


class TestImageSafetyDownscaling:
    """Test safe downscaling for large images (> max_dimension)."""

    def test_large_image_downscaled_to_max_dimension(self) -> None:
        # Create a 3000 x 1500 image
        large_arr = np.zeros((1500, 3000, 3), dtype=np.uint8)
        large_img = Image.fromarray(large_arr)

        img = load_image(large_img, max_dimension=1000)
        assert max(img.size) == 1000
        # Aspect ratio preserved (3000/1500 = 2.0 -> 1000 / 500)
        assert img.size == (1000, 500)

    def test_small_image_not_upscaled(self, sample_rgb_image: Image.Image) -> None:
        img = load_image(sample_rgb_image, max_dimension=2048)
        assert img.size == (100, 80)


class TestImageHashing:
    """Test compute_image_hash output and determinism."""

    def test_hash_is_valid_sha256_hex(self, sample_rgb_image: Image.Image) -> None:
        h = compute_image_hash(sample_rgb_image)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex string length

    def test_identical_images_produce_identical_hashes(self, sample_rgb_image: Image.Image) -> None:
        h1 = compute_image_hash(sample_rgb_image)
        h2 = compute_image_hash(sample_rgb_image.copy())
        assert h1 == h2


class TestPreprocessFrame:
    """Test preprocess_frame function and PreprocessedFrame container."""

    def test_preprocess_frame_without_registry(self, sample_jpeg_bytes: bytes) -> None:
        frame = preprocess_frame(sample_jpeg_bytes, frame_index=5)

        assert isinstance(frame, PreprocessedFrame)
        assert frame.original_size == (100, 80)
        assert frame.width == 100
        assert frame.height == 80
        assert frame.aspect_ratio == pytest.approx(1.25)
        assert frame.frame_index == 5
        assert len(frame.image_hash) == 64
        assert frame.dinov2_tensor is None
        assert frame.segformer_inputs is None
        assert frame.clip_inputs is None

    def test_preprocess_frame_with_mocked_registry(self, sample_rgb_image: Image.Image) -> None:
        mock_registry = MagicMock()
        mock_registry.device = "cpu"

        # Mock processors
        mock_processor = MagicMock()
        mock_processor.side_effect = lambda images, return_tensors: {
            "pixel_values": MagicMock()
        }

        mock_registry.get_dinov2.return_value = (MagicMock(), mock_processor)
        mock_registry.get_segformer.return_value = (MagicMock(), mock_processor)
        mock_registry.get_clip.return_value = (MagicMock(), mock_processor)

        frame = preprocess_frame(sample_rgb_image, registry=mock_registry)

        assert frame.dinov2_tensor is not None
        assert frame.segformer_inputs is not None
        assert frame.clip_inputs is not None
