#!/usr/bin/env python3
"""
APEX Setup Verification Script.

Run this before starting development to verify that the environment
is correctly configured for this machine's hardware.

Usage:
    cd backend
    python scripts/verify_setup.py

What it checks:
    1. Python version (>= 3.10 required, 3.11 recommended)
    2. Core package availability (fastapi, pydantic, loguru)
    3. PyTorch availability and CUDA status
    4. HuggingFace libraries
    5. Computer vision libraries (Pillow, OpenCV, numpy)
    6. Estimated memory budget vs model requirements
    7. HF token (warns if missing)
    8. Required directories (creates them if missing)
    9. Disk space estimate for model downloads
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Colour codes for terminal output (works on Windows 10+ and all Unix)
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

PASS   = f"{GREEN}✓ PASS{RESET}"
WARN   = f"{YELLOW}⚠ WARN{RESET}"
FAIL   = f"{RED}✗ FAIL{RESET}"
INFO   = f"{CYAN}ℹ INFO{RESET}"


def _header(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")


def _check(label: str, passed: bool, message: str, warn_only: bool = False) -> bool:
    """Print a check result. Returns True if passed (or warn_only)."""
    icon = PASS if passed else (WARN if warn_only else FAIL)
    print(f"  {icon}  {label:<40} {DIM}{message}{RESET}")
    return passed or warn_only


def check_python() -> bool:
    _header("Python Environment")
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    ok = v >= (3, 10)
    _check("Python version", ok, f"{version_str} {'(OK)' if ok else '(needs >= 3.10)'}")
    if not ok:
        print(f"\n  {RED}Install Python 3.10+ and re-run this script.{RESET}")
    return ok


def check_core_packages() -> bool:
    _header("Core Framework Packages")
    packages = {
        "fastapi": "Web framework",
        "pydantic": "Data validation",
        "pydantic_settings": "Settings management",
        "loguru": "Logging",
        "uvicorn": "ASGI server",
        "dotenv": "Env file loading (python-dotenv)",
        "httpx": "Async HTTP client",
        "aiofiles": "Async file I/O",
        "cachetools": "LRU caching",
        "psutil": "System monitoring",
    }
    all_ok = True
    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            _check(desc, True, f"[{pkg}] installed")
        except ImportError:
            _check(desc, False, f"[{pkg}] MISSING — pip install {pkg}")
            all_ok = False
    return all_ok


def check_torch() -> tuple[bool, bool]:
    """Returns (torch_available, cuda_available)."""
    _header("PyTorch (compute backbone)")
    try:
        import torch  # noqa: PLC0415
        version = torch.__version__
        _check("PyTorch", True, f"version {version}")

        cuda = torch.cuda.is_available()
        if cuda:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            _check("CUDA", True, f"{device_name} ({vram_gb:.1f} GB VRAM)")
        else:
            _check(
                "CUDA",
                True,
                "Not available — CPU-only mode (perfectly fine for this project)",
                warn_only=True,
            )

        # Check dtype support
        if not cuda:
            _check(
                "float32 CPU",
                True,
                "Will use float32 on CPU (float16 is NOT efficient on CPU)",
            )

        return True, cuda

    except ImportError:
        _check(
            "PyTorch",
            False,
            "NOT INSTALLED. Install with:\n"
            "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
        )
        return False, False


def check_hf_libraries() -> bool:
    _header("HuggingFace Libraries")
    libs = {
        "transformers": "Model loading (BERT, ViT, etc.)",
        "huggingface_hub": "Hub access + Inference API",
        "accelerate": "Efficient model placement",
        "safetensors": "Fast/safe weight format",
        "tokenizers": "Fast tokenisation",
    }
    all_ok = True
    for lib, desc in libs.items():
        try:
            __import__(lib)
            _check(desc, True, f"[{lib}]")
        except ImportError:
            _check(desc, False, f"[{lib}] MISSING — pip install {lib}")
            all_ok = False
    return all_ok


def check_cv_libraries() -> bool:
    _header("Computer Vision Libraries")
    libs = {
        "PIL": "Pillow — image loading/processing",
        "cv2": "OpenCV (headless) — video processing",
        "numpy": "NumPy — array operations",
        "scipy": "SciPy — temporal regression",
    }
    all_ok = True
    for lib, desc in libs.items():
        try:
            __import__(lib)
            _check(desc, True, f"[{lib}]")
        except ImportError:
            pip_name = {
                "PIL": "Pillow",
                "cv2": "opencv-python-headless",
            }.get(lib, lib)
            _check(desc, False, f"[{lib}] MISSING — pip install {pip_name}")
            all_ok = False
    return all_ok


def check_memory(cuda_available: bool) -> bool:
    _header("Memory Budget")

    try:
        import psutil  # noqa: PLC0415
        mem = psutil.virtual_memory()
        ram_gb = mem.total / 1e9
        avail_gb = mem.available / 1e9

        _check("Total RAM", ram_gb >= 6.0, f"{ram_gb:.1f} GB {'(OK)' if ram_gb >= 6.0 else '(low — may be tight)'}")
        _check("Available RAM now", avail_gb >= 3.0, f"{avail_gb:.1f} GB free")

    except ImportError:
        _check("RAM check", True, "psutil not installed — skipping", warn_only=True)
        ram_gb = 8.0  # assume OK

    # Model memory estimates (float32, CPU):
    estimates = {
        "DINOv2-base":       0.34,   # 86M params × 4 bytes
        "SegFormer-B2":      0.11,   # 27M params × 4 bytes
        "CLIP ViT-L/14":     1.20,   # 307M params × 4 bytes
        "VLM (HF API)":      0.00,   # Runs on HF servers — 0 local RAM
    }
    total_local_gb = sum(v for k, v in estimates.items() if "HF API" not in k)

    print(f"\n  {DIM}Model RAM estimates (float32, CPU):{RESET}")
    for name, gb in estimates.items():
        note = "→ served by HuggingFace, 0 local RAM" if gb == 0.0 else f"≈ {gb:.2f} GB"
        print(f"    {DIM}{name:<30} {note}{RESET}")
    print(f"    {DIM}{'─' * 45}{RESET}")
    print(f"    {DIM}{'Total local models':<30} ≈ {total_local_gb:.2f} GB{RESET}")

    budget_ok = ram_gb >= (total_local_gb + 3.0)  # 3 GB headroom
    _check(
        "Memory budget",
        budget_ok,
        (
            f"Models need {total_local_gb:.1f} GB + 3 GB OS headroom = {total_local_gb + 3:.1f} GB total"
            f" — {'OK' if budget_ok else 'MAY BE TIGHT — use lazy loading (already enabled)'}"
        ),
        warn_only=not budget_ok,
    )
    return True  # Warn only — don't block setup


def check_hf_token() -> bool:
    _header("HuggingFace Token")

    # Look for token in .env file first
    env_path = Path(__file__).parent.parent / ".env"
    token_found = False

    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("HF_TOKEN=") and "=" in line:
                    token = line.split("=", 1)[1].strip()
                    if token and token != '""' and token != "''":
                        token_found = True
                        break

    if not token_found:
        token_found = bool(os.environ.get("HF_TOKEN"))

    _check(
        "HF_TOKEN configured",
        token_found,
        (
            "Set in .env ✓"
            if token_found
            else "NOT SET — VLM explanations will use structured fallback\n"
            "    Get a free token at: https://huggingface.co/settings/tokens\n"
            "    Add HF_TOKEN=hf_... to backend/.env"
        ),
        warn_only=True,  # System works without it
    )
    return True


def check_directories() -> bool:
    _header("Project Directories")

    backend_root = Path(__file__).parent.parent
    dirs_to_check = {
        "backend/":          backend_root,
        "backend/app/":      backend_root / "app",
        "backend/logs/":     backend_root / "logs",
        "backend/.hf_cache/": backend_root / ".hf_cache",
        "backend/demo_clips/": backend_root / "demo_clips",
        "backend/tests/":    backend_root / "tests",
        "backend/scripts/":  backend_root / "scripts",
    }

    all_ok = True
    for label, path in dirs_to_check.items():
        if path.exists():
            _check(label, True, "exists")
        else:
            path.mkdir(parents=True, exist_ok=True)
            _check(label, True, "created ✓")

    return all_ok


def check_disk_space() -> bool:
    _header("Disk Space (for model downloads)")

    try:
        import shutil  # noqa: PLC0415
        backend_root = Path(__file__).parent.parent
        total, used, free = shutil.disk_usage(backend_root)
        free_gb = free / 1e9

        # Model download sizes:
        # DINOv2-base:        ~340 MB
        # SegFormer-B2:       ~110 MB
        # CLIP ViT-L/14:      ~890 MB
        # Total:              ~1.34 GB
        needed_gb = 2.0  # 1.34 + buffer

        _check(
            "Free disk space",
            free_gb >= needed_gb,
            f"{free_gb:.1f} GB free (need ~{needed_gb:.0f} GB for model downloads)",
            warn_only=free_gb < needed_gb,
        )
    except Exception as e:
        _check("Disk space", True, f"Could not check: {e}", warn_only=True)
    return True


def main() -> int:
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  APEX PitWall Intelligence — Setup Verification{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")
    print(f"  {DIM}Running on Python {sys.version}{RESET}")

    results = []
    results.append(check_python())
    results.append(check_core_packages())
    torch_ok, cuda_ok = check_torch()
    results.append(torch_ok)
    results.append(check_hf_libraries())
    results.append(check_cv_libraries())
    check_memory(cuda_ok)
    check_hf_token()
    check_directories()
    check_disk_space()

    # Summary
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    if all(results):
        print(f"{GREEN}{BOLD}  ✓ All critical checks passed. Ready to develop!{RESET}")
        print(f"\n  Next step: {CYAN}uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload{RESET}")
        return 0
    else:
        print(f"{RED}{BOLD}  ✗ Some critical checks failed. Fix the issues above before proceeding.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
