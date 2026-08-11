# PitWall Intelligence
### APEX — Adaptive Perception & Evolution eXplainer

> AI-powered race track condition analysis and tyre strategy engine.

---

## Quick Start (Development — CPU-only, 8 GB RAM)

```bash
# 1. Create virtual environment
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 2. Install PyTorch (CPU build — avoids 2 GB CUDA download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install all other dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env — add your HF_TOKEN for VLM explanations

# 5. Verify setup
python scripts/verify_setup.py

# 6. Pre-download AI models (one-time, ~1.3 GB)
python scripts/download_models.py

# 7. Start the backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 8. Open frontend (in a separate terminal or browser)
# Just open frontend/index.html in your browser, or:
# python -m http.server 3000 --directory frontend
```

Backend API docs: **http://localhost:8000/docs**  
Health check: **http://localhost:8000/api/v1/health**

---

## Architecture Overview

```
PitWall Intelligence / APEX
├── backend/                    FastAPI inference server
│   ├── app/
│   │   ├── config.py           Pydantic settings (CPU-first defaults)
│   │   ├── main.py             FastAPI app factory + lifespan
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── health.py   /health, /api/v1/health, /api/v1/warmup
│   │   │   │   ├── analysis.py POST /api/v1/analyze/image|video  (Module 14)
│   │   │   │   ├── session.py  GET /api/v1/session/{id}          (Module 14)
│   │   │   │   └── websocket.py WS /ws/{session_id}              (Module 14)
│   │   │   └── dependencies.py DI: settings, model registry, session store
│   │   ├── core/               APEX inference pipeline
│   │   │   ├── pipeline.py     Master orchestrator               (Module 11)
│   │   │   ├── perception.py   DINOv2 + SegFormer + CLIP         (Module 03)
│   │   │   ├── temporal.py     Sliding window reasoning          (Module 07)
│   │   │   ├── physics.py      Grip estimation + state machine   (Module 06)
│   │   │   ├── recommender.py  Strategy recommendation engine    (Module 09)
│   │   │   └── explainer.py    VLM + structured fallback         (Module 10)
│   │   ├── models/
│   │   │   ├── registry.py     Model lazy-loader                 (Module 01)
│   │   │   └── schemas.py      All Pydantic request/response types
│   │   ├── services/
│   │   │   ├── session_store.py In-memory session state          (Module 12)
│   │   │   ├── weather.py       OpenWeatherMap integration        (Module 15)
│   │   │   └── video_processor.py Frame extraction pipeline      (Module 13)
│   │   └── utils/
│   │       ├── logging_config.py Loguru setup
│   │       ├── image_utils.py    Preprocessing                   (Module 02)
│   │       ├── visualization.py  Heatmap + overlay generation    (Module 04)
│   │       └── calibration.py    Confidence calibration          (Module 08)
│   ├── scripts/
│   │   ├── verify_setup.py     Pre-flight hardware check
│   │   └── download_models.py  Model pre-download utility
│   ├── tests/
│   │   └── test_health.py      Module 00 test suite
│   ├── demo_clips/             Pre-loaded race scenarios
│   ├── requirements.txt        Python dependencies
│   └── .env.example            Configuration template
├── frontend/
│   └── index.html              Frontend (single HTML file during Module 00)
├── docker-compose.yml
├── nginx.conf
└── README.md
```

---

## AI Pipeline (APEX)

```
Image / Video Frame
        │
        ▼
┌─────────────────────────────────────────┐
│  Stage 1: Perception Layer              │
│  ├─ DINOv2 ViT-B/14   (features)       │  facebook/dinov2-base
│  ├─ SegFormer-B2       (segmentation)  │  nvidia/segformer-b2-cityscapes
│  └─ CLIP ViT-L/14     (cross-valid.)  │  openai/clip-vit-large-patch14
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Stage 2: Surface Feature Extraction   │
│  wet_pixel_ratio, rubber_line_pct,      │
│  puddle_count, reflectance_score, …    │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Stage 3: Physics & State Machine      │
│  μ̂ (grip coefficient) estimation       │
│  ConditionState: WET_SEVERE → DRY_EV.  │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Stage 4: Temporal Reasoning           │
│  Sliding window N=8 frames             │
│  drying_rate (Δμ/frame), EWMA, forecast│
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Stage 5: Recommendation Engine        │
│  IMMEDIATE / HIGH / MONITOR / HOLD /   │
│  ABORT  +  evidence + tyre delta       │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Stage 6: Explainability               │
│  Qwen2-VL-7B (HF API) OR              │  Qwen/Qwen2-VL-7B-Instruct
│  Structured template fallback          │
└─────────────────────┬───────────────────┘
                      │
                      ▼
              APEXResult JSON
```

---

## Hardware Requirements

| Component | Minimum (Demo) | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores (2014+) | 8+ cores |
| GPU | Not required | NVIDIA 6+ GB VRAM |
| Disk | 4 GB free | 10 GB free |
| Python | 3.10+ | 3.11 |

**CPU-only mode is fully supported.** Inference is slower (~5–10 s/frame vs. ~1 s on GPU) but all features work. The VLM uses the HuggingFace Inference API and requires no local GPU.

---

## Configuration

Copy `backend/.env.example` to `backend/.env` and configure:

| Variable | Default | Notes |
|---|---|---|
| `HF_TOKEN` | _(empty)_ | Get at huggingface.co/settings/tokens |
| `VLM_PROVIDER` | `api` | `api` recommended for 8 GB RAM |
| `DEVICE` | `auto` | Auto-detects CUDA, falls back to CPU |
| `MAX_RAM_GB` | `5.0` | Model budget (leave 3 GB for OS) |
| `DEMO_MODE_ENABLED` | `true` | Pre-loaded race scenarios |
| `WEATHER_ENABLED` | `false` | Requires `WEATHER_API_KEY` |

---

## Docker Deployment

```bash
# Build and start (CPU)
docker compose up --build

# Access
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## Implementation Status

| Module | Description | Status |
|---|---|---|
| M00 | Project scaffold + config + health | ✅ Complete |
| M01 | Model Registry (lazy loading) | ⬜ Next |
| M02 | Image preprocessing | ⬜ Pending |
| M03 | Perception layer | ⬜ Pending |
| M04 | Visualisation engine | ⬜ Pending |
| M05 | Surface feature extractor | ⬜ Pending |
| M06 | Physics & state machine | ⬜ Pending |
| M07 | Temporal reasoning | ⬜ Pending |
| M08 | Confidence calibration | ⬜ Pending |
| M09 | Recommendation engine | ⬜ Pending |
| M10 | VLM explainability | ⬜ Pending |
| M11 | APEX pipeline orchestrator | ⬜ Pending |
| M12 | Session store | ⬜ Pending |
| M13 | Video processor | ⬜ Pending |
| M14 | FastAPI routes | ⬜ Pending |
| M15 | Weather service | ⬜ Pending |
| M16 | Frontend design system | ⬜ Pending |
| M17–M25 | Frontend components | ⬜ Pending |
