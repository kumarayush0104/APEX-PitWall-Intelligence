# APEX PitWall Intelligence

**Adaptive Perception and Evolution eXplainer for Formula 1 Track Condition Analysis**

APEX PitWall Intelligence is an AI-powered race engineering decision support system that performs real-time track condition analysis and tyre strategy recommendation from camera footage. The system replicates the visual intelligence of a senior race engineer — translating raw visual input from a track camera into structured, actionable strategy decisions backed by deep learning, temporal reasoning, and explainable AI.

---

## Overview

In Formula 1, track condition assessment is one of the most consequential and time-pressured decisions a race engineer makes. A misjudged tyre call during a weather transition can cost multiple positions or end a race. APEX automates this decision loop by applying a multi-stage computer vision pipeline to any track-facing image or live camera feed.

The system outputs:
- A classified track condition (Dry, Damp, Wet, or Flooded) with confidence scores
- A grip coefficient estimate derived from surface segmentation data
- Per-sector risk assessment across three track sectors
- A tyre compound recommendation with engineering rationale
- Temporal trend analysis across a rolling 30-frame window
- Structured or VLM-generated explanation in race engineer language
- Segmentation overlays, DINOv2 attention heatmaps, and visualization bundles

---

## Technical Architecture

```
Image Upload / Live Camera Frame
            |
            v
    Image Preprocessing
    (safety downscaling, EXIF correction, format normalization)
            |
            v
    Perception Layer
    |-- DINOv2 ViT-B/14       (self-supervised visual features + attention maps)
    |-- SegFormer-B2           (semantic segmentation: road surface classification)
    `-- CLIP ViT-L/14          (zero-shot condition cross-validation)
            |
            v
    Condition Classifier
    (wetness index, puddle coverage, grip coefficient, tyre recommendation)
            |
            v
    Temporal Reasoner
    (rolling 30-frame window, EWMA, trend detection, crossover alerts)
            |
            v
    Explainability Engine
    (structured template or Qwen2-VL-7B via HuggingFace Inference API)
            |
            v
    Visualization Bundle
    (segmentation overlay, attention heatmap, class legend)
            |
            v
    REST API Response  /  WebSocket Stream Frame
            |
            v
    Frontend Telemetry Command Center
    (Grip Gauge, Sector Map, Track Vision, Telemetry Chart, Strategy Panel)
```

### Model Stack

| Model | Provider | Role |
|---|---|---|
| `facebook/dinov2-base` | HuggingFace | Feature extraction, attention visualization |
| `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` | HuggingFace | Pixel-level road surface segmentation |
| `openai/clip-vit-large-patch14` | HuggingFace | Zero-shot condition classification |
| `Qwen/Qwen2-VL-7B-Instruct` | HuggingFace Inference API | Natural language engineering rationale (optional) |

All models run sequentially under a single-lock Model Registry to enforce strict RAM budget compliance on CPU-only hardware.

---

## Project Structure

```
APEX-PitWall-Intelligence/
|-- backend/
|   |-- app/
|   |   |-- api/
|   |   |   |-- dependencies.py       Dependency injection (registry, settings)
|   |   |   `-- routes/
|   |   |       |-- analysis.py       POST /api/v1/analyze, WS /api/v1/stream, GET /api/v1/history
|   |   |       `-- health.py         GET /health, GET /api/v1/health, POST /api/v1/warmup
|   |   |-- core/
|   |   |   |-- pipeline.py           Unified orchestrator (all 7 stages)
|   |   |   |-- perception.py         DINOv2 + SegFormer + CLIP inference
|   |   |   |-- condition_classifier.py  Wetness metrics, grip estimation, tyre recommendation
|   |   |   |-- temporal_reasoner.py  Rolling window, trend, crossover alerts
|   |   |   `-- explainability.py     Structured + VLM explanation engine
|   |   |-- models/
|   |   |   |-- registry.py           Thread-safe lazy model loader (RLock, sequential)
|   |   |   `-- schemas.py            All Pydantic request/response types
|   |   |-- utils/
|   |   |   |-- image_utils.py        Multi-format image loading and preprocessing
|   |   |   |-- visualization.py      Segmentation overlays, attention heatmaps
|   |   |   `-- logging_config.py     Loguru structured logging
|   |   |-- config.py                 Pydantic settings (CPU-first, RAM-aware)
|   |   `-- main.py                   FastAPI factory, lifespan, middleware
|   |-- scripts/
|   |   |-- verify_setup.py           Pre-flight hardware and dependency check
|   |   `-- download_models.py        One-time model pre-download utility
|   |-- tests/                        125-test pytest suite (unit + integration)
|   |-- requirements.txt
|   |-- Dockerfile
|   `-- .env.example
|-- frontend/
|   |-- index.html                    F1 telemetry command center UI
|   |-- css/
|   |   `-- styles.css                Design system (dark theme, glassmorphism)
|   `-- js/
|       |-- app.js                    Master controller, state management
|       |-- api.js                    REST and WebSocket communication layer
|       |-- demo_data.js              Pre-computed F1 race scenarios
|       `-- components/
|           |-- grip_gauge.js         Animated grip coefficient radial gauge
|           |-- sector_map.js         Three-sector risk map with color coding
|           |-- track_canvas.js       Segmentation overlay + attention heatmap viewer
|           |-- telemetry_chart.js    Wetness and grip trend chart
|           |-- recommendation_panel.js  Tyre strategy card with pit window alert
|           |-- ai_explanation.js     Engineering rationale and key factors display
|           |-- session_history.js    Analysis history table with replay
|           |-- report_generator.js   Printable race strategy report
|           `-- health_monitor.js     System health and model load status panel
|-- docker-compose.yml
|-- nginx.conf
`-- README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- 8 GB RAM minimum (16 GB recommended for comfort)
- No GPU required — CPU-only inference is fully supported

### Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/kumarayush0104/APEX-PitWall-Intelligence.git
cd APEX-PitWall-Intelligence

# 2. Create and activate a virtual environment
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Install PyTorch (CPU build — avoids the 2 GB CUDA wheel)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install all remaining dependencies
pip install -r requirements.txt

# 5. Configure environment variables
copy .env.example .env
# Edit .env and add your HF_TOKEN for VLM-generated explanations (optional)

# 6. Verify your hardware and installation
python scripts/verify_setup.py

# 7. Pre-download AI models (one-time, approximately 1.3 GB total)
python scripts/download_models.py

# 8. Start the backend server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 9. Serve the frontend (separate terminal)
python -m http.server 3000 --directory ../frontend
```

**Frontend**: http://localhost:3000  
**API Documentation**: http://localhost:8000/docs  
**Health Check**: http://localhost:8000/api/v1/health

---

## Docker Deployment

```bash
# Build and start all services
docker compose up --build

# Stream backend logs
docker compose logs -f backend
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

## API Reference

### POST /api/v1/analyze

Analyze a single track image and return the full APEX intelligence payload.

**Request** (`multipart/form-data`):

| Field | Type | Description |
|---|---|---|
| `file` | File | JPEG, PNG, or WebP image |
| `image_base64` | string | Base64-encoded image (alternative to file) |
| `frame_index` | integer | Sequential frame number |
| `generate_visualizations` | boolean | Include overlay and heatmap in response |

**Response**:

```json
{
  "frame_index": 42,
  "track_condition": "WET",
  "metrics": {
    "wetness_index": 0.71,
    "puddle_coverage_pct": 18.4,
    "grip_mu": 0.38,
    "clip_wet_confidence": 0.89
  },
  "tyre_recommendation": {
    "compound": "INTERMEDIATE",
    "confidence": 0.87,
    "pit_window_open": true,
    "reasoning": "..."
  },
  "temporal_analysis": {
    "trend": "DRYING",
    "volatility": "MEDIUM",
    "tyre_window_alert": { "alert_active": true, "message": "..." }
  },
  "sector_risk": {
    "sector_1": { "risk_level": "HIGH", "grip_mu": 0.41 },
    "sector_2": { "risk_level": "MEDIUM", "grip_mu": 0.48 },
    "sector_3": { "risk_level": "HIGH", "grip_mu": 0.38 }
  },
  "explainability": {
    "headline": "DRYING RACING LINE DETECTED — INTERMEDIATE WINDOW OPEN",
    "detailed_summary": "...",
    "key_factors": [...]
  },
  "visualization": {
    "overlay": "<base64>",
    "segmentation_mask": "<base64>",
    "attention_heatmap": "<base64>"
  }
}
```

### GET /api/v1/history

Returns the last 20 analysis results from the in-memory buffer.

### WebSocket /api/v1/stream

Live frame streaming endpoint. Send frames as JSON:

```json
{ "image": "data:image/jpeg;base64,..." }
```

Receive structured telemetry JSON for each frame in real time.

### GET /health

Liveness probe. Returns `200 OK` immediately. Used by Docker HEALTHCHECK.

### GET /api/v1/health

Detailed system diagnostics including model load status, hardware info, RAM usage, and service availability.

---

## Configuration Reference

Copy `backend/.env.example` to `backend/.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | (empty) | HuggingFace token for VLM API and gated models |
| `VLM_ENABLED` | `true` | Enable/disable LLM-generated explanations |
| `VLM_PROVIDER` | `api` | `api` (HF Inference API) or `local` (not recommended on 8 GB) |
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `MAX_RAM_GB` | `5.0` | Model memory budget in GB |
| `DEMO_MODE_ENABLED` | `true` | Enable pre-loaded race scenario data |
| `WEATHER_ENABLED` | `false` | OpenWeatherMap integration (requires `WEATHER_API_KEY`) |
| `CORS_ALLOW_ALL` | `true` | Permissive CORS for local development |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8 cores |
| GPU | Not required | NVIDIA with 6+ GB VRAM |
| Storage | 4 GB free | 10 GB free |
| Python | 3.10 | 3.11 |

CPU-only inference is fully supported. Frame analysis takes approximately 5–10 seconds per frame on CPU versus approximately 1 second on a mid-range GPU. The VLM explanation layer uses the HuggingFace Inference API and requires no local GPU regardless of mode.

---

## Test Suite

```bash
cd backend

# Run all tests except model download integration tests
pytest tests/ --ignore=tests/test_registry.py -v

# Run the complete suite including registry tests (requires downloaded models)
pytest tests/ -v
```

**Current status: 125 tests passing, 2 skipped (model download integration tests).**

Test coverage spans: API routes, image preprocessing, perception layer, condition classification, temporal reasoning, visualization engine, explainability engine, pipeline orchestration, and system health endpoints.

---

## Demo Scenarios

Four pre-computed F1 race scenarios are bundled for instant demonstration without requiring a connected backend:

| Scenario | Circuit | Condition | Strategy |
|---|---|---|---|
| Belgian GP | Circuit de Spa-Francorchamps | Flooded | Full Wet — Box immediately |
| Monaco GP | Circuit de Monaco | Drying line | Intermediate — Window open |
| British GP | Silverstone Circuit | Sudden shower | Wet — Crossover imminent |
| Italian GP | Autodromo Nazionale Monza | Dry evolution | Soft — Optimal stint |

---

## Implementation Status

| Module | Component | Status |
|---|---|---|
| Model Registry | Thread-safe lazy loader with RLock and RAM budget enforcement | Complete |
| Image Preprocessing | Multi-format loading, EXIF correction, safety downscaling | Complete |
| Perception Layer | DINOv2 feature extraction, SegFormer segmentation, CLIP zero-shot | Complete |
| Visualization Engine | Segmentation overlays, attention heatmaps, base64 bundles | Complete |
| Condition Classifier | Wetness metrics, grip estimation, tyre recommendation | Complete |
| Temporal Reasoner | Rolling window, linear regression trend, crossover alerts | Complete |
| Explainability Engine | Structured fallback and Qwen2-VL-7B via HF Inference API | Complete |
| Unified Pipeline | Sequential 7-stage orchestrator with timing and logging | Complete |
| REST API | POST analyze, GET history, GET health, POST warmup | Complete |
| WebSocket Stream | Live frame ingestion with JSON payload parsing | Complete |
| Frontend UI | F1 telemetry command center with 9 instrumentation components | Complete |
| Docker Deployment | Multi-service compose with Nginx reverse proxy | Complete |

---

## License

This project is submitted as a hackathon entry. All rights reserved by the author.

---

## Author

**Yuvraj Goyal**  
GitHub: [kumarayush0104](https://github.com/kumarayush0104)  
Repository: [APEX-PitWall-Intelligence](https://github.com/kumarayush0104/APEX-PitWall-Intelligence)
