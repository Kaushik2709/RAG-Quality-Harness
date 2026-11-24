# Deployment Runbook

## Overview

The platform is deployed via Docker Compose or Kubernetes containers using distinct services:
1. `rag-api`: FastAPI backend service (`infra/Dockerfile.api`)
2. `rag-dashboard`: Next.js frontend UI (`infra/Dockerfile.dashboard`)
3. `qdrant`: Persistent vector database (`qdrant/qdrant:latest`)
4. `postgres`: Persistent telemetry and trace metadata database (`postgres:16-alpine`)

---

## 1. Environment Configuration

Create a production `.env` file from `.env.example`:

```bash
# Core Application Settings
APP_ENV=production
LOG_LEVEL=INFO
API_KEY=prod-sec-key-replace-with-high-entropy-token

# Model & Provider Configuration
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...
MODEL_INIT_MODE=degraded      # 'strict' (fail-closed) or 'degraded' (graceful fallback)
RERANK_POLICY=degraded        # 'strict', 'degraded', or 'disabled'

# Storage & Database
STORE_BACKEND=qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
POSTGRES_URI=postgresql://postgres:postgres@postgres:5432/rag_eval

# Concurrency & Safeguards
MAX_INDEX_CHUNKS=250000
RATE_LIMIT_PER_MINUTE=120
RATE_LIMIT_BURST=30
```

---

## 2. Startup Procedures

### API + Storage Stack (Headless)
```bash
docker compose -f infra/docker-compose.yml up -d
```

### Full Stack (Including Next.js Dashboard)
```bash
docker compose -f infra/docker-compose.full.yml up -d
```

---

## 3. Health & Readiness Verification

Run post-deployment smoke tests:

```bash
# 1. Verify Deep Readiness
curl -f http://localhost:8000/ready

# 2. Inspect Runtime Diagnostics
curl -s http://localhost:8000/runtime/status | jq .

# 3. Verify Prometheus Metrics
curl -s http://localhost:8000/metrics | grep rag_uptime_seconds
```

---

## 4. Zero-Downtime Rolling Update

When updating application code or dependencies:
1. Build new API image:
   ```bash
   docker build -t rag-api:latest -f infra/Dockerfile.api .
   ```
2. Deploy new container alongside old container.
3. Wait for `GET /ready` to return `200 OK`.
4. Switch ingress / reverse proxy traffic to new container.
5. Terminate old container gracefully (allowing in-flight requests to complete).
