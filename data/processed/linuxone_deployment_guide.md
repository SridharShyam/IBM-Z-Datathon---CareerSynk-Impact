# IBM LinuxONE (s390x) & x86_64 Deployment & Operations Guide

## Milestone M7: FastAPI Service Deployment on IBM LinuxONE

---

### 1. Executive Summary & Architecture Overview

The **CareerSynk Impact API** exposes the M1–M5 candidate career readiness scoring and counterfactual skill-gap ranking engines via a minimal, high-performance REST microservice. It is engineered specifically for deployment on **IBM LinuxONE (s390x architecture)** and standard **x86_64 (amd64)** Linux environments.

#### Key Architectural Characteristics:
- **Framework**: FastAPI (Pydantic v2 schemas) running on Uvicorn ASGI server.
- **Zero Network Latency**: Zero external API calls at inference time. All inference runs directly in memory against pre-compiled artifacts (`skill_graph.pkl` and `baseline_relevance_model.pkl`).
- **Memory Footprint**: Measured at **276.55 MB** upon startup and **279.51 MB** under active inference load, comfortably satisfying the **<350 MB RAM** constraint for dense LinuxONE LPAR container consolidation.
- **Strict Scope Discipline**: Exposes only three endpoints (`/health`, `/api/v1/readiness`, `/api/v1/counterfactual-ranking`). No superfluous database or auth layers.
- **Mandatory Non-Causal Framing**: Every response payload includes verbatim statistical framing disclaimers.

---

### 2. Framing the Crosswalk Coverage Number (Weakness #2 Disclosure)

> **Official Crosswalk Architecture & Boundary Disclosure**:
> 
> The CareerSynk heterogeneous skill graph comprises **17,479 total nodes** (13,530 unique skill competencies and 3,949 occupations across O*NET and ESCO) and **158,179 requirement and relationship edges**.
>
> Within this unified graph, exactly **333 high-confidence semantic bridge edges** connect the 35 broad O*NET constructs to granular ESCO competencies (top-10 per construct selected at cosine similarity $\ge 0.45$, boosted by ESCO hierarchical parent-child structure). 
>
> The API operates natively over the **full graph**; however, cross-taxonomy skill unification and transferability credit only apply where a high-confidence mapping exists. We intentionally refrain from forcing low-confidence matches or full-corpus lexical merges to prevent false-friend associations (e.g. mapping "Speaking" to "Lipspeaking" or "Time Management" to "Waste Management").

---

### 3. Docker Image Build & Multi-Architecture Specifications

#### Base Image Selection
- **Selected Base Image**: `python:3.13-slim` (verified official Docker Hub manifest containing multi-arch support for `linux/s390x`, `linux/amd64`, `linux/arm64`, and `linux/ppc64le`).
- **User Security**: Runs as an unprivileged, non-root user (`appuser`, UID 1000).
- **Artifact Protection**: Only `src/`, `data/processed/*.pkl`, and `requirements.txt` are packaged. `data/raw/` is strictly excluded via `.dockerignore`.
- **CPU-Only PyTorch**: PyTorch is constrained to CPU wheels (`--extra-index-url https://download.pytorch.org/whl/cpu`) to prevent multi-gigabyte CUDA bloat and ensure seamless compilation on s390x.

#### Build Commands

##### A. Local x86_64 (amd64) Build:
```bash
docker build -t careersynk-impact-api:latest .
```

##### B. Multi-Architecture Build (s390x + amd64) via Docker Buildx:
```bash
# 1. Initialize and switch to a multi-platform buildx builder
docker buildx create --name linuxone-builder --use
docker buildx inspect --bootstrap

# 2. Build multi-arch image targeting IBM LinuxONE (s390x) and x86_64
docker buildx build \
  --platform linux/s390x,linux/amd64 \
  -t careersynk-impact-api:v1.0 \
  --load .
```

---

### 4. Verification Boundary: What Was Tested vs. Organizer Requirements

In accordance with strict technical transparency and truth-in-reporting guidelines, the verification status of this milestone is documented below:

| Component / Layer | Status | Verification Evidence / Mechanism |
| :--- | :---: | :--- |
| **FastAPI REST Application** | **VERIFIED** | 6/6 API unit & integration tests passing (`tests/test_api.py`). Live Uvicorn server verified with live HTTP queries. |
| **Sensitivity Analysis** | **VERIFIED** | Evaluated on $\alpha \in [0.2, 0.5, 0.8]$ across 4 synthetic profiles and 2 target occupations ($J = 0.65$). |
| **Full Regression Suite** | **VERIFIED** | 38/38 pytest tests passing across all milestones M1–M7 (`pytest -v`). |
| **Memory Footprint Target** | **VERIFIED** | Measured at **279.51 MB** under live inference (target was <350 MB). |
| **Docker Build (amd64)** | **PREPARED** | Dockerfile and `.dockerignore` written to production standards. (Local Windows host lacks native Docker daemon). |
| **Docker Build (linux/s390x)** | **PENDING LINUXONE ACCESS** | Base image manifest confirmed for `s390x` on Docker Hub. Actual binary execution on s390x silicon requires organizer-provided LinuxONE LPAR / Community Cloud access. |

#### Manual Steps Remaining for Organizer / LinuxONE Environment:
1. Log into the IBM LinuxONE Community Cloud or assigned s390x VM instance.
2. Clone repository: `git clone <repo-url> && cd careersynk-impact`
3. Ensure processed model artifacts exist in `data/processed/`:
   - `data/processed/skill_graph.pkl`
   - `data/processed/baseline_relevance_model.pkl`
4. Build the container on the s390x host:
   ```bash
   docker build -t careersynk-impact-api .
   ```
5. Run the container:
   ```bash
   docker run -d \
     --name careersynk-api \
     -p 8000:8000 \
     --memory="512m" \
     --cpus="1.0" \
     careersynk-impact-api
   ```
6. Monitor memory via Docker stats:
   ```bash
   docker stats careersynk-api
   ```

---

### 5. API Endpoints & Production Usage Examples

#### 1. Health & Readiness Probe
- **Endpoint**: `GET /health`
- **Purpose**: Verifies that the container is healthy and all ML models and graphs are loaded in memory.
```bash
curl -X GET http://localhost:8000/health
```
**Example Response (`200 OK`)**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "graph_loaded": true
}
```

---

#### 2. Candidate Career Readiness Scoring
- **Endpoint**: `POST /api/v1/readiness`
- **Purpose**: Calculates normalized readiness score ($[0.0, 1.0]$) for a candidate against a target occupation using the formula $W_s = \alpha \cdot w(O, s) + (1 - \alpha) \cdot \hat{r}(O, s)$ ($\alpha = 0.50$).
```bash
curl -X POST http://localhost:8000/api/v1/readiness \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_skills": ["Python", "Software Engineering", "Git", "SQL"],
    "occupation_id": "15-1252.00"
  }'
```
**Example Response (`200 OK`)**:
```json
{
  "occupation_id": "occ:onet:15-1252.00",
  "occupation_title": "Software Developers",
  "baseline_score": 0.0824,
  "skills_evaluated_count": 35,
  "disclaimer": "DISCLAIMER: These are model-based predicted associations and score deltas, not causal claims about employability, hiring probability, or salary outcomes. The underlying model'\''s accuracy is bounded by the M4 evaluation metrics (MAE 0.068, mean Spearman 0.88), and occupation-specific accuracy may vary."
}
```

---

#### 3. Counterfactual Skill-Gap Ranking
- **Endpoint**: `POST /api/v1/counterfactual-ranking`
- **Purpose**: Simulates single-skill interventions, strictly excluding skills already held by the candidate, and ranks missing competencies by predicted marginal readiness increase ($+\Delta$).
```bash
curl -X POST http://localhost:8000/api/v1/counterfactual-ranking \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_skills": ["Python", "Git"],
    "occupation_id": "15-1252.00",
    "top_n": 3
  }'
```
**Example Response (`200 OK`)**:
```json
{
  "occupation_id": "occ:onet:15-1252.00",
  "occupation_title": "Software Developers",
  "baseline_score": 0.0,
  "ranked_gaps": [
    {
      "skill_name": "Systems Analysis",
      "hypothetical_score": 0.0447,
      "predicted_delta": 0.0447,
      "requirement_weight": 0.5891,
      "explanation": "Substantial operational requirement (weight: 0.589). Acquiring this competency directly closes a primary qualification gap (+4.47% readiness)."
    },
    {
      "skill_name": "Systems Evaluation",
      "hypothetical_score": 0.0443,
      "predicted_delta": 0.0443,
      "requirement_weight": 0.578,
      "explanation": "Substantial operational requirement (weight: 0.578). Acquiring this competency directly closes a primary qualification gap (+4.43% readiness)."
    },
    {
      "skill_name": "Critical Thinking",
      "hypothetical_score": 0.0404,
      "predicted_delta": 0.0404,
      "requirement_weight": 0.7272,
      "explanation": "Substantial operational requirement (weight: 0.727). Acquiring this competency directly closes a primary qualification gap (+4.04% readiness)."
    }
  ],
  "disclaimer": "DISCLAIMER: These are model-based predicted associations and score deltas, not causal claims about employability, hiring probability, or salary outcomes. The underlying model'\''s accuracy is bounded by the M4 evaluation metrics (MAE 0.068, mean Spearman 0.88), and occupation-specific accuracy may vary."
}
```

---

#### 4. Error Handling: Unknown Occupation (`404 Not Found`)
```bash
curl -X POST http://localhost:8000/api/v1/readiness \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_skills": ["Python"],
    "occupation_id": "99-9999.99"
  }'
```
**Response (`404 Not Found`)**:
```json
{
  "detail": "Occupation ID '99-9999.99' was not found in the graph. Please provide a valid O*NET SOC Code (e.g. '15-1252.00') or full node identifier (e.g. 'occ:onet:15-1252.00')."
}
```
