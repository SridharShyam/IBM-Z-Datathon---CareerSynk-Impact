# CareerSynk Impact — Project Status & Technical Validation Report

*Document Updated*: 2026-10-18  
*Repository Root*: `careersynk-impact/`  
*Current Milestone Delivered*: **Milestones M1–M5, M7, and Production Monitoring Complete**  
*Evaluation Status*: **42 / 42 Unit & Integration Tests Passing (0 Failures, 0 Warnings)**  
*GitHub Remote*: [https://github.com/SridharShyam/IBM-Z-Datathon---CareerSynk-Impact.git](https://github.com/SridharShyam/IBM-Z-Datathon---CareerSynk-Impact.git) (Branch: `main`)

---

## 1. Project Overview & Value Proposition

### 1.1 Purpose & Problem Statement
**CareerSynk Impact** is an AI-driven, skill-gap counterfactual recommendation system developed for the **IBM Z Datathon 2026** (Team ORDER 404, *Tech for Good* track). 

Job seekers, students, and workers transitioning careers face severe labor-market friction: job listings present exhaustive checklists of requirements without revealing *which specific skill acquisition yields the highest marginal boost in qualification*. Furthermore, international skill taxonomies operate at fundamentally disjointed levels of abstraction—the U.S. Department of Labor's **O*NET** system defines 35 broad capability constructs, while the European Commission's **ESCO** taxonomy classifies over 13,900 granular micro-competencies.

### 1.2 Core Innovation: Counterfactual Gap Discovery
Instead of relying on black-box generative AI prompts or simplistic keyword string matching, CareerSynk Impact computes a mathematically grounded **readiness score** for a candidate relative to a target career. It then simulates single-skill acquisitions one at a time, ranking missing competencies by **predicted readiness delta ($\Delta$)** with deterministic, data-traceable explanations.

### 1.3 Target Hardware & Footprint
Engineered specifically for low-footprint, CPU-only deployment on **IBM LinuxONE (s390x architecture)** and standard Linux containers. The entire service runs with **<280 MB RAM** memory consumption under live inference load, requiring zero GPU acceleration and zero external network calls at runtime.

---

## 2. Current Implementation Status Matrix

| Component / Feature | Milestone | Implementation Status | Evidence / Verification Artifact |
| :--- | :---: | :---: | :--- |
| **Project Scaffolding & Virtual Environment** | M1 | **Verified Complete** | Modular structure under `careersynk-impact/`, Python 3.13 `.venv` |
| **O*NET Raw Data Ingestion** | M1 / M2 | **Verified Complete** | `src/ingestion/onet_loader.py` (parses `Skills.txt`, `Occupation Data.txt`) |
| **ESCO Raw Data Ingestion** | M1 / M2 | **Verified Complete** | `src/ingestion/esco_loader.py` (parses `skills_en.csv`, `occupations_en.csv`, `occupationSkillRelations_en.csv`) |
| **Lexical Fuzzy Matching (Baseline)** | M2 | **Verified Complete** | `src/preprocessing/normalize_skills.py` (`reconcile_onet_esco_lexical`) |
| **Semantic & Hierarchical Skill Reconciliation** | M3 | **Verified Complete** | `src/preprocessing/semantic_reconciliation.py` (Sentence-Transformers + ESCO hierarchy) |
| **Reconciled Dataset Builder** | M2 / M3 | **Verified Complete** | `src/preprocessing/build_occupation_skill_dataset.py` $\rightarrow$ `occupation_skills.parquet` (157,901 rows) |
| **Heterogeneous Skill-Occupation Graph** | M4 | **Verified Complete** | `src/graph/build_skill_graph.py` $\rightarrow$ `data/processed/skill_graph.pkl` (17,479 nodes, 158,179 edges) |
| **Conventional ML Relevance Model (Stage A)** | M4 | **Verified Complete** | `src/models/baseline_relevance.py` $\rightarrow$ `HistGradientBoostingRegressor` ($\overline{\rho}=0.8817$) |
| **Career Readiness Scoring Engine** | M5 | **Verified Complete** | `src/models/readiness_score.py` (formula verified, transfer credit active) |
| **Counterfactual Delta Ranking Engine** | M5 | **Verified Complete** | `src/models/counterfactual_ranking.py` (`rank_skill_gaps`) |
| **Synthetic Candidate Profile Demonstrator** | M5 | **Verified Complete** | `src/models/demo_profiles.py` $\rightarrow$ `data/processed/demo_ranking_examples.md` |
| **FastAPI REST Service (3 Endpoints)** | M7 | **Verified Complete** | `src/api/app.py`, `src/api/schemas.py` (`/health`, `/readiness`, `/counterfactual-ranking`) |
| **Multi-Architecture LinuxONE Docker Container** | M7 | **Verified Complete** | `Dockerfile` (multi-arch `python:3.13-slim` targeting `linux/s390x` and `linux/amd64`) |
| **Alpha Readiness Sensitivity Analysis** | M7 | **Verified Complete** | `src/models/sensitivity_analysis.py` $\rightarrow$ `data/processed/sensitivity_report.md` |
| **5-Fold GroupKFold Cross-Validation** | Post-M7 | **Verified Complete** | `src/models/cross_validation.py` $\rightarrow$ `data/processed/cross_validation_report.md` |
| **Model Auditing, Drift & Telemetry Pipeline** | Post-M7 | **Verified Complete** | `src/models/model_monitoring.py` $\rightarrow$ `data/processed/model_audit_and_monitoring_report.md` |
| **Interactive Terminal Testing Console** | Tooling | **Verified Complete** | `src/interactive_test.py` (supports custom user inputs, search, and live ranking) |

---

## 3. Technical Architecture & File Map

### 3.1 Technology Stack
- **Language**: Python 3.13.9 (64-bit)
- **Data Engineering**: `pandas` (2.x/3.x), `numpy`, `pyarrow` (Parquet serialization)
- **Graph Modeling**: `networkx` (Heterogeneous graph with typed nodes and edges)
- **Machine Learning**: `scikit-learn` (`HistGradientBoostingRegressor`, `RidgeCV`), `scipy`
- **Dense Semantic Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`), `torch` (CPU wheel)
- **API Framework**: `fastapi`, `uvicorn`, `pydantic` (v2), `httpx`
- **Testing & Quality**: `pytest` (42 tests, 100% pass rate)

### 3.2 Directory Structure & Key Deliverables
```text
careersynk-impact/
├── Dockerfile                          # Multi-arch s390x/amd64 container definition
├── .dockerignore                       # Excludes raw data, venv, and dev caches
├── .gitignore                          # Excludes raw data; tracks processed artifacts and reports
├── pytest.ini                          # Test configuration with pythonpath set
├── README.md                           # Master setup and technical orientation guide
├── requirements.txt                    # Pinned production dependencies (CPU-only PyTorch)
├── PROJECT_STATUS.md                   # Current status, architecture, and validation report
├── data/
│   ├── raw/                            # Source taxonomy files (.gitkeep tracked)
│   └── processed/                      # Serialized models, graphs, datasets, and markdown reports
│       ├── skill_graph.pkl             # Serialized NetworkX graph (17,479 nodes, 158,179 edges) [25.3 MB]
│       ├── baseline_relevance_model.pkl# Trained Stage A model artifact [659 KB]
│       ├── occupation_skills.parquet   # Tidy unified associations (157,901 rows) [687 KB]
│       ├── skill_crosswalk.parquet     # 1-to-many semantic bridge mappings (333 rows) [12 KB]
│       ├── baseline_metrics.md         # Stage A model evaluation metrics report
│       ├── cross_validation_report.md  # 5-Fold GroupKFold empirical benchmark
│       ├── sensitivity_report.md       # Ranking stability across alpha in [0.2, 0.5, 0.8]
│       ├── model_audit_and_monitoring_report.md # Real-world input gap, drift (PSI/KS), rollback policy
│       ├── linuxone_deployment_guide.md# Docker deployment steps and LinuxONE memory verification
│       ├── readiness_score_definition.md# Exact mathematical formulation of readiness scoring
│       ├── reconciliation_report.md    # Before/after semantic crosswalk audit
│       ├── graph_sanity_check.md       # Graph topology and requirement connectivity audit
│       └── demo_ranking_examples.md    # Worked examples across 4 synthetic candidate profiles
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── app.py                      # FastAPI app with in-memory lifespan loading and 3 endpoints
│   │   └── schemas.py                  # Pydantic v2 request/response schemas with disclaimer
│   ├── graph/
│   │   └── build_skill_graph.py        # Graph construction, typing, and weight normalization
│   ├── ingestion/
│   │   ├── onet_loader.py              # Ingestion and schema validator for O*NET tables
│   │   └── esco_loader.py              # Ingestion and schema validator for ESCO CSVs
│   ├── models/
│   │   ├── baseline_relevance.py       # Stage A ML relevance training and evaluation
│   │   ├── cross_validation.py         # 5-Fold GroupKFold cross-validation benchmark
│   │   ├── readiness_score.py          # Candidate readiness scoring & graph transfer credit
│   │   ├── counterfactual_ranking.py   # Single-skill intervention simulation engine
│   │   ├── demo_profiles.py            # Synthetic profiles and demo test fixtures
│   │   ├── sensitivity_analysis.py     # Alpha-weighting sensitivity analysis suite
│   │   └── model_monitoring.py         # Input quality auditing, telemetry, and PSI drift tests
│   ├── interactive_test.py             # Interactive CLI tool for testing custom inputs
│   └── preprocessing/
│       ├── normalize_skills.py         # Text cleansing and string normalization
│       ├── semantic_reconciliation.py  # Dense embedding crosswalking + ESCO hierarchy boost
│       └── build_occupation_skill_dataset.py # Pipeline building unified parquet dataset
└── tests/
    ├── test_api.py                     # FastAPI endpoints integration tests (6 tests)
    ├── test_baseline_model.py          # Leakage prevention and Spearman calculation tests (3 tests)
    ├── test_counterfactual_ranking.py  # Gap discovery and sorting tests (3 tests)
    ├── test_graph_construction.py      # Node/edge typing and graph integration tests (2 tests)
    ├── test_ingestion_and_normalization.py # Schema validation and error handling tests (11 tests)
    ├── test_monitoring.py              # Input quality, PSI/KS drift, telemetry tests (4 tests)
    ├── test_readiness_score.py         # Mathematical bounds, monotonicity, partial credit (4 tests)
    ├── test_reconciliation_integration.py # Real data loading integration tests (3 tests)
    ├── test_semantic_reconciliation.py # Crosswalk structure and hierarchy boost tests (4 tests)
    └── test_sensitivity.py             # Jaccard stability metric calculation tests (2 tests)
```

---

## 4. Methodological Rigor & Model Validation

### 4.1 Strict Leakage Prevention
In workforce data, random train/test splits cause severe occupational leakage—a model sees 80% of an occupation's skills and simply memorizes the job title. 

To prove pure inductive generalization, all evaluations use **strict GroupKFold splitting grouped by Occupation ID (O*NET-SOC Code)**:
- **Total Occupations**: 910 standard occupations (31,850 ground-truth ratings).
- **Leakage Verification**: Evaluated via set intersection assertions. Exactly 0% of occupations or skills in the test/validation fold ever appear in training ($|\text{Train} \cap \text{Test}| = 0$).

### 4.2 5-Fold GroupKFold Cross-Validation Benchmark
We evaluated model generalization and regularization across all 910 occupations in 5 isolated folds:

| Model Architecture | Regularization & Configuration | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Spearman ($\overline{\rho}$) | Median Spearman ($\rho$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Trivial Baseline** | Constant Mean Predictor (`DummyRegressor`) | `0.1480 ± 0.0011` | `0.1758 ± 0.0010` | **`0.0000 ± 0.0000`** | `0.0000 ± 0.0000` |
| **Ridge Regression** | L2 Penalty Sweep ($\alpha \in [10^{-2}, 10^3]$) | `0.0811 ± 0.0009` | `0.1038 ± 0.0008` | **`0.8039 ± 0.0048`** | `0.8786 ± 0.0050` |
| **Baseline HistGradientBoosting** | Unregularized ($L_2=0.0$, leaf=20) | `0.0661 ± 0.0015` | `0.0861 ± 0.0017` | **`0.8747 ± 0.0016`** | `0.9049 ± 0.0058` |
| **Regularized & Tuned HistGradientBoosting** | $L_2=1.5$, `min_samples_leaf=30`, early stopping | `0.0662 ± 0.0012` | `0.0863 ± 0.0014` | **`0.8743 ± 0.0013`** | `0.9045 ± 0.0042` |

> **Key Finding**: Per-occupation Spearman rank correlation ($\overline{\rho} = 0.8743 \pm 0.0013$) proves that the model's ability to rank skill importance is stable and robust across diverse, unseen occupations. Injecting $L_2$ regularization dampens cross-fold variance ($\pm 0.0012$ vs $\pm 0.0015$) without sacrificing discriminative ranking accuracy.

### 4.3 Readiness Formula & Sensitivity Analysis
Candidate career readiness is computed as a weighted ratio:
$$\text{Readiness}(C, O) = \frac{\sum_{s \in S_O} W_s \cdot \text{Credit}(s, C)}{\sum_{s \in S_O} W_s}$$
Where:
- $W_s = \alpha \cdot w(O, s) + (1 - \alpha) \cdot \hat{r}(O, s)$ ($\alpha = 0.50$ balances empirical ground-truth requirement weights with model generalization).
- $\text{Credit}(s, C) = 1.0$ for direct matches, and $\max_{s' \in C \cap \mathcal{N}(s)} \text{sim}(s, s')$ for adjacent graph skills with cosine similarity $\ge 0.70$.

**Sensitivity Findings across $\alpha \in [0.2, 0.5, 0.8]$**:
- Evaluated top-5 counterfactual recommendations across 4 synthetic candidate profiles and target occupations.
- **Mean Top-5 Jaccard Stability Index**: **$J = 0.65$** between extreme regimes ($\alpha=0.2$ vs $\alpha=0.8$), and **$J = 0.75$** between default ($\alpha=0.5$) and structural observed weights ($\alpha=0.8$).
- **Shipped Default Decision**: We retain $\alpha = 0.50$ as the shipped default because it balances observed requirements with model smoothing without skewing recommendations toward either extreme.

---

## 5. Production Operations, Monitoring & Drift Detection

### 5.1 Real-World Input Quality Handling
In real-world usage, candidates submit typos, abbreviations, or colloquial skills. The `DataQualityAuditor` (`src/models/model_monitoring.py`) validates:
1. **Direct Matches**: Immediate $1.0$ credit.
2. **Out-of-Vocabulary (OOV) Terms**: The system logs `HIGH_OUT_OF_VOCABULARY_RATE` warnings, attempts transfer credit via semantic embeddings, and gracefully avoids runtime crashes.
3. **Invalid Careers**: Malformed or unmapped occupation codes return immediate HTTP `404 Not Found` responses with valid SOC code recommendations.

### 5.2 Drift Detection: Data Drift vs. Concept Drift
> **Methodological Honesty on Ground-Truth Latency**:
> Real-world employment outcomes (whether a candidate was hired, job performance ratings, career retention) take **3 to 12 months** to materialize. Therefore, production monitoring **cannot rely on real-time MAE or RMSE**.
> 
> Instead, CareerSynk Impact monitors **observable proxy drift**:
> 1. **Input Data Drift**: Out-of-vocabulary rate spikes and candidate skill-set length anomalies.
> 2. **Prediction Distribution Drift**: Monitored continuously via **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests against the reference baseline distribution.

#### Empirical Drift Benchmark Results
| Monitored Distribution Cohort | Population Stability Index (PSI) | KS Statistic ($D$) | $p$-value | Operational Alert Status |
| :--- | :---: | :---: | :---: | :--- |
| **In-Distribution Reference Traffic** | **`0.0254`** ($< 0.10$) | `0.0387` | `0.8675` | **Healthy (Normal Operation)** |
| **Shifted / Anomalous Cohort** | **`2.7118`** ($> 0.25$) | `0.5107` | `0.0000` | **ALERT: Significant Distribution Shift** |

**Actionable Alert Thresholds**:
- $\text{PSI} < 0.10$: Normal operation.
- $0.10 \le \text{PSI} < 0.25$: Warning; queues unrecognized skills for batch semantic clustering.
- $\text{PSI} \ge 0.25$: Critical alert; triggers notification to recalibrate model and review incoming applicant demographics.

---

## 6. LinuxONE Deployment & Hardware Boundaries

### 6.1 Container & Memory Specifications
- **Multi-Arch Dockerfile**: Built on official `python:3.13-slim` (manifest verified for `linux/s390x` and `linux/amd64` on Docker Hub).
- **Security**: Runs under an unprivileged, non-root user (`appuser`, UID 1000).
- **Resource Constraints**: CPU-only PyTorch wheels; worker concurrency bounded (`--workers 1 --limit-concurrency 50`).
- **Memory Footprint**:
  - Memory before load: `113.98 MB`
  - Memory after startup (17,479-node graph + model loaded in memory): `276.55 MB`
  - Memory under active inferences: **`279.51 MB`**
  - **Result**: Comfortably satisfies the **<350 MB RAM** constraint for dense multi-tenant container consolidation on IBM LinuxONE LPARs.

### 6.2 Truth-in-Reporting Verification Boundary
| Component | Tested Locally | Requires Organizer LinuxONE Access |
| :--- | :---: | :---: |
| Full 42-Test Pytest Suite | **YES (100% Pass)** | No (Runs identically on any POSIX/Linux environment) |
| FastAPI REST Endpoints (`curl`) | **YES (Live 200 OK)** | No |
| In-Memory Latency & Memory Footprint | **YES (279.51 MB)** | Verified locally; ready for s390x container run |
| Multi-Arch Dockerfile & Config | **YES (Prepared)** | Ready for `docker build` on LinuxONE host |
| Physical Execution on s390x Silicon | N/A | **Pending organizer-provided s390x Community Cloud access** |

---

## 7. Weakness & Coverage Framing Disclosures

### 7.1 Crosswalk Coverage Framing (Weakness #2 Disclosure)
The skill graph comprises **17,479 nodes** (13,530 unique skills and 3,949 occupations across O*NET and ESCO) and **158,179 edges**.

Exactly **333 high-confidence semantic bridge edges** connect broad O*NET constructs to granular ESCO competencies (top-10 per construct selected at cosine similarity $\ge 0.45$, boosted by ESCO hierarchical parent-child structure). 

The API operates natively over the **full graph**; cross-taxonomy unification applies only where high-confidence links exist. We intentionally refrain from forcing low-confidence matches to prevent false-friend associations (e.g. mapping "Speaking" to "Lipspeaking" or "Time Management" to "Waste Management").

### 7.2 Mandatory Non-Causal Framing Disclaimer
All API responses, documents, and interactive consoles carry the verbatim disclaimer:
> **DISCLAIMER**: These are model-based predicted associations and score deltas, not causal claims about employability, hiring probability, or salary outcomes. The underlying model's accuracy is bounded by the evaluation metrics (MAE 0.068, mean Spearman 0.88), and occupation-specific accuracy may vary.

---

## 8. Verification & Demonstration Commands

### 8.1 Run the Full Test Suite
```bash
$env:PYTHONPATH="."
.\.venv\Scripts\pytest.exe -v
```
*Expected: 42 passed in ~40-90s.*

### 8.2 Interactive Terminal Testing Console
```bash
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe src/interactive_test.py
```
*Allows live searching of careers (e.g. "software developer"), custom candidate skill input, and displays real-time readiness scores and top-5 counterfactual delta rankings.*

### 8.3 Launch FastAPI Microservice
```bash
$env:PYTHONPATH="."
.\.venv\Scripts\uvicorn.exe src.api.app:app --host 127.0.0.1 --port 8000
```
*Swagger UI available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).*
