# CareerSynk Impact — Project Status & Handover Document

*Document Generated*: 2026-10-17  
*Repository Root*: `careersynk-impact/`  
*Current Milestone Delivered*: **Milestone M5 Complete**  
*Evaluation Status*: **30 / 30 Unit & Integration Tests Passing (0 Failures, 0 Warnings)**

---

## 1. Project Overview

### 1.1 Purpose & Problem Statement
**CareerSynk Impact** is an AI-driven, skill-gap counterfactual recommendation system developed for the **IBM Z Datathon 2026** (Team ORDER 404, *Tech for Good* track). 

Job seekers, students, and career-transitioning workers face significant labor-market asymmetry: job descriptions list dozens of disjointed requirements without clarifying *which specific skills produce the highest marginal boost in qualification*. Meanwhile, international taxonomies describe occupations at fundamentally discordant abstraction levels—the U.S. Department of Labor's **O*NET** system defines broad, macro-level capability constructs (35 core skills), whereas the European Commission's **ESCO** classification defines over 13,900 granular, task-oriented micro-competencies.

### 1.2 Target Users & Expected Impact
- **Career Switchers & Adult Learners**: Gain immediate visibility into which 2–3 skill investments yield the greatest mathematical jump in role readiness.
- **Workforce Development Agencies & Educators**: Align vocational curricula against empirical labor requirements rather than subjective heuristics.
- **Target Deployment Platform**: Designed for CPU-efficient inference and eventual containerized deployment on **IBM LinuxONE** via FastAPI.

### 1.3 Core Idea & Value Proposition
- **Counterfactual Skill-Gap Simulation**: Instead of static keyword matching or black-box LLM suggestions, CareerSynk Impact computes a baseline career readiness score for a candidate's profile against a target occupation, hypothetically injects candidate skill acquisitions one at a time, and ranks skill gaps by **predicted readiness delta ($\Delta$)**.
- **Transparent, Deterministic Explanations**: Every recommendation includes a data-traceable explanation derived from composite requirement weights ($W_s$) and semantic graph distance, eliminating generative hallucination risks.

---

## 2. Current Implementation Status

| Component / Feature | Milestone | Implementation Status | Evidence / Artifact |
| :--- | :--- | :--- | :--- |
| **Project Scaffolding & Virtual Environment** | M1 | **Verified Complete** | `src/`, `data/raw/`, `data/processed/`, `.venv/` (Python 3.13) |
| **O*NET Raw Data Ingestion** | M1 / M2 | **Verified Complete** | `src/ingestion/onet_loader.py` (parses `Skills.txt`, `Occupation Data.txt`) |
| **ESCO Raw Data Ingestion** | M1 / M2 | **Verified Complete** | `src/ingestion/esco_loader.py` (parses `skills_en.csv`, `occupations_en.csv`, `occupationSkillRelations_en.csv`) |
| **Lexical Fuzzy Matching (Baseline)** | M2 | **Verified Complete** | `src/preprocessing/normalize_skills.py` (`reconcile_onet_esco_lexical`) |
| **Semantic & Hierarchical Skill Reconciliation** | M3 | **Verified Complete** | `src/preprocessing/semantic_reconciliation.py` (Sentence-Transformers + ESCO hierarchy) |
| **Reconciled Datasets Build Pipeline** | M2 / M3 | **Verified Complete** | `src/preprocessing/build_occupation_skill_dataset.py` (CLI `--method {semantic,lexical}`) |
| **Heterogeneous Skill-Occupation Graph** | M4 | **Verified Complete** | `src/graph/build_skill_graph.py` $\rightarrow$ `data/processed/skill_graph.pkl` (17,479 nodes) |
| **Conventional ML Relevance Model (Stage A)** | M4 | **Verified Complete** | `src/models/baseline_relevance.py` $\rightarrow$ `HistGradientBoostingRegressor` ($\overline{\rho}=0.8817$) |
| **Career Readiness Scoring Engine** | M5 | **Verified Complete** | `src/models/readiness_score.py` (formula verified, partial credit active) |
| **Counterfactual Delta Ranking Engine** | M5 | **Verified Complete** | `src/models/counterfactual_ranking.py` (`rank_skill_gaps`) |
| **Synthetic Candidate Profile Demo** | M5 | **Verified Complete** | `src/models/demo_profiles.py` $\rightarrow$ `data/processed/demo_ranking_examples.md` |
| **Stage C Graph Neural Network (GNN)** | M6 | **Not Started** | Gated on evaluating whether GNN out-competes Stage A ML baseline ($\overline{\rho}=0.8817$) |
| **FastAPI REST Service on IBM LinuxONE** | M7 | **Not Started** | `src/api/` folder scaffolded; endpoints pending implementation |

---

## 3. Technical Architecture

### 3.1 Technology Stack
- **Language**: Python 3.13.9 (64-bit)
- **Data Engineering**: `pandas` (3.0.6), `numpy` (2.5.4), `pyarrow` (26.0.0), `openpyxl` (3.1.5)
- **Graph Modeling**: `networkx` (3.7)
- **Classical Machine Learning**: `scikit-learn` (1.9.1), `scipy` (1.18.1)
- **Dense Semantic Embeddings**: `sentence-transformers` (6.1.0), `transformers` (5.20.1), `torch` (2.14.1+cpu)
- **String Alignment**: `rapidfuzz` (3.14.6)
- **Testing & Quality Assurance**: `pytest` (9.1.1)

### 3.2 Directory Structure & Key Files
```text
careersynk-impact/
├── .gitignore                          # Excludes raw/processed data, models, and .venv
├── pytest.ini                          # Test runner config with pythonpath and custom markers
├── README.md                           # Setup and developer documentation
├── requirements.txt                    # Project dependency specification
├── data/
│   ├── raw/                            # Uncommitted source files (.gitkeep tracked)
│   │   ├── Skills.txt                  # O*NET 31.0 Skills table (63,700 rows)
│   │   ├── Occupation Data.txt         # O*NET 31.0 Occupations (1,016 rows)
│   │   ├── skills_en.csv               # ESCO v1.1.2 Skills (13,960 concepts)
│   │   ├── occupations_en.csv          # ESCO v1.1.2 Occupations (3,047 concepts)
│   │   ├── occupationSkillRelations_en.csv # ESCO Graph Edges (126,051 relations)
│   │   ├── skillsHierarchy_en.csv      # ESCO 4-level taxonomy hierarchy
│   │   ├── transversalSkillsCollection_en.csv # ESCO 95 transversal skills
│   │   └── broaderRelationsSkillPillar_en.csv # ESCO ancestor parent-child links
│   └── processed/                      # Preprocessed datasets and serialized models
│       ├── occupation_skills.parquet   # Unified tidy occupation-skill associations (157,901 rows)
│       ├── skill_crosswalk.parquet     # 1-to-many semantic crosswalk (333 mappings)
│       ├── skill_graph.pkl             # Serialized NetworkX heterogeneous graph (17,479 nodes)
│       ├── baseline_relevance_model.pkl# Trained Stage A HistGradientBoosting artifact
│       ├── reconciliation_report.md    # Audit report comparing lexical vs semantic mapping
│       ├── graph_sanity_check.md       # Graph topology and 3-occupation requirement inspection
│       ├── baseline_metrics.md         # Stage A model evaluation metrics report
│       ├── readiness_score_definition.md # Mathematical formulation of readiness scoring
│       └── demo_ranking_examples.md    # Counterfactual simulations across 4 synthetic profiles
├── src/
│   ├── __init__.py
│   ├── api/                            # Future FastAPI endpoints (Stage M7)
│   ├── graph/
│   │   └── build_skill_graph.py        # Graph construction, typing, and unification
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── onet_loader.py              # Ingestion and schema validator for O*NET tables
│   │   └── esco_loader.py              # Ingestion and schema validator for ESCO CSVs
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_relevance.py       # Stage A ML relevance model training and evaluation
│   │   ├── readiness_score.py          # Candidate readiness scoring & graph partial credit
│   │   ├── counterfactual_ranking.py   # One-at-a-time hypothetical simulation engine
│   │   └── demo_profiles.py            # Synthetic profile definitions and demonstration runner
│   └── preprocessing/
│       ├── __init__.py
│       ├── normalize_skills.py         # String normalization and lexical baseline crosswalk
│       ├── semantic_reconciliation.py  # Dense sentence-transformer + hierarchy crosswalk
│       └── build_occupation_skill_dataset.py # End-to-end dataset builder CLI
├── tests/
│   ├── test_baseline_model.py          # Tests for group-split leakage and ranking metrics
│   ├── test_counterfactual_ranking.py  # Tests for delta ordering and owned-skill exclusion
│   ├── test_graph_construction.py      # Tests for node/edge typing, weights, and unification
│   ├── test_ingestion_and_normalization.py # Unit tests for loaders and string cleaners
│   ├── test_readiness_score.py         # Tests for mathematical bounds and directional credit
│   ├── test_reconciliation_integration.py # Integration tests for real raw data loading
│   └── test_semantic_reconciliation.py # Tests for embedding shapes, top-K, and hierarchy boost
└── notebooks/                          # Interactive exploration directory
```

### 3.3 Data Flow & Architecture Pipeline
```
[Raw O*NET & ESCO Data] 
       │
       ▼
[src/ingestion] (onet_loader.py, esco_loader.py)
       │
       ▼
[src/preprocessing] (semantic_reconciliation.py)
       ├── sentence-transformers (all-MiniLM-L6-v2)
       └── ESCO hierarchy & transversal boosting
       │
       ▼
[Processed Parquet Artifacts]
       ├── data/processed/occupation_skills.parquet (157,901 rows)
       └── data/processed/skill_crosswalk.parquet (333 high-confidence mappings)
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
[src/graph/build_skill_graph.py]     [src/models/baseline_relevance.py]
  • 3,949 Occupations                  • GroupShuffleSplit by Occupation
  • 13,530 Skills                      • HistGradientBoostingRegressor
  • 158,179 Typed Edges                • Features: 384-d Embedding + TF-IDF + Interaction
       │                               • Evaluated: Spearman Rho = 0.8817
       │                                        │
       └────────────────────┬───────────────────┘
                            ▼
              [src/models/readiness_score.py]
                • Target Occupation Skills S_O
                • Composite Weights W_s = 0.5*w + 0.5*r_hat
                • Graph Adjacency Partial Credit
                            │
                            ▼
           [src/models/counterfactual_ranking.py]
             • Simulate S_new = C U {s_k}
             • Delta_k = Readiness(S_new) - Readiness(C)
             • Deterministic Template Explanations
                            │
                            ▼
                  [M7: FastAPI on LinuxONE]
```

---

## 4. AI/ML and Core Logic

### 4.1 Dense Semantic Reconciliation
- **Model**: `all-MiniLM-L6-v2` (`sentence-transformers`), 22.7M parameters, 384-dimensional dense vectors.
- **CPU Wall-Clock Time**: Encodes 35 O*NET skills in **0.04s** and all 13,939 ESCO skills in **19.83s** on standard CPU.
- **Hierarchy Integration**: Candidate matches are cross-referenced with `transversalSkillsCollection_en.csv` and `broaderRelationsSkillPillar_en.csv`. ESCO skills in transversal categories or sharing taxonomic root keywords receive a $+0.05$ score boost, resolving prior lexical false friends:
  - *Time Management*: Formerly collided with `waste management` ($83.9\%$). Now correctly grounded in transversal competency `manage time` ($0.8936$) and domain scheduling.
  - *Speaking*: Formerly collided with `lipspeaking` ($84.2\%$). Now correctly maps to `communication` ($0.4735$) and `prepare speeches` ($0.4556$).
  - *Operations Analysis*: Formerly collided with `navy operations` ($82.4\%$). Now correctly maps to `operations management` ($0.7588$) and `business analysis` ($0.6014$).

### 4.2 Stage A Conventional Machine Learning Baseline
- **Model**: `HistGradientBoostingRegressor` (max iterations: 150, learning rate: 0.08, max depth: 6).
- **Target Variable**: Continuous O*NET skill importance score normalized to $[0.0, 1.0]$.
  - *Ground Truth Justification*: O*NET provides verified continuous importance ratings (1.0 - 5.0) for standard occupation profiles. ESCO's binary/coarse ratings (`essential`/`optional`) are held out from ground-truth regression to prevent synthetic scale artifacts.
- **Leakage Prevention**: Evaluated via strict **Group-Split by Occupation** ($80\%$ train, $20\%$ test). No skills from test occupations appear in the training split:
  - Training Occupations: 728 (25,480 pairs)
  - Held-out Test Occupations: 182 (6,370 pairs)
- **Features (449 dimensions)**:
  1. Dense skill sentence embedding: 384 dimensions.
  2. Occupation title & description TF-IDF: 64 dimensions.
  3. Semantic interaction feature: 1 dimension (dot product between skill and occupation embeddings).

### 4.3 Readiness Scoring & Counterfactual Simulation Formula
For candidate skills $C$ and target occupation $O$:
$$\text{Readiness}(C, O) = \frac{\sum_{s \in S_O} W_s \cdot \text{Credit}(s, C)}{\sum_{s \in S_O} W_s}$$
- **Composite Weight**: $W_s = 0.50 \cdot w(O, s) + 0.50 \cdot \hat{r}(O, s)$
- **Partial Credit**: Direct possession $= 1.0$; adjacent graph skill with similarity $\ge 0.70 \implies \text{sim} \times 0.50$; otherwise $0.0$.
- **Counterfactual Delta**: $\Delta_k = \text{Readiness}(C \cup \{s_k\}, O) - \text{Readiness}(C, O)$, strictly calculated for missing skills $s_k \in S_O \setminus C$.

---

## 5. Testing and Quality

### 5.1 Test Execution Results
Execution of `pytest -v` across the entire project produces **30 passed tests in 27.46 seconds with 0 failures and 0 warnings**:

```text
tests/test_baseline_model.py::TestBaselineModelLogic::test_group_split_prevents_occupation_leakage PASSED
tests/test_baseline_model.py::TestBaselineModelLogic::test_per_occupation_spearman_calculation PASSED
tests/test_baseline_model.py::test_real_baseline_model_artifact_and_metrics PASSED
tests/test_counterfactual_ranking.py::TestCounterfactualRankingSynthetic::test_ranking_excludes_already_owned_skills_and_sorts_by_delta PASSED
tests/test_counterfactual_ranking.py::TestCounterfactualRankingSynthetic::test_full_candidate_returns_empty_gap_ranking PASSED
tests/test_counterfactual_ranking.py::test_real_counterfactual_ranking_integration PASSED
tests/test_graph_construction.py::TestGraphConstructionFixtures::test_synthetic_graph_building_and_typing PASSED
tests/test_graph_construction.py::test_real_data_skill_graph_integration PASSED
tests/test_ingestion_and_normalization.py::TestSkillNormalization::test_normalize_basic_casing_and_whitespace PASSED
tests/test_ingestion_and_normalization.py::TestSkillNormalization::test_normalize_punctuation_and_delimiters PASSED
tests/test_ingestion_and_normalization.py::TestSkillNormalization::test_normalize_non_string_input PASSED
tests/test_ingestion_and_normalization.py::TestSkillNormalization::test_reconcile_exact_and_fuzzy_synthetic PASSED
tests/test_ingestion_and_normalization.py::TestONETLoaderErrorHandling::test_missing_file_raises_typed_exception PASSED
tests/test_ingestion_and_normalization.py::TestONETLoaderErrorHandling::test_invalid_schema_raises_format_error PASSED
tests/test_ingestion_and_normalization.py::TestONETLoaderErrorHandling::test_missing_occupations_file_raises_typed_exception PASSED
tests/test_esco_loader_error_handling (skills, occupations, relations) PASSED [4 tests]
tests/test_readiness_score.py (bounds, directional monotonicity, partial credit, real integration) PASSED [4 tests]
tests/test_reconciliation_integration.py (onet load, esco load, taxonomy reconciliation) PASSED [3 tests]
tests/test_semantic_reconciliation.py (hierarchy support, graceful handling, top-K constraints, real data integration) PASSED [4 tests]

============================= 30 passed in 27.46s =============================
```

### 5.2 Model Performance Summary (Held-out Test Occupations)
| Metric | Trivial Mean Baseline | HistGradientBoosting Baseline | Relative Improvement |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | `0.1517` | `0.0676` | **-0.0841** (*55.4% error reduction*) |
| **Root Mean Squared Error (RMSE)** | `0.1797` | `0.0871` | **-0.0926** (*51.5% error reduction*) |
| **Mean Spearman Rank Correlation ($\overline{\rho}$)** | `0.0000` | **`0.8817`** | **+0.8817** (*High rank-order fidelity*) |
| **Median Spearman Rank Correlation** | `0.0000` | **`0.9040`** | **+0.9040** (*Consistently high across roles*) |

---

## 6. Completed Work and Pending Tasks

### 6.1 Completed Work
1. **Scaffolded Independent Repository (M1)**: Set up modular package structure under `careersynk-impact/` with isolated `.venv` and Git tracking.
2. **Robust Multi-Source Ingestion (M2)**: Implemented typed exception handling and schema verification for O*NET 31.0 and ESCO v1.1.2.
3. **Semantic Skill Alignment (M3)**: Built 1-to-many embedding crosswalk with ESCO hierarchy boosting, replacing failure-prone 1-to-1 lexical matching.
4. **Graph Construction (M4)**: Generated typed, weighted heterogeneous graph with 17,479 nodes and 158,179 edges.
5. **Stage A ML Model (M4)**: Validated predictive model with $\overline{\rho} = 0.8817$ on 182 unseen occupations.
6. **Counterfactual Engine & Demo Profiles (M5)**: Implemented mathematically bounded readiness scoring and simulated interventions across 4 synthetic candidate profiles.

### 6.2 Prioritized Remaining Tasks
- **[P0] Milestone M7: FastAPI Service Deployment on IBM LinuxONE**
  - Implement REST endpoints in `src/api/` (`POST /api/v1/readiness`, `POST /api/v1/counterfactual-ranking`, `GET /api/v1/occupations`).
  - Containerize the service (`Dockerfile`) with non-root user execution, CPU-optimized thread allocation, and health checks.
  - Profile synchronous inference latency ($< 50\text{ ms}$ target).
- **[P1] Milestone M6: Stage C GNN Comparison & Ablation**
  - Evaluate whether a Graph Convolutional Network / HeteroGraphSAGE provides meaningful ranking gains over the Stage A baseline ($\overline{\rho} = 0.8817$).
  - If marginal gain is $< 2\%$, formally document the decision to retain the lightweight tabular baseline on LinuxONE to reduce container footprint and memory consumption.
- **[P2] Curated Prerequisite Proxy Graph**
  - Construct directional prerequisite edges ($S_A \rightarrow S_B$) using ESCO Level-0 transversal competency roots or O*NET Job Zone complexity tiers.
- **[P3] Streamlit / Web Demonstration Dashboard**
  - Build an interactive demonstration UI allowing judges to pick occupations, toggle candidate skills, and visualize the counterfactual delta ranking in real time.

---

## 7. Setup and Execution

### 7.1 Prerequisites
- Python 3.10+ (tested on Python 3.13.9 64-bit on Windows).
- Git.
- Raw O*NET and ESCO files placed into `data/raw/` (see Section 3.2).

### 7.2 Installation & Test Instructions
```bash
# 1. Navigate to project root
cd careersynk-impact

# 2. Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run test suite
pytest -v
```

### 7.3 Reproducing Dataset Generation & Simulation
```bash
# Step 1: Build processed datasets (semantic reconciliation default)
python src/preprocessing/build_occupation_skill_dataset.py --method semantic

# Step 2: Build heterogeneous skill-occupation graph
python src/graph/build_skill_graph.py

# Step 3: Train and evaluate Stage A ML baseline
python src/models/baseline_relevance.py

# Step 4: Run counterfactual simulations on synthetic profiles
python -m src.models.demo_profiles
```

---

## 8. Deployment and Repository Readiness

- **Repository Cleanliness**: `.gitignore` strictly protects against committing raw datasets (`data/raw/*`), processed datasets (`data/processed/*`), serialized model artifacts (`*.pkl`), and `.venv/` while maintaining `.gitkeep` placeholders.
- **Secrets & Credentials**: **Zero hardcoded credentials, API keys, or access tokens**. The system uses purely local, offline CPU models and local open datasets.
- **Container Readiness**: The code has zero GPU dependencies (`torch.cuda.is_available() == False`). Memory consumption during inference is under 350 MB, making it ideally suited for IBM LinuxONE s390x containers or standard Linux x86 VMs.

---

## 9. Risks, Limitations, and Improvement Strategies

### 9.1 Technical & Data Limitations
1. **Absence of Directed Prerequisite Sequences**:
   - *Limitation*: O*NET and ESCO state *what* skills an occupation requires, not the chronological sequence in which a human must learn them.
   - *Mitigation*: The system explicitly ranks by **predicted readiness impact ($\Delta$)**, not prerequisite curriculum order.
2. **Taxonomic Granularity Divergence**:
   - *Limitation*: 35 broad O*NET constructs vs. 13,900+ specific ESCO competencies.
   - *Mitigation*: Solved via 1-to-many semantic crosswalk with ESCO hierarchy boosting, retaining up to Top-10 competencies per construct.
3. **ESCO Flat Weighting**:
   - *Limitation*: ESCO only labels skills as `essential` or `optional`.
   - *Mitigation*: Our Stage A ML model predicts continuous relevance scores $\hat{r}(O, s)$, smoothly refining categorical ESCO edges.

### 9.2 Mandatory Non-Causal Modeling Disclaimer
All output artifacts, API responses, and user documentation carry the following mandatory disclaimer:
> **DISCLAIMER**: These are model-based predicted associations and score deltas, not causal claims about employability, hiring probability, or salary outcomes. The underlying model's accuracy is bounded by the M4 evaluation metrics (MAE 0.068, mean Spearman 0.88), and occupation-specific accuracy may vary.

---

## 10. Final Handover Checklist & Next Steps

- [x] **M1**: Project structure, venv, and initial ingestion stubs passing tests.
- [x] **M2**: Real raw file parsing for O*NET and ESCO; dataset serialization.
- [x] **M3**: Semantic sentence-embedding crosswalk with hierarchy boosting; false friends resolved.
- [x] **M4**: Heterogeneous graph construction (17,479 nodes); Stage A ML baseline trained ($\overline{\rho} = 0.8817$).
- [x] **M5**: Mathematical readiness scoring, partial credit, and counterfactual simulation verified across 4 synthetic profiles.
- [x] **Test Verification**: 30/30 tests passing with 0 warnings.
- [ ] **Next Step (M7)**: Expose `src/models/readiness_score.py` and `counterfactual_ranking.py` via FastAPI endpoints in `src/api/` and package into a LinuxONE-compatible Docker container.
