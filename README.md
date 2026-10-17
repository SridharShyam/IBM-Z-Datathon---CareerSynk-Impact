# CareerSynk Impact 🚀
**Skill-Gap Counterfactual Ranking System for Career Readiness**  
*IBM Z Datathon 2026 — Team ORDER 404 (Tech for Good Track)*  
*Target Deployment: IBM LinuxONE via FastAPI*

---

## 1. Project Overview
CareerSynk Impact is an AI-driven career readiness framework that models labor-market skills and occupation pathways. It ingests international skill taxonomies (O\*NET and ESCO) and provides counterfactual recommendations: identifying the minimal, high-impact skills a learner needs to bridge readiness gaps for target roles.

---

## 2. Repository Layout
```text
careersynk-impact/
  data/
    raw/                 # Raw O*NET and ESCO datasets (.gitkeep, uncommitted)
    processed/           # Cleaned and reconciled dataset artifacts (.gitkeep)
  src/
    ingestion/           # Raw file loaders and schema validators (O*NET, ESCO)
    preprocessing/       # Skill name cleaning and taxonomy reconciliation stubs
    graph/               # Heterogeneous skill-job graph builders (M2/M3)
    models/              # Counterfactual ranking and inference models (M3/M4)
    api/                 # FastAPI service endpoints for IBM LinuxONE (M4)
  tests/                 # Pytest test suite
  notebooks/             # Exploratory analysis and prototyping
  requirements.txt       # Project dependencies
  README.md              # Setup and usage guide
  .gitignore             # Git exclusions
```

---

## 3. Environment Setup & Installation

### Prerequisites
- Python 3.10+ (Tested with Python 3.11 / 3.13)
- Windows PowerShell, macOS, or Linux terminal

### Step 1: Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Test Suite
The initial test suite tests normalization and dataset loader error handling using mocks and temporary fixtures, requiring no local raw data downloads:
```bash
pytest -v
```

---

## 4. Populating Raw Datasets
> **Important Note:** In accordance with project design, raw datasets are **not** committed to version control and must be placed manually into `data/raw/` prior to running full ingestion pipelines.

1. **O\*NET Database**:
   - Download the official O\*NET database exports from [O\*NET Resource Center](https://www.onetcenter.org/database.html).
   - Place the database files (e.g. `Skills.txt`) inside `data/raw/`.
   - *Note for local development:* If you have local O\*NET files (such as under `../ONET/`), copy the required files into `data/raw/`.

2. **ESCO Dataset**:
   - Download the ESCO v1.x classification package from [ESCO Download Portal](https://esco.ec.europa.eu/en/use-esco/download).
   - Extract and place `skills_en.csv` and `occupationSkillRelations_en.csv` into `data/raw/`.
   - *Note for local development:* If you have local ESCO files (such as under `../ESCO dataset/`), copy the required CSV files into `data/raw/`.

---

## 5. Upcoming Milestones & Roadmap
- **M2 (Data Preprocessing & Local Mapping):** Ingest raw O\*NET & ESCO tables from `data/raw/`, extract standardized skill and role entities, and serialize clean tables into `data/processed/`.
- **M3 (Taxonomy Reconciliation & Skill Graph):** Implement exact and semantic joins in `reconcile_onet_esco` to build a unified skill-occupation graph.
- **M4 (Counterfactual Ranking & FastAPI on LinuxONE):** Train/evaluate counterfactual ranking algorithm and expose REST API endpoints.
