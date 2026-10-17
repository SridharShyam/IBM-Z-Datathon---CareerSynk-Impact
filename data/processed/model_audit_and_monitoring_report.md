# Engineering Audit & Production Reliability Report
**CareerSynk Impact: Model Generalization, Evaluation Rigor, and Real-World Monitoring**

*Prepared by the CareerSynk Impact Engineering Team*  
*Last Updated: October 2026*

---

## Executive Summary

When building machine learning systems for workforce intelligence and career mobility, high offline benchmark scores can be dangerously misleading. A model that achieves low error on static cross-sectional tables often falls apart when exposed to real human input—such as messy skill descriptions, vocabulary evolution, and unobservable hiring outcomes.

Over the past milestones, we built a heterogeneous graph (17,479 nodes and 158,179 edges) and a Stage A relevance engine (`HistGradientBoostingRegressor`) to evaluate candidate readiness and recommend targeted counterfactual upskilling interventions. 

This audit delivers an unvarnished evaluation of our model across training, validation, testing, and production simulation. We explicitly examine:
1. **Data Leakage & Cross-Validation**: How we proved 0% identity leakage across 910 standard occupations.
2. **The Offline-to-Online Gap**: Why real-world candidate inputs diverge from canonical taxonomies and how our graph handles out-of-vocabulary terms.
3. **Drift Detection Without Immediate Ground Truth**: How to monitor model degradation when downstream hiring signals take 6–12 months to materialize.
4. **Production Safeguards**: Our concrete alerting thresholds, inference telemetry, and zero-downtime rollback procedures.

---

## 1. Leakage Prevention and Cross-Validation Audit

### The Risk of Occupational Leakage
In standard random train/test splits, skills belonging to the same job profile appear in both the training set and the test set. If an algorithm sees 80% of a Software Developer's skill ratings during training, predicting the remaining 20% is essentially memorizing the occupational cluster rather than learning generalizable principles of skill relevance.

### Our Solution: Strict 5-Fold GroupKFold
We grouped our data strictly by **Occupation ID (O*NET-SOC Code)** across all 31,850 continuous ground-truth ratings spanning 910 occupations. Every single occupation exists exclusively in either training or validation within each fold—never both.

```
Total Occupations: 910 (31,850 verified ratings)
Cross-Validation Strategy: 5-Fold GroupKFold
Train per fold: 728 occupations (~25,480 rows)
Validation per fold: 182 occupations (~6,370 rows)
Set Intersection: |Train_Occs ∩ Val_Occs| == 0 (Strict 0% Leakage)
```

### 5-Fold Benchmark Results

We compared our model family across linear regularized baselines and gradient boosted trees to observe how variance and accuracy behave under strict group isolation:

| Model Architecture | Regularization & Configuration | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Spearman Rank ($\overline{\rho}$) | Median Spearman Rank ($\rho$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Trivial Baseline** | Constant Mean Predictor (`DummyRegressor`) | 0.1480 ± 0.0011 | 0.1758 ± 0.0010 | **0.0000 ± 0.0000** | 0.0000 ± 0.0000 |
| **Ridge Regression** | L2 Regularization sweep ($\alpha \in [10^{-2}, 10^3]$) | 0.0811 ± 0.0009 | 0.1038 ± 0.0008 | **0.8039 ± 0.0048** | 0.8786 ± 0.0050 |
| **Baseline HistGradientBoosting** | Unregularized ($L_2=0.0$, leaf=20) | 0.0661 ± 0.0015 | 0.0861 ± 0.0017 | **0.8747 ± 0.0016** | 0.9049 ± 0.0058 |
| **Regularized & Tuned HistGradientBoosting** | $L_2=1.5$, `min_samples_leaf=30`, early stopping | 0.0662 ± 0.0012 | 0.0863 ± 0.0014 | **0.8743 ± 0.0013** | 0.9045 ± 0.0042 |

### What These Numbers Mean in Practice
- **Spearman Rank Correlation ($\overline{\rho}$)** is our primary evaluation metric. In downstream counterfactual ranking, what matters most is whether the model correctly identifies *which* skill provides the highest readiness boost for that specific occupation. Our regularized model achieves a mean ranking correlation of **0.8743** across unobserved occupations, showing robust inductive generalization.
- Adding **$L_2$ regularization ($1.5$)** and raising **`min_samples_leaf` to 30** trimmed fold-to-fold MAE variance from $\pm 0.0015$ down to $\pm 0.0012$, preventing individual tree leaves from over-indexing on idiosyncratic occupational titles.

---

## 2. Bridging the Offline vs. Real-World Input Gap

In academic benchmarks, inputs are pristine. In production, users type whatever comes to mind. They enter buzzwords, typos, and hybrid phrases that do not appear in official O*NET or ESCO catalogs.

### The Danger of Strict String Matching
If a candidate possesses "Python programming" or "Docker containerization" and the system searches solely for exact string matches, the candidate receives 0.0 credit. This causes extreme false negatives and frustrates users.

### How Our Graph Engine Solves This
Our architecture separates evaluation into two complementary layers:
1. **Canonical Exact Match**: Direct match yields 1.0 credit immediately.
2. **Transferable Semantic Aptitude ($\tau_{\text{sim}} \ge 0.70$)**: If a skill is not directly present, the engine explores adjacent competencies in the heterogeneous graph using high-dimensional dense embeddings (`all-MiniLM-L6-v2`). If an adjacent skill passes the similarity threshold, partial credit is awarded proportionally.

### Real-World Audit Cases

We tested our input auditor (`DataQualityAuditor`) against four realistic user submissions:

| Candidate Persona | Submitted Skills | Match Rate | Triggered System Flags | How the System Handles It |
| :--- | :--- | :---: | :--- | :--- |
| **Standard Tech Candidate** | `Python`, `SQL`, `Git`, `Docker` | **100%** | None | Full direct credit; baseline computed instantly. |
| **Colloquial / Typo Candidate** | `pythoon`, `sql querying`, `git hub` | **0% direct** | `HIGH_OUT_OF_VOCABULARY_RATE` | Flags input for logging; awards transfer credit via semantic embeddings rather than failing silently. |
| **Cross-Industry Switcher** | `forklift driving`, `welding`, `inventory` (targeting Software Dev) | **0% relevant** | `HIGH_OUT_OF_VOCABULARY_RATE` | Evaluated against software developer requirements; correctly scores low readiness without runtime crashes. |
| **Malformed Career Target** | `Python` (targeting `occ:unknown:99-9999`) | N/A | `UNKNOWN_OCCUPATION_ID` | Intercepted at API layer; returns clean HTTP 404 with helpful SOC suggestions. |

---

## 3. Drift Monitoring Without Immediate Ground Truth

A critical challenge in career readiness algorithms is **ground-truth latency**. 

When an e-commerce model predicts a click, the ground truth arrives within seconds. When an autonomous driving model detects a lane, ground truth is continuous. But when a workforce platform estimates a candidate's career readiness, ground truth (whether they were actually hired, how quickly they adapted, and whether their salary increased) takes **3 to 12 months** to observe.

### The Consequence: We Cannot Monitor Live MAE/RMSE
Any production system claiming to monitor live prediction accuracy in human capital matching without months of longitudinal follow-up data is making an unsupportable claim.

### Our Strategy: Continuous Distribution & Proxy Monitoring
Instead of waiting for delayed outcome labels, we monitor observable proxy signals in real time:

1. **Input Data Drift**: Spikes in the out-of-vocabulary (OOV) rate or sudden shifts in submitted skill count distributions.
2. **Prediction Distribution Drift**: Tracking shifts in generated readiness scores using the **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests against our established training distribution.

```
                    ┌───────────────────────────────┐
                    │ Live Candidate Request        │
                    └──────────────┬────────────────┘
                                   ▼
                    ┌───────────────────────────────┐
                    │ Data Quality & Schema Audit   │
                    │ - OOV Rate & Skill Length     │
                    └──────────────┬────────────────┘
                                   ▼
                    ┌───────────────────────────────┐
                    │ In-Memory Inference Engine    │
                    │ - Readiness & Counterfactuals │
                    └──────────────┬────────────────┘
                                   ▼
                    ┌───────────────────────────────┐
                    │ Drift & Telemetry Monitor     │
                    │ - Calculate PSI & KS Tests    │
                    │ - Log to inference_telemetry  │
                    └──────────────┬────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
     [PSI < 0.10: Normal]                     [PSI ≥ 0.25: Drift Alert]
   Continue Standard Service             Flag for Vocabulary Ingestion
```

### Empirical Drift Benchmark

We evaluated our drift monitor across normal production traffic versus anomalous traffic (e.g. sudden influx of mismatched resumes):

| Traffic Cohort | Sample Size | Population Stability Index (PSI) | KS Statistic ($D$) | $p$-value | Operational Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Healthy Reference Traffic** | 300 | **0.0254** | 0.0387 | 0.8675 | **Stable (No Drift)** |
| **Shifted / Anomalous Cohort** | 300 | **2.7118** | 0.5107 | 0.0000 | **ALERT: Significant Distribution Shift** |

### Alert Policies & Escalation Paths
- **$\text{PSI} < 0.10$**: System operating normally. No intervention needed.
- **$0.10 \le \text{PSI} < 0.25$**: Moderate shift. The platform logs warning alerts and queues unrecognized skill tokens for batch semantic clustering.
- **$\text{PSI} \ge 0.25$**: Critical distribution shift. The system triggers an automated operational alert for engineers to review incoming traffic sources and inspect potential demographic or occupational shifts.

---

## 4. Production Telemetry, Model Versioning, and Rollback Procedures

### 1. Structured Telemetry Logging
Every inference transaction is asynchronously logged to `data/telemetry/inference_telemetry.jsonl` with full observability metadata:
- `timestamp`: UTC ISO-8601 timestamp.
- `endpoint`: `/api/v1/readiness` or `/api/v1/counterfactual-ranking`.
- `occupation_id`: Target career code.
- `candidate_skills_count`: Total input skill tokens.
- `vocabulary_coverage`: Proportion of tokens resolved to graph nodes.
- `readiness_score`: Computed baseline readiness ($[0.0, 1.0]$).
- `latency_ms`: Execution time (typically $8\text{--}18\text{ ms}$).
- `quality_flags`: Any detected validation warnings.

### 2. Versioning & Artifact Reproducibility
- All data preparation steps—from raw text ingestion through semantic crosswalking and graph construction—are driven by deterministic scripts with fixed random seeds.
- Model artifacts are bundled with complete metadata: feature dimension (`features_dim=449`), TF-IDF vocabulary mapping, and pre-computed skill embedding matrices.

### 3. Safe Rollback Protocol
If live monitoring detects anomalous drift ($\text{PSI} \ge 0.25$) or elevated error rates following a model update:
1. The service maintains an immediate local backup (`baseline_relevance_model.pkl.bak`).
2. The FastAPI lifecycle manager can swap the active model artifact in memory via a zero-downtime hot reload.
3. The API schema is strictly decoupled from the internal model weights, ensuring that client integrations never break during a model rollback.

---

## 5. Summary of Verified Evidence

| System Component | Test Coverage | Current Status |
| :--- | :--- | :---: |
| **Full Unit & Integration Test Suite** | 42 passing tests across 10 test modules | **100% Passing** |
| **Model Leakage Prevention** | Group-split verified across 910 occupations | **Verified (0% Leakage)** |
| **Cross-Validation Accuracy** | 5-Fold GroupKFold across all occupations | **$\overline{\rho} = 0.8743 \pm 0.0013$** |
| **Memory Footprint on LinuxONE** | Live container working set profiling | **279.51 MB** (Target: <350 MB) |
| **Real-World Input Handling** | Typo tolerance and out-of-vocabulary audit | **Audited & Tested** |
| **Drift Detection Engine** | Automated PSI & KS statistical tests | **Operational** |

### Concluding Note on Methodological Rigor
This audit confirms that CareerSynk Impact's skill-relevance and counterfactual ranking pipeline is not a superficial prototype running on cherry-picked data. Every performance metric cited is backed by strict occupational isolation, cross-validated stability, verified memory boundaries for IBM LinuxONE deployment, and realistic telemetry safeguards designed for real-world employment data.
