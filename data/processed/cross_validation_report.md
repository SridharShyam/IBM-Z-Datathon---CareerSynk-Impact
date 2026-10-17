# Model Cross-Validation, Regularization, and Generalization Report
*Stage A Skill-Relevance Model Optimization*

## 1. Cross-Validation & Leakage Prevention Setup
- **Cross-Validation Scheme**: 5-Fold `GroupKFold` grouped strictly by **Occupation ID** (910 total occupations).
- **Leakage Guarantee**: 0% overlap between train and validation occupations across every fold. Evaluates pure inductive generalization on unseen occupations.
- **Features**: Dense SentenceTransformer embeddings (384) + TF-IDF functional descriptors (64) + Skill-Occupation semantic interaction (1) = 449 features.

---

## 2. 5-Fold Cross-Validation Benchmark Results

| Model Architecture | Regularization / Hyperparameters | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Spearman $\overline{\rho}$ | Median Spearman $\rho$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Trivial_Mean** | Constant mean predictor | `0.1480 ± 0.0011` | `0.1758 ± 0.0010` | **`0.0000 ± 0.0000`** | `0.0000 ± 0.0000` |
| **Ridge_L2_Regularized** | RidgeCV L2 penalty ($\alpha \in [10^{-2}, 10^3]$) | `0.0811 ± 0.0009` | `0.1038 ± 0.0008` | **`0.8039 ± 0.0048`** | `0.8786 ± 0.0050` |
| **Baseline_HistGradientBoosting** | Unregularized (L2=0.0, leaf=20) | `0.0661 ± 0.0015` | `0.0861 ± 0.0017` | **`0.8747 ± 0.0016`** | `0.9049 ± 0.0058` |
| **Regularized_Tuned_HistGradientBoosting** | Regularized (L2=1.5, leaf=30, early stopping) | `0.0662 ± 0.0012` | `0.0863 ± 0.0014` | **`0.8743 ± 0.0013`** | `0.9045 ± 0.0042` |

---

## 3. Generalization & Regularization Findings
1. **Ranking Stability ($\overline{\rho}$)**: Cross-validation confirms the model's ranking ability generalizes robustly across all 5 folds without overfitting to specific occupational clusters.
2. **L2 Regularization & Early Stopping Benefit**: Injecting L2 penalty (`l2_regularization=1.5`) and restricting tree granularity (`min_samples_leaf=30`) maintains high ranking fidelity while bounding leaf weight magnitudes, dampening variance across diverse occupational profiles.
3. **Downstream Impact on Counterfactual Discovery**: Stable per-occupation ranking ensures that counterfactual skill-gap prioritization ($\Delta = 	ext{Readiness}(C \cup \{s\}, O) - 	ext{Readiness}(C, O)$) produces consistent, trustworthy recommendations across disparate target careers.
