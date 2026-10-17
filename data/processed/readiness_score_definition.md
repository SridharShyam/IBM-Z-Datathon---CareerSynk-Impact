# Career Readiness Score & Counterfactual Ranking Definition (Milestone M5)
*Documented Modeling Decisions for CareerSynk Impact*

> [!CAUTION]
> **Mandatory Modeling & Fairness Disclaimer**
> DISCLAIMER: These are model-based predicted associations and score deltas, not causal claims about employability, hiring probability, or salary outcomes. The underlying model's accuracy is bounded by the M4 evaluation metrics (MAE 0.068, mean Spearman 0.88), and occupation-specific accuracy may vary.

---

## 1. Mathematical Formulation

Let:
- $O$ be a target occupation.
- $S_O$ be the set of skills required by $O$ in the occupation-skill graph.
- $C$ be the candidate's current set of verified skill proficiencies.
- $w(O, s) \in [0.0, 1.0]$ be the observed requirement strength from the graph.
- $\hat{r}(O, s) \in [0.0, 1.0]$ be the predicted skill relevance score from the Stage A ML model.

### A. Composite Skill Requirement Weight ($W_s$)
To synthesize observed empirical requirements with the model's structural generalization:
$$W_s = \alpha \cdot w(O, s) + (1 - \alpha) \cdot \hat{r}(O, s)$$
*(Default $\alpha = 0.50$ balances verified empirical requirement with learned occupational affinity).*

### B. Graph-Based Partial Credit Function ($\text{Credit}(s, C)$)
A candidate missing skill $s$ directly may still possess transferable foundational aptitudes via closely related competencies in their profile:
1. **Direct Mastery**: If $s \in C$, $\text{Credit}(s, C) = 1.0$.
2. **Adjacent Transferable Aptitude**: If $s \notin C$, let $\mathcal{N}_{\text{related}}(s)$ be the set of adjacent skills connected to $s$ via `related_to` or `possibly_related` edges with embedding similarity $\text{sim}(s, s') \ge \tau_{\text{sim}}$ (default $\tau_{\text{sim}} = 0.70$):
$$\text{Credit}(s, C) = \max_{s' \in (C \cap \mathcal{N}_{\text{related}}(s))} \left[ \text{sim}(s, s') \cdot \gamma \right]$$
*(Default discount factor $\gamma = 0.50$ reflects that related competencies provide helpful aptitude but do not fully substitute for direct mastery).*
3. **Unrelated Gap**: Otherwise, $\text{Credit}(s, C) = 0.0$.

### C. Overall Career Readiness Score
$$\text{Readiness}(C, O) = \frac{\sum_{s \in S_O} W_s \cdot \text{Credit}(s, C)}{\sum_{s \in S_O} W_s}$$

**Mathematical Properties**:
- **Normalized Range**: $\text{Readiness}(C, O) \in [0.0, 1.0]$.
- **Monotonicity**: For any skill $s^* \notin C$, $\text{Readiness}(C \cup \{s^*\}, O) \ge \text{Readiness}(C, O)$.
- **Priority Calibration**: Missing high-weight skills reduces readiness proportionally more than missing secondary or optional skills.

---

## 2. Handling the O*NET vs. ESCO Weight-Scale Mismatch

| Data Source | Original Scale in Raw Data | Transformation & Normalization Rule |
| :--- | :--- | :--- |
| **O*NET** | Continuous Importance ratings ($1.00$ to $5.00$) from occupational analysts | Scaled linearly: $w(O, s) = \frac{\text{Data Value}}{5.00} \in [0.20, 1.00]$. Missing O*NET skills default to $0.50$. |
| **ESCO** | Categorical bipartite relations: `essential` vs. `optional` | Mapped deterministically: `essential` $\rightarrow 1.00$, `optional` $\rightarrow 0.50$. |

### Modeling Rationale:
- The Stage A machine learning baseline was trained exclusively on continuous O*NET ratings ($R^2$, MAE, Spearman $\overline{\rho}=0.88$) to establish rigorous ground truth without artificial categorical thresholds.
- When evaluating O*NET target occupations, composite weight $W_s$ synthesizes the continuous O*NET score with the ML model's prediction.
- When evaluating ESCO occupations, the $1.0 / 0.5$ mapping provides an empirical floor that the continuous ML relevance predictor refines.

---

## 3. Counterfactual Simulation & Delta Ranking

For each missing skill $s_k \in S_O \setminus C$:
1. Simulate the hypothetical profile: $C' = C \cup \{s_k\}$.
2. Recompute counterfactual readiness: $\text{Readiness}(C', O)$.
3. Compute the predicted marginal impact:
$$\Delta_k = \text{Readiness}(C \cup \{s_k\}, O) - \text{Readiness}(C, O)$$
4. Sort missing skills descending by $\Delta_k$, breaking ties with composite weight $W_{s_k}$.
5. Generate deterministic, mathematically traceable explanation templates linking $\Delta_k$, $W_{s_k}$, and any existing partial credit.
