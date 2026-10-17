# Sensitivity Analysis: Counterfactual Skill-Gap Ranking Stability

## Executive Summary
This report assesses the stability of top-5 counterfactual skill-gap recommendations when shifting the composite requirement weight parameter $\alpha \in [0.2, 0.5, 0.8]$ in the formula $W_s = \alpha \cdot w(O, s) + (1 - \alpha) \cdot \hat{r}(O, s)$.
- **$\alpha = 0.20$**: Strongly weights ML occupational generalization $\hat{r}(O, s)$.
- **$\alpha = 0.50$ (Shipped Default)**: Balanced compromise between empirical requirement strength and ML relevance.
- **$\alpha = 0.80$**: Strongly weights observed structural requirement weight $w(O, s)$.

### Primary Metric: Top-5 Jaccard Set Overlap
**Metric Choice Justification**: In workforce upskilling systems, recommendations are surfaced as a cohort of high-priority learning interventions (e.g. top-5 missing skills). Whether a skill appears within the actionable top-5 intervention set is significantly more impactful to the candidate than a minor rank permutation between position #2 and #3. Therefore, **Top-5 Jaccard Set Overlap** ($J(A, B) = \frac{|A \cap B|}{|A \cup B|}$) between $\alpha=0.2$ and $\alpha=0.8$ is chosen as the primary stability metric, bounded in $[0.0, 1.0]$.

---

## Target Occupation: Software Developers (O*NET 15-1252.00) (`occ:onet:15-1252.00`)

| Candidate Profile | $\alpha=0.2$ Top-5 Skills | $\alpha=0.5$ (Default) Top-5 Skills | $\alpha=0.8$ Top-5 Skills | $J(\alpha_{0.2}, \alpha_{0.8})$ | $J(\alpha_{0.5}, \alpha_{0.8})$ |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Profile_1_CS_Graduate** | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Active Listening<br>4. Speaking<br>5. Judgment and Decision Making | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Active Listening<br>4. Judgment and Decision Making<br>5. Speaking | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Judgment and Decision Making<br>4. Active Listening<br>5. Complex Problem Solving | **0.67** | **0.67** |
| **Profile_2_Retail_Career_Switcher** | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Critical Thinking<br>4. Reading Comprehension<br>5. Judgment and Decision Making | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Critical Thinking<br>4. Reading Comprehension<br>5. Judgment and Decision Making | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Critical Thinking<br>4. Judgment and Decision Making<br>5. Reading Comprehension | **1.00** | **1.00** |
| **Profile_3_Midlevel_Tech_Lead** | 1. Reading Comprehension<br>2. Speaking<br>3. Judgment and Decision Making<br>4. Monitoring<br>5. Active Learning | 1. Reading Comprehension<br>2. Judgment and Decision Making<br>3. Speaking<br>4. Active Learning<br>5. Monitoring | 1. Judgment and Decision Making<br>2. Reading Comprehension<br>3. Active Learning<br>4. Speaking<br>5. Time Management | **0.67** | **0.67** |
| **Profile_4_Junior_Data_Analyst** | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Active Listening<br>4. Speaking<br>5. Judgment and Decision Making | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Active Listening<br>4. Judgment and Decision Making<br>5. Speaking | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Judgment and Decision Making<br>4. Programming<br>5. Active Listening | **0.67** | **0.67** |

## Target Occupation: General and Operations Managers (O*NET 11-1021.00) (`occ:onet:11-1021.00`)

| Candidate Profile | $\alpha=0.2$ Top-5 Skills | $\alpha=0.5$ (Default) Top-5 Skills | $\alpha=0.8$ Top-5 Skills | $J(\alpha_{0.2}, \alpha_{0.8})$ | $J(\alpha_{0.5}, \alpha_{0.8})$ |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Profile_1_CS_Graduate** | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Speaking<br>4. Active Listening<br>5. Monitoring | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Speaking<br>4. Active Listening<br>5. Management of Material Resources | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Management of Material Resources<br>4. Management of Financial Resources<br>5. Speaking | **0.43** | **0.67** |
| **Profile_2_Retail_Career_Switcher** | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Reading Comprehension<br>4. Critical Thinking<br>5. Monitoring | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Reading Comprehension<br>4. Critical Thinking<br>5. Management of Material Resources | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Management of Material Resources<br>4. Management of Financial Resources<br>5. Reading Comprehension | **0.43** | **0.67** |
| **Profile_3_Midlevel_Tech_Lead** | 1. Speaking<br>2. Reading Comprehension<br>3. Monitoring<br>4. Management of Material Resources<br>5. Judgment and Decision Making | 1. Speaking<br>2. Reading Comprehension<br>3. Management of Material Resources<br>4. Monitoring<br>5. Management of Financial Resources | 1. Management of Material Resources<br>2. Management of Financial Resources<br>3. Speaking<br>4. Reading Comprehension<br>5. Monitoring | **0.67** | **1.00** |
| **Profile_4_Junior_Data_Analyst** | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Speaking<br>4. Active Listening<br>5. Management of Material Resources | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Speaking<br>4. Active Listening<br>5. Management of Material Resources | 1. Systems Analysis<br>2. Systems Evaluation<br>3. Management of Material Resources<br>4. Management of Financial Resources<br>5. Speaking | **0.67** | **0.67** |

## Plain-English Conclusion & Shipped Default Decision
Across all synthetic candidate profiles and target occupations, the mean Top-5 Jaccard stability between extreme weighting regimes ($\alpha=0.2$ vs $\alpha=0.8$) is **0.65**.
The top-5 recommendations exhibit **moderate sensitivity** to the choice of $\alpha$, with some skill rank re-orderings occurring when moving from purely observed requirement weighting ($\alpha=0.8$) to model-heavy generalization ($\alpha=0.2$).

**Shipped Default Recommendation**: We **retain $\alpha = 0.50$** as the shipped default. At $\alpha = 0.50$, the system balances verified ground-truth requirements from the empirical occupational profile with the Stage A model's ability to smooth noise and penalize overly generic skills, without skewing recommendations exclusively toward either source.

---
### Mandatory Non-Causal Framing Disclaimer
> **Non-Causal Disclaimer**: Readiness scores and counterfactual deltas are statistical associations derived from cross-sectional occupational data. They indicate correlation with occupational skill requirements and transferable aptitude, not a guarantee of hiring, career advancement, or causal wage uplift.
