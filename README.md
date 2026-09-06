# DAC 2026 — NHPA Claim Fraud Detection

## Team
**Team Name:** 2.0
**Members:**
- Mutiara Dwi Artono
- Vannya Ade Gunawan

## Project Summary
A fraud probability prediction model for healthcare insurance claims, optimized
to support NHPA's limited audit capacity allocation (3%, 5%, 7%).

## Pipeline
1. **EDA** (`01_EDA.ipynb`) — data exploration, duplicate detection, leakage checks, sparsity analysis
2. **Preprocessing** (`02_preprocessing.ipynb`) — encoding, transformation, train/val split
3. **Modeling** (`03_modeling_randomforest.ipynb`) — 33 Random Forest configurations,
   5-fold Stratified CV (OOF), 3 ensemble methods (simple average, rank, weighted rank)
4. **Audit Allocation** (`04_audit_allocation.ipynb`) — Task C
5. **Fairness Evaluation** (`05_fairness_evaluation.ipynb`) — Task D

## Key EDA Findings
- The label is binary (~50/50 split), not a continuous probability as described in the data dictionary — essentially no class imbalance.
- Found 1,737 groups of identical feature combinations with differing fraud labels (5,361 rows) — indicating **irreducible error**: an upper bound on model performance that cannot be overcome with the currently available features.
- High-cardinality columns: `kdkc` (126 categories), `dati2` (486 categories), `typeppk`, `cmg`, `diagprimer` → handled via smoothed target encoding.
- Count columns (`dx2_*`, `proc*`) are highly sparse (>99% zeros in many columns) → transformed into binary indicators.

## Final Model
**Weighted Rank Ensemble** built from the top 10 (out of 33 candidates) Random Forest configurations, selected based on OOF performance and leaderboard results.

- **NormalizedRecall@5% (OOF): 0.9683**

## Task C Results — Audit Allocation

| Capacity | Claims Audited | Fraud Caught | Fraud Missed | Legitimate Claims Audited | Recall | Precision | NormalizedRecall@k | Lift vs Random |
|---|---|---|---|---|---|---|---|---|
| 3% | 4,805 | 4,706 | 75,497 | 99 | 5.87% | 97.94% | 0.9794 | 1.96x |
| 5% | 8,008 | 7,754 | 72,449 | 254 | 9.67% | 96.83% | 0.9683 | 1.93x |
| 7% | 11,212 | 10,748 | 69,455 | 464 | 13.40% | 95.86% | 0.9586 | 1.91x |

**Interpretation:**
- The model consistently catches nearly **2x more fraud** than random audit selection across all capacity levels.
- Audit precision is very high (95.9%–97.9%) — most flagged claims are genuinely fraudulent, minimizing wasted audit resources on legitimate claims.
- NormalizedRecall decreases slightly as audit capacity increases (0.979 → 0.968 → 0.959) — expected, since maintaining ranking precision becomes harder further down the list.
- **Cutoff sensitivity:** the score gap right at the 5% cutoff (0.000004) is smaller than the average gap between scores overall (0.000006) — meaning the audit decision at this threshold is **fairly sensitive**: small changes in the model or data could shift which claims fall just inside or outside the cutoff. This is noted as a limitation, particularly for claims scoring near the boundary.
- A total of 2,002 claims are recommended for audit from the test set at 5% capacity.

## Task D Results — Fairness

### Gender (jkpst)
| | Population Share | Audited Share | Difference |
|---|---|---|---|
| P (Female) | 53.62% | 54.37% | +0.75% |
| L (Male) | 46.38% | 45.63% | -0.75% |

- Base fraud rate (from true labels) is **identical** between L and P (0.5007 for both) — no bias in the source labels with respect to gender.
- Audit precision differs slightly: L (97.04%) vs P (96.65%) — a small (~0.4%) difference, not practically significant.
- **Conclusion:** audit disparity by gender is very small (<1%) and shows no systematic bias pattern.

### Age Group
| | Population Share | Audited Share | Difference |
|---|---|---|---|
| 40-59 | 29.87% | 30.33% | +0.46% |
| 18-39 | 25.84% | 26.21% | +0.37% |
| 0-17 | 25.05% | 25.62% | +0.57% |
| 60+ | 19.24% | 17.83% | **-1.41%** |

- Audit precision is relatively consistent across age groups (18-39: 96.67%, 40-59: 96.95%, 60+: 96.92%).
- The **60+ group is audited slightly LESS OFTEN** relative to its population share (a -1.41% difference, the largest among all age groups) — a pattern worth further discussion. This does **not necessarily indicate discrimination against elderly patients**; possible explanations include differing clinical patterns in this age group (e.g., types of diagnoses/procedures that are naturally less pronounced within the available features), rather than direct age-based bias.

### Gender × Age Group Interaction
Audit rates range from 4.56%–5.30% across all 8 gender×age group combinations — variation is small and does not indicate any subgroup being notably over- or under-represented, except for the (L, 60+) combination, which shows a slightly lower audit rate (4.56%) compared to other combinations.

## Tools & Libraries
Python 3.13, pandas, numpy, scikit-learn (RandomForestClassifier, StratifiedKFold), scipy (rankdata), matplotlib, seaborn.

## Project Structure
```
dac2026-fraud/
├── Dataset/{RawDataset, Processed}/
├── Notebooks/{01_EDA, 02_preprocessing, 03_modeling_randomforest,
│              04_audit_allocation, 05_fairness_evaluation}.ipynb
├── outputs/submissions/
├── results/{audit_allocation, fairness}/
└── README.md
```
