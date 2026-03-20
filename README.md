# 🏦 Credit Risk Scorecard — Indian Retail Lending

## Overview
Built an end-to-end credit risk scorecard using logistic regression to predict the probability of loan default for retail borrowers in India. The model scores borrowers on a 300–900 scale and segments them into risk bands — directly replicating the methodology used by credit rating agencies and banks in practice.

---

## Motivation
Credit risk modelling is at the core of what firms like CRISIL, KPMG, and internal bank risk teams do every day. I built this project to understand how borrower characteristics translate into default probability, and how that probability is converted into an actionable credit score that can inform lending decisions.

---

## Data
- **5,000 synthetic loan accounts** structured to reflect Indian retail lending patterns
- **Variables:**
  - Borrower: age, monthly income, employment type, years employed
  - Loan: amount, tenure, purpose
  - Credit history: credit score, existing loans, missed payments
  - Derived: debt-to-income ratio
- **Default rate: ~11%** (consistent with RBI published NPA ranges for retail segments)

---

## Methodology

### 1. Exploratory Data Analysis
- Analysed default rates across employment type, loan purpose, and DTI buckets
- Examined credit score distributions for defaulters vs non-defaulters
- Correlation analysis to identify multicollinearity

### 2. Logistic Regression Model
- One-hot encoded categorical variables; standardised numerical features
- 75/25 train-test split with stratification on default flag
- Used `class_weight="balanced"` to handle class imbalance
- Evaluated via ROC-AUC, classification report, and confusion matrix

### 3. Credit Scorecard Scaling
- Converted model log-odds to a 300–900 point scale using industry-standard PDO (Points to Double Odds) methodology
- Base score: 600 | PDO: 20 points

### 4. Model Validation
- **KS Statistic** — measures separation between default and non-default score distributions
- **Gini Coefficient** — derived from ROC-AUC: `Gini = 2 × AUC − 1`
- Benchmarked against industry standards (good model: AUC > 0.70, Gini > 0.40)

### 5. Risk Band Segmentation
Borrowers segmented into 5 bands:

| Band | Score Range | Risk Level |
|------|-------------|------------|
| A | > 720 | Very Low Risk |
| B | 650 – 720 | Low Risk |
| C | 580 – 650 | Medium Risk |
| D | 500 – 580 | High Risk |
| E | < 500 | Very High Risk |

---

## Key Results
- **ROC-AUC: ~0.80** — strong discriminatory power
- **Gini Coefficient: ~0.60** — above industry benchmark for retail credit
- **KS Statistic: ~0.45** — meaningful separation between defaulters and non-defaulters
- Credit score gap between defaulters and non-defaulters: ~80–100 points
- Top default predictors: missed past payments, debt-to-income ratio, credit score, employment type

---

## Tools & Libraries
| Tool | Purpose |
|------|---------|
| Python | Core analysis |
| Pandas, NumPy | Data processing |
| Scikit-learn | Logistic regression, model evaluation |
| Matplotlib | Visualisation |
| SciPy | KS statistic |

---

## Files
```
├── data/
│   └── loan_data.csv           # Loan dataset (5,000 accounts)
├── outputs/
│   ├── 01_default_by_category.png
│   ├── 01_credit_score_dist.png
│   ├── 02_roc_confusion.png
│   ├── 02_feature_importance.png
│   ├── 02_score_distribution.png
│   ├── 03_ks_plot.png
│   └── 03_risk_bands.png
├── 00_generate_data.py         # Synthetic data generation
├── 01_eda.py                   # Exploratory analysis
├── 02_scorecard.py             # Logistic regression & scorecard
├── 03_risk_segmentation.py     # KS, Gini, risk bands
└── README.md
```

---

## Why This Matters for Credit Risk Roles
This project maps directly to the work done in:
- **CRISIL** — rating models and probability of default estimation
- **Bank risk teams** — retail credit underwriting scorecards
- **KPMG/Deloitte risk advisory** — credit model validation and stress testing

---

## Author
**Rini Awasthi**
MA General Economics, Madras School of Economics
ge25rini@mse.ac.in
