# ============================================================
# Credit Risk Scorecard — India
# 00: Generate Realistic Loan Dataset
# Rini Awasthi | MA Economics, Madras School of Economics
# ============================================================

import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

N = 5000  # number of loan accounts

# ── Borrower characteristics ─────────────────────────────────
age = np.random.randint(22, 62, N)

income = np.random.choice(
    [np.random.normal(25000, 5000),
     np.random.normal(55000, 10000),
     np.random.normal(120000, 30000)],
    N
)
income = np.clip(income, 10000, 500000).astype(int)

employment_type = np.random.choice(
    ["Salaried", "Self-Employed", "Business", "Contract"],
    N, p=[0.50, 0.25, 0.15, 0.10]
)

years_employed = np.clip(np.random.exponential(4, N), 0.5, 35).round(1)

loan_amount = np.random.choice(
    [np.random.normal(150000, 50000),
     np.random.normal(500000, 100000),
     np.random.normal(1500000, 400000)],
    N
).astype(int)
loan_amount = np.clip(loan_amount, 50000, 5000000)

loan_tenure = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], N)

loan_purpose = np.random.choice(
    ["Home", "Auto", "Personal", "Education", "Business"],
    N, p=[0.25, 0.20, 0.30, 0.10, 0.15]
)

credit_score = np.clip(
    np.random.normal(680, 80, N), 300, 900
).astype(int)

existing_loans = np.random.choice([0, 1, 2, 3, 4], N,
                                   p=[0.35, 0.30, 0.20, 0.10, 0.05])

debt_to_income = np.clip(
    loan_amount / (income * loan_tenure / 12), 0.05, 0.95
).round(3)

missed_payments_past = np.random.choice([0, 1, 2, 3], N,
                                         p=[0.60, 0.22, 0.12, 0.06])

# ── Default probability (realistic, driven by features) ──────
log_odds = (
    -4.5
    + 0.015 * (700 - credit_score) / 50        # low credit score → higher default
    + 0.8  * (debt_to_income - 0.3)            # high DTI → higher default
    - 0.3  * np.log(income / 30000)            # higher income → lower default
    + 0.4  * (employment_type == "Contract").astype(int)
    + 0.2  * (employment_type == "Self-Employed").astype(int)
    - 0.3  * (employment_type == "Salaried").astype(int)
    + 0.6  * missed_payments_past              # past behaviour predicts future
    - 0.1  * np.log1p(years_employed)          # more experience → lower default
    + 0.3  * (loan_purpose == "Personal").astype(int)
    - 0.2  * (loan_purpose == "Home").astype(int)
    + np.random.normal(0, 0.3, N)              # noise
)

prob_default = 1 / (1 + np.exp(-log_odds))
default = (np.random.uniform(0, 1, N) < prob_default).astype(int)

print(f"Default rate: {default.mean():.1%}  (realistic range: 8–15%)")

# ── Assemble ─────────────────────────────────────────────────
df = pd.DataFrame({
    "loan_id":             [f"LN{str(i).zfill(5)}" for i in range(1, N+1)],
    "age":                 age,
    "income_monthly":      income,
    "employment_type":     employment_type,
    "years_employed":      years_employed,
    "loan_amount":         loan_amount,
    "loan_tenure_months":  loan_tenure,
    "loan_purpose":        loan_purpose,
    "credit_score":        credit_score,
    "existing_loans":      existing_loans,
    "debt_to_income":      debt_to_income,
    "missed_payments_past":missed_payments_past,
    "default":             default,
})

df.to_csv("data/loan_data.csv", index=False)
print(f"Dataset saved: data/loan_data.csv  ({len(df)} rows, {df.columns.tolist()})")
print(df.describe().round(2))
