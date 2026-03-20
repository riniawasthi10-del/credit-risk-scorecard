# ============================================================
# Credit Risk Scorecard — India
# 01: Exploratory Data Analysis
# Rini Awasthi | MA Economics, Madras School of Economics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/loan_data.csv")

print("Dataset shape:", df.shape)
print("\nDefault rate:", df["default"].mean().round(4))
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary:\n", df.describe().round(2))

# ── Plot 1: Default rate by key categories ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Default Rate by Borrower Characteristics", fontsize=14, fontweight="bold")

for ax, col in zip(axes, ["employment_type", "loan_purpose", "existing_loans"]):
    rates = df.groupby(col)["default"].mean().sort_values(ascending=False)
    bars = ax.bar(rates.index, rates.values * 100,
                  color=["#d62728" if v > df["default"].mean()*100 else "#1f77b4"
                         for v in rates.values], alpha=0.85)
    ax.axhline(df["default"].mean() * 100, color="black",
               linestyle="--", linewidth=1, label="Overall avg")
    ax.set_title(f"By {col.replace('_',' ').title()}", fontsize=11)
    ax.set_ylabel("Default Rate (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("outputs/01_default_by_category.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/01_default_by_category.png")

# ── Plot 2: Credit score distribution by default status ──────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (label, color) in zip(axes, [(0, "#1f77b4"), (1, "#d62728")]):
    subset = df[df["default"] == label]["credit_score"]
    ax.hist(subset, bins=30, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(subset.mean(), color="black", linestyle="--",
               label=f"Mean: {subset.mean():.0f}")
    ax.set_title(f"Credit Score — {'Non-Default' if label==0 else 'Default'}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Credit Score Distribution by Default Status", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/01_credit_score_dist.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/01_credit_score_dist.png")

# ── Plot 3: DTI vs default ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
bins = pd.cut(df["debt_to_income"], bins=8)
dti_default = df.groupby(bins)["default"].mean() * 100
ax.bar(range(len(dti_default)), dti_default.values,
       color="#ff7f0e", alpha=0.85, edgecolor="white")
ax.set_xticks(range(len(dti_default)))
ax.set_xticklabels([str(b) for b in dti_default.index], rotation=30, fontsize=8)
ax.set_title("Default Rate by Debt-to-Income Ratio", fontsize=13, fontweight="bold")
ax.set_ylabel("Default Rate (%)")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/01_dti_default.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/01_dti_default.png")

# ── Plot 4: Correlation heatmap ──────────────────────────────
num_cols = ["age","income_monthly","years_employed","loan_amount",
            "credit_score","existing_loans","debt_to_income",
            "missed_payments_past","default"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
labels = [c.replace("_","\n") for c in num_cols]
ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(labels, fontsize=7, rotation=30)
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(labels, fontsize=7)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6)
ax.set_title("Correlation Matrix — Loan Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/01_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/01_correlation.png")

print("\nEDA complete.")
