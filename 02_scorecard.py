# ============================================================
# Credit Risk Scorecard — India
# 02: Logistic Regression & Scorecard Model
# Rini Awasthi | MA Economics, Madras School of Economics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)

df = pd.read_csv("data/loan_data.csv")

# ── 1. Feature Engineering ───────────────────────────────────
# One-hot encode categoricals
df_model = pd.get_dummies(df, columns=["employment_type", "loan_purpose"],
                          drop_first=True)

feature_cols = [
    "age", "income_monthly", "years_employed", "loan_amount",
    "loan_tenure_months", "credit_score", "existing_loans",
    "debt_to_income", "missed_payments_past",
    "employment_type_Contract", "employment_type_Self-Employed",
    "employment_type_Salaried",
    "loan_purpose_Education", "loan_purpose_Home",
    "loan_purpose_Personal", "loan_purpose_Business",
]
# Keep only columns that exist after dummies
feature_cols = [c for c in feature_cols if c in df_model.columns]

X = df_model[feature_cols]
y = df_model["default"]

# ── 2. Train/test split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
print(f"Default rate (train): {y_train.mean():.2%}  |  (test): {y_test.mean():.2%}")

# ── 3. Scale & fit logistic regression ──────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
model.fit(X_train_sc, y_train)

# ── 4. Model performance ─────────────────────────────────────
y_pred       = model.predict(X_test_sc)
y_pred_proba = model.predict_proba(X_test_sc)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Default","Default"]))

# ── 5. ROC Curve ─────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
ax.plot([0,1],[0,1], color="grey", linestyle="--", lw=1, label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.1, color="#1f77b4")
ax.set_title("ROC Curve — Credit Default Model", fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Confusion matrix
ax = axes[1]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Non-Default","Default"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/02_roc_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/02_roc_confusion.png")

# ── 6. Feature importance ────────────────────────────────────
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False).head(12)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#d62728" if c > 0 else "#1f77b4" for c in coef_df["coefficient"]]
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Top 12 Feature Coefficients — Logistic Regression\n"
             "(Red = increases default risk, Blue = decreases)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Coefficient (standardised)")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/02_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/02_feature_importance.png")

# ── 7. Build credit scorecard ────────────────────────────────
# Convert log-odds to points (industry standard PDO=20, pdo_odds=1/19)
PDO   = 20       # points to double the odds
ODDS  = 1/19     # target odds at base score
BASE  = 600      # base score

factor = PDO / np.log(2)
offset = BASE - factor * np.log(ODDS)

# Score = offset + factor * (-log-odds from model)
log_odds_test = model.decision_function(X_test_sc)
scores = (offset + factor * (-log_odds_test)).astype(int)
scores = np.clip(scores, 300, 900)

score_df = pd.DataFrame({"score": scores, "default": y_test.values,
                          "pd": y_pred_proba})

print(f"\nCredit Score Summary:")
print(score_df["score"].describe().round(0))
print(f"\nAvg score — Non-Default: {score_df[score_df.default==0].score.mean():.0f}")
print(f"Avg score — Default:     {score_df[score_df.default==1].score.mean():.0f}")

# Score distribution
fig, ax = plt.subplots(figsize=(10, 4))
for label, color, name in [(0,"#1f77b4","Non-Default"),(1,"#d62728","Default")]:
    subset = score_df[score_df.default==label]["score"]
    ax.hist(subset, bins=30, alpha=0.65, color=color, label=name, edgecolor="white")
ax.set_title("Credit Score Distribution by Default Status", fontsize=13, fontweight="bold")
ax.set_xlabel("Credit Score")
ax.set_ylabel("Count")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/02_score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/02_score_distribution.png")

# Save model outputs for next script
score_df.to_csv("outputs/scores.csv", index=False)
import joblib, pickle
with open("outputs/model.pkl","wb") as f:
    pickle.dump({"model": model, "scaler": scaler,
                 "features": feature_cols}, f)
print("\nLogistic regression & scorecard complete.")
