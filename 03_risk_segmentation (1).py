# ============================================================
# Credit Risk Scorecard — India
# 03: Risk Segmentation, KS Statistic & Gini Coefficient
# Rini Awasthi | MA Economics, Madras School of Economics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

# ── Load scores ──────────────────────────────────────────────
score_df = pd.read_csv("outputs/scores.csv")

defaults     = score_df[score_df.default == 1]["score"]
non_defaults = score_df[score_df.default == 0]["score"]

# ── 1. KS Statistic ──────────────────────────────────────────
ks_stat, ks_pval = ks_2samp(non_defaults, defaults)
print(f"KS Statistic : {ks_stat:.4f}")
print(f"KS p-value   : {ks_pval:.6f}")
print(f"Interpretation: {'Strong discriminatory power ✓' if ks_stat > 0.3 else 'Moderate discriminatory power'}")

# ── 2. Gini Coefficient ──────────────────────────────────────
auc  = roc_auc_score(score_df["default"], score_df["pd"])
gini = 2 * auc - 1
print(f"\nGini Coefficient : {gini:.4f}")
print(f"ROC-AUC          : {auc:.4f}")

# ── 3. KS Plot ───────────────────────────────────────────────
score_range = np.linspace(score_df["score"].min(), score_df["score"].max(), 200)
cdf_nd = np.array([( non_defaults <= s).mean() for s in score_range])
cdf_d  = np.array([(     defaults <= s).mean() for s in score_range])
ks_diff = np.abs(cdf_nd - cdf_d)
ks_idx  = np.argmax(ks_diff)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(score_range, cdf_nd, color="#1f77b4", lw=2, label="Non-Default CDF")
ax.plot(score_range, cdf_d,  color="#d62728", lw=2, label="Default CDF")
ax.axvline(score_range[ks_idx], color="grey", linestyle="--", lw=1.2)
ax.annotate(f"KS = {ks_stat:.3f}",
            xy=(score_range[ks_idx], (cdf_nd[ks_idx]+cdf_d[ks_idx])/2),
            xytext=(score_range[ks_idx]+15, 0.45),
            fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black"))
ax.set_title("KS Plot — Separation Between Default and Non-Default",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Credit Score")
ax.set_ylabel("Cumulative Distribution")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/03_ks_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/03_ks_plot.png")

# ── 4. Risk Band Segmentation ────────────────────────────────
bins   = [299, 500, 580, 650, 720, 900]
labels = ["Very High Risk\n(<500)", "High Risk\n(500-580)",
          "Medium Risk\n(580-650)", "Low Risk\n(650-720)",
          "Very Low Risk\n(>720)"]

score_df["risk_band"] = pd.cut(score_df["score"], bins=bins, labels=labels)

band_summary = score_df.groupby("risk_band", observed=True).agg(
    Count=("default","count"),
    Defaults=("default","sum"),
    Default_Rate=("default","mean"),
    Avg_Score=("score","mean"),
    Avg_PD=("pd","mean")
).reset_index()

band_summary["Default_Rate_%"] = (band_summary["Default_Rate"] * 100).round(1)
band_summary["Avg_PD_%"]       = (band_summary["Avg_PD"] * 100).round(1)
band_summary["Avg_Score"]      = band_summary["Avg_Score"].round(0).astype(int)

print("\nRisk Band Segmentation:")
print(band_summary[["risk_band","Count","Defaults","Default_Rate_%",
                     "Avg_Score","Avg_PD_%"]].to_string(index=False))

# ── 5. Risk Band Chart ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Credit Risk Band Segmentation", fontsize=14, fontweight="bold")

colors = ["#d62728","#ff7f0e","#ffbb78","#98df8a","#2ca02c"]

ax = axes[0]
ax.bar(range(len(band_summary)), band_summary["Count"],
       color=colors, alpha=0.85, edgecolor="white")
ax.set_xticks(range(len(band_summary)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_title("Number of Borrowers per Band", fontsize=11)
ax.set_ylabel("Count")
ax.grid(True, axis="y", alpha=0.3)

ax = axes[1]
bars = ax.bar(range(len(band_summary)), band_summary["Default_Rate_%"],
              color=colors, alpha=0.85, edgecolor="white")
ax.set_xticks(range(len(band_summary)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_title("Default Rate per Band (%)", fontsize=11)
ax.set_ylabel("Default Rate (%)")
ax.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars, band_summary["Default_Rate_%"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/03_risk_bands.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/03_risk_bands.png")

# ── 6. Summary metrics table ─────────────────────────────────
print(f"""
╔══════════════════════════════════════════╗
║        MODEL PERFORMANCE SUMMARY        ║
╠══════════════════════════════════════════╣
║  ROC-AUC        : {auc:.4f}               ║
║  Gini           : {gini:.4f}               ║
║  KS Statistic   : {ks_stat:.4f}               ║
╠══════════════════════════════════════════╣
║  Industry benchmarks (retail credit):   ║
║  Good model   → AUC > 0.70, Gini > 0.40║
║  Strong model → AUC > 0.75, Gini > 0.50║
╚══════════════════════════════════════════╝
""")

print("Risk segmentation & model validation complete.")
