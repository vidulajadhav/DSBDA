# ============================================================
#  PRACTICAL 10 — Data Visualization : Iris Dataset
#  Source : https://archive.ics.uci.edu/ml/datasets/Iris
#  1. Feature types
#  2. Histogram per feature
#  3. Boxplot per feature
#  4. Compare distributions & identify outliers
#  + Line Chart and Scatter Plot
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ── Load Iris Dataset ────────────────────────────────────────
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df  = pd.read_csv(URL)
print("=== Iris Dataset ===")
print("Shape:", df.shape)
print(df.head())

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# ── Feature Types ────────────────────────────────────────────
print("\n=== Feature Types ===")
print("  sepal_length : Numeric (continuous) — length of sepal in cm")
print("  sepal_width  : Numeric (continuous) — width  of sepal in cm")
print("  petal_length : Numeric (continuous) — length of petal in cm")
print("  petal_width  : Numeric (continuous) — width  of petal in cm")
print("  species      : Nominal (categorical) — setosa/versicolor/virginica")

# ════════════════════════════════════════════════════════════
# PLOT 1 — HISTOGRAMS : All 4 features (2×2 grid)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
colors = ["steelblue", "tomato", "mediumseagreen", "orchid"]
for ax, feat, col in zip(axes.flatten(), features, colors):
    sns.histplot(df[feat], kde=True, color=col, ax=ax, bins=20)
    ax.set_title(f"Histogram: {feat}")
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
plt.suptitle("Practical 10 — Histograms of Iris Features", fontsize=13)
plt.tight_layout()
plt.savefig("p10_histograms.png")
plt.show()
print(">> Plot 1 saved: p10_histograms.png")

# ════════════════════════════════════════════════════════════
# PLOT 2 — BOX PLOTS : All 4 features by Species
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, feat in zip(axes.flatten(), features):
    sns.boxplot(
    data=df,
    x="species",
    y=feat,
    hue="species",
    palette=["steelblue","tomato","mediumseagreen"],
     ax=ax,
    legend=False
)
    ax.set_title(f"Box Plot: {feat}")
    ax.set_xlabel("Species")
    ax.set_ylabel(feat)
plt.suptitle("Practical 10 — Box Plots of Iris Features by Species", fontsize=13)
plt.tight_layout()
plt.savefig("p10_boxplots.png")
plt.show()
print(">> Plot 2 saved: p10_boxplots.png")

# ════════════════════════════════════════════════════════════
# PLOT 3 — LINE CHART : Mean of each feature per Species
# ════════════════════════════════════════════════════════════
mean_df = df.groupby("species")[features].mean().reset_index()
mean_melted = mean_df.melt(id_vars="species", var_name="Feature", value_name="Mean")

plt.figure(figsize=(9, 5))
sns.lineplot(data=mean_melted, x="Feature", y="Mean",
             hue="species", marker="o", linewidth=2.5,
             palette=["steelblue","tomato","mediumseagreen"])
plt.title("Practical 10 — Line Chart: Mean Feature Values per Species")
plt.xlabel("Feature")
plt.ylabel("Mean Value (cm)")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("p10_linechart_means.png")
plt.show()
print(">> Plot 3 saved: p10_linechart_means.png")

# ════════════════════════════════════════════════════════════
# PLOT 4 — SCATTER PLOT : Sepal vs Petal by Species
# ════════════════════════════════════════════════════════════
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="petal_length", y="petal_width",
                hue="species",
                palette=["steelblue","tomato","mediumseagreen"],
                s=80, alpha=0.8)
plt.title("Practical 10 — Scatter Plot: Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.tight_layout()
plt.savefig("p10_scatter_petal.png")
plt.show()
print(">> Plot 4 saved: p10_scatter_petal.png")

# ── Outlier Detection via IQR ────────────────────────────────
print("\n=== Outlier Summary (IQR Method) ===")
for feat in features:
    Q1, Q3 = df[feat].quantile(0.25), df[feat].quantile(0.75)
    IQR = Q3 - Q1
    out = df[(df[feat] < Q1 - 1.5*IQR) | (df[feat] > Q3 + 1.5*IQR)]
    print(f"  {feat:15s}: {len(out)} outlier(s)")

# ── Observations ────────────────────────────────────────────
print("""
=== Observations (Practical 10) ===
1. HISTOGRAM : petal_length and petal_width show bimodal/trimodal
   distributions — clear separation between species.
   sepal_width is closest to a normal distribution.
2. BOX PLOT  : Setosa has the smallest petals with tight spread.
   Virginica has the largest petals. Versicolor lies in between.
   sepal_width has a few outliers (dots outside whiskers).
3. LINE CHART: All 3 species follow the same trend across features
   but at different scales — Virginica is consistently largest.
4. SCATTER   : Setosa is perfectly separable from the other two.
   Versicolor and Virginica slightly overlap in petal dimensions.
""")
