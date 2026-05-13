#  PRACTICAL 10 — Data Visualization : Iris Dataset USING sklearn Iris Dataset (NO INTERNET REQUIRED)
#  1. Feature types
#  2. Histogram per feature
#  3. Boxplot per feature
#  4. Compare distributions & identify outliers
#  + Line Chart and Scatter Plot

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris

#LOAD IRIS DATASET FROM sklearn
iris = load_iris(as_frame=True)

df = iris.frame

#Add species names
df["species"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

#Remove target column
df.drop("target", axis=1, inplace=True)

#Rename columns
df.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

#SAVE DATASET LOCALLY AS CSV
df.to_csv("Iris_Dataset.csv", index=False)

print("Dataset saved successfully!")
print("File Name : Iris_Dataset.csv")

#DISPLAY DATASET
print("\n=== Iris Dataset ===")

print("Shape:", df.shape)

print("\nFirst 5 Rows:")

print(df.head())

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

#FEATURE TYPES
print("\n=== Feature Types ===")

print("  sepal_length : Numeric (continuous)")

print("  sepal_width  : Numeric (continuous)")

print("  petal_length : Numeric (continuous)")

print("  petal_width  : Numeric (continuous)")

print("  species      : Nominal (categorical)")

#PLOT 1 — HISTOGRAMS
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

colors = [
    "steelblue",
    "tomato",
    "mediumseagreen",
    "orchid"
]

for ax, feat, col in zip(axes.flatten(), features, colors):

    sns.histplot(
        df[feat],
        kde=True,
        color=col,
        ax=ax,
        bins=20
    )

    ax.set_title(f"Histogram: {feat}")

    ax.set_xlabel(feat)

    ax.set_ylabel("Count")

plt.suptitle(
    "Practical 10 — Histograms of Iris Features",
    fontsize=13
)

plt.tight_layout()

plt.savefig("p10_histograms.png")

plt.show()

print(">> Plot 1 saved: p10_histograms.png")

#PLOT 2 — BOX PLOTS
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for ax, feat in zip(axes.flatten(), features):

    sns.boxplot(
        data=df,
        x="species",
        y=feat,
        hue="species",
        palette=["steelblue", "tomato", "mediumseagreen"],
        ax=ax,
        legend=False
    )

    ax.set_title(f"Box Plot: {feat}")

    ax.set_xlabel("Species")

    ax.set_ylabel(feat)

plt.suptitle(
    "Practical 10 — Box Plots of Iris Features by Species",
    fontsize=13
)

plt.tight_layout()

plt.savefig("p10_boxplots.png")

plt.show()

print(">> Plot 2 saved: p10_boxplots.png")

#PLOT 3 — LINE CHART
mean_df = df.groupby("species")[features].mean().reset_index()

mean_melted = mean_df.melt(
    id_vars="species",
    var_name="Feature",
    value_name="Mean"
)

plt.figure(figsize=(9, 5))

sns.lineplot(
    data=mean_melted,
    x="Feature",
    y="Mean",
    hue="species",
    marker="o",
    linewidth=2.5,
    palette=["steelblue", "tomato", "mediumseagreen"]
)

plt.title(
    "Practical 10 — Line Chart: Mean Feature Values per Species"
)

plt.xlabel("Feature")

plt.ylabel("Mean Value (cm)")

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig("p10_linechart_means.png")

plt.show()

print(">> Plot 3 saved: p10_linechart_means.png")

#PLOT 4 — SCATTER PLOT
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="petal_length",
    y="petal_width",
    hue="species",
    palette=["steelblue", "tomato", "mediumseagreen"],
    s=80,
    alpha=0.8
)

plt.title(
    "Practical 10 — Scatter Plot: Petal Length vs Petal Width"
)

plt.xlabel("Petal Length (cm)")

plt.ylabel("Petal Width (cm)")

plt.tight_layout()

plt.savefig("p10_scatter_petal.png")

plt.show()

print(">> Plot 4 saved: p10_scatter_petal.png")

#OUTLIER DETECTION USING IQR
print("\n=== Outlier Summary (IQR Method) ===")

for feat in features:

    Q1 = df[feat].quantile(0.25)

    Q3 = df[feat].quantile(0.75)

    IQR = Q3 - Q1

    outliers = df[
        (df[feat] < Q1 - 1.5 * IQR) |
        (df[feat] > Q3 + 1.5 * IQR)
    ]

    print(f"{feat:15s}: {len(outliers)} outlier(s)")

#OBSERVATIONS
# print("""
# === Observations (Practical 10) ===

# 1. HISTOGRAM :
#    petal_length and petal_width show clear separation
#    between species.

# 2. BOX PLOT :
#    Setosa has the smallest petals.
#    Virginica has the largest petals.

# 3. LINE CHART :
#    Virginica shows highest average feature values.

# 4. SCATTER PLOT :
#    Setosa is clearly separable from the other species.
# """)
