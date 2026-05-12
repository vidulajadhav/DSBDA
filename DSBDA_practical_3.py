# ============================================================
# DATA SCIENCE PRACTICAL
# PART 1 : Summary Statistics grouped by Categorical Variable
# ============================================================

import pandas as pd

# ── Load Dataset from CSV File ──────────────────────────────
df = pd.read_csv("data.csv")

print("=== Dataset Sample ===")
print(df.head())

print("\n=== Shape of Dataset ===")
print(df.shape)

print("\n=== Column Names ===")
print(df.columns)

# ============================================================
# Example Assumption:
# Categorical Column : Grade_Group
# Numeric Columns    : Score, StudyHrs
# ============================================================

# ── Summary Statistics of Score grouped by Grade_Group ──────
print("\n=== Summary Statistics of Score grouped by Grade_Group ===")

grouped = df.groupby("Grade_Group")["Score"].agg(
    Mean   = "mean",
    Median = "median",
    Min    = "min",
    Max    = "max",
    Std    = "std"
).round(2)

print(grouped)

# ── Summary Statistics of StudyHrs grouped by Grade_Group ───
print("\n=== Summary Statistics of StudyHrs grouped by Grade_Group ===")

grouped2 = df.groupby("Grade_Group")["StudyHrs"].agg(
    Mean   = "mean",
    Median = "median",
    Min    = "min",
    Max    = "max",
    Std    = "std"
).round(2)

print(grouped2)

# ── Numeric List for each Category ──────────────────────────
print("\n=== Numeric List per Grade Group (Score) ===")

grade_list = {
    grade: group["Score"].tolist()
    for grade, group in df.groupby("Grade_Group")
}

for grade, scores in grade_list.items():
    print(f"\nGrade {grade} ({len(scores)} students)")
    print(scores)

# ============================================================
# DATA SCIENCE PRACTICAL
# PART 2 : Iris Dataset Statistical Details
# Using Default Iris Dataset from sklearn
# ============================================================

import pandas as pd
from sklearn import datasets

# ── Load Iris Dataset from sklearn ──────────────────────────
iris = datasets.load_iris()

# Convert into DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add species column
df["species"] = pd.Categorical.from_codes(
    iris.target,
    iris.target_names
)

print("=== Iris Dataset Sample ===")
print(df.head())

print("\n=== Shape of Dataset ===")
print(df.shape)

print("\n=== Species Present ===")
print(df["species"].unique())

# ============================================================
# Statistical Details of Each Species
# ============================================================

for species in df["species"].unique():

    subset = df[df["species"] == species].drop(columns="species")

    print(f"\n{'='*55}")
    print(f" Species : {species}")
    print(f"{'='*55}")

    # ── describe() ──────────────────────────────────────────
    print("\n=== Statistical Summary using describe() ===")
    print(subset.describe().round(3))

    # ── Percentiles ─────────────────────────────────────────
    print("\n=== Percentiles ===")

    pct = subset.quantile([0.10, 0.25, 0.50, 0.75, 0.90])

    pct.index = ["10%", "25%", "50%", "75%", "90%"]

    print(pct.round(3))

    # ── Mean ────────────────────────────────────────────────
    print("\n=== Mean ===")
    print(subset.mean().round(3))

    # ── Standard Deviation ─────────────────────────────────
    print("\n=== Standard Deviation ===")
    print(subset.std().round(3))
