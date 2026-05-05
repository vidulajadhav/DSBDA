# ============================================================
#  DATA SCIENCE PRACTICAL
#  PART 1 : Summary Statistics grouped by Categorical Variable
#  PART 2 : Statistical Details of Iris Dataset Species
# ============================================================

import pandas as pd
import numpy as np

# ════════════════════════════════════════════════════════════
# PART 1 — Summary Statistics Grouped by Categorical Variable
# Dataset : Student Academic Performance (created manually)
# Categorical Variable : Grade_Group  (A / B / C)
# Numeric Variables    : Age, StudyHrs, Score
# ════════════════════════════════════════════════════════════

np.random.seed(42)
n = 60

df = pd.DataFrame({
    "StudentID"   : range(1, n+1),
    "Grade_Group" : np.random.choice(["A", "B", "C"], n, p=[0.3, 0.4, 0.3]),
    "Age"         : np.random.randint(17, 23, n),
    "StudyHrs"    : np.random.uniform(1, 9, n).round(1),
    "Score"       : np.random.randint(40, 100, n),
})

print("=== Dataset Sample ===")
print(df.head(8))
print("\nShape:", df.shape)

# ── Summary Statistics grouped by Grade_Group ───────────────
print("\n=== Summary Stats of Score grouped by Grade_Group ===")
grouped = df.groupby("Grade_Group")["Score"].agg(
    Mean   = "mean",
    Median = "median",
    Min    = "min",
    Max    = "max",
    Std    = "std"
).round(2)
print(grouped)

print("\n=== Summary Stats of StudyHrs grouped by Grade_Group ===")
grouped2 = df.groupby("Grade_Group")["StudyHrs"].agg(
    Mean   = "mean",
    Median = "median",
    Min    = "min",
    Max    = "max",
    Std    = "std"
).round(2)
print(grouped2)

# ── Numeric list for each response of categorical variable ───
print("\n=== Numeric List per Grade Group (Score) ===")
grade_list = {grade: group["Score"].tolist()
              for grade, group in df.groupby("Grade_Group")}
for grade, scores in grade_list.items():
    print(f"Grade {grade} ({len(scores)} students): {scores}")


# ════════════════════════════════════════════════════════════
# PART 2 — Iris Dataset: Statistical Details per Species
# Source  : https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
# ════════════════════════════════════════════════════════════

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(URL)

# Rename species to match standard names
iris["species"] = iris["species"].map({
    "setosa"    : "Iris-setosa",
    "versicolor": "Iris-versicolor",
    "virginica" : "Iris-virginica"
})

print("\n=== Iris Dataset Sample ===")
print(iris.head(5))
print("\nSpecies in dataset:", iris["species"].unique())
print("Shape:", iris.shape)

# ── Detailed Stats per Species ───────────────────────────────
for species in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]:
    subset = iris[iris["species"] == species].drop(columns="species")
    print(f"\n{'='*55}")
    print(f"  Species: {species}  ({len(subset)} records)")
    print(f"{'='*55}")

    print("\n-- describe() [mean, std, min, max, quartiles] --")
    print(subset.describe().round(3))

    print("\n-- Percentiles (10th, 25th, 50th, 75th, 90th) --")
    pct = subset.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    pct.index = ["10%", "25%", "50%", "75%", "90%"]
    print(pct.round(3))

    print("\n-- Mean & Standard Deviation --")
    print("Mean:\n", subset.mean().round(3))
    print("\nStd Dev:\n", subset.std().round(3))
