#DATA SCIENCE PRACTICAL - Descriptive Statistics
#PART 1 : Summary Statistics grouped by Categorical Variable
#Dataset : Student Performance (Self-Created → data.csv)

#Q1. Import Libraries 
import pandas as pd
import numpy as np

#STEP 0 : CREATE DATASET AND SAVE TO CSV
def create_student_dataset(filename="data.csv", seed=42):
    """
    Creates a realistic student performance dataset with:
    - StudentID, Age, Gender, Grade_Group, Score, StudyHrs
    - ~50 entries using numpy random
    - Realistic score ranges per grade group
    Saves to CSV file.
    """
    np.random.seed(seed)

    #Grade Groups with realistic score and study hour ranges
    grade_configs = {
        "A": {"count": 12, "score": (80, 100), "study": (5.0, 8.0)},
        "B": {"count": 14, "score": (65,  79), "study": (3.5, 6.0)},
        "C": {"count": 13, "score": (50,  64), "study": (2.0, 4.5)},
        "D": {"count": 11, "score": (35,  49), "study": (0.5, 3.0)},
    }

    rows = []
    sid  = 1

    for grade, cfg in grade_configs.items():
        for _ in range(cfg["count"]):
            rows.append({
                "StudentID"  : sid,
                "Age"        : int(np.random.randint(17, 23)),
                "Gender"     : np.random.choice(["Male","Female"],
                                                 p=[0.52, 0.48]),
                "Score"      : int(np.random.randint(*cfg["score"])),
                "StudyHrs"   : round(np.random.uniform(*cfg["study"]), 1),
                "Grade_Group": grade,
            })
            sid += 1

    df = pd.DataFrame(rows)

    #Save to CSV
    df.to_csv(filename, index=False)
    print(f" Dataset created and saved to '{filename}'")
    print(f"   Rows: {len(df)} | Columns: {list(df.columns)}")

create_student_dataset("data.csv")

#STEP 1 : LOAD DATASET
df = pd.read_csv("data.csv")

print("\n=== Dataset Sample ===")
print(df.head())

print("\n=== Shape of Dataset ===")
print(df.shape)

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== Data Types ===")
print(df.dtypes)

#PART 1 - Q1 : SUMMARY STATISTICS GROUPED BY Grade_Group
#Score grouped by Grade_Group
print("\n=== Summary Statistics of Score grouped by Grade_Group ===")
grouped = df.groupby("Grade_Group")["Score"].agg(
    Mean   = "mean",
    Median = "median",
    Min    = "min",
    Max    = "max",
    Std    = "std"
).round(2)
print(grouped)

#StudyHrs grouped by Grade_Group 
print("\n=== Summary Statistics of StudyHrs grouped by Grade_Group ===")
grouped2 = df.groupby("Grade_Group")["StudyHrs"].agg(
    Mean   = "mean",
    Median = "median",
    Min    = "min",
    Max    = "max",
    Std    = "std"
).round(2)
print(grouped2)

#Age grouped by Grade_Group 
print("\n=== Summary Statistics of Age grouped by Grade_Group ===")
grouped3 = df.groupby("Grade_Group")["Age"].agg(
    Mean   = "mean",
    Median = "median",
    Min    = "min",
    Max    = "max",
    Std    = "std"
).round(2)
print(grouped3)

#PART 1 - Q2 : NUMERIC LIST PER CATEGORY
print("\n=== Numeric List per Grade Group (Score) ===")
grade_list = {
    grade: group["Score"].tolist()
    for grade, group in df.groupby("Grade_Group")
}

for grade, scores in grade_list.items():
    print(f"\nGrade {grade} ({len(scores)} students)")
    print(scores)

#DATA SCIENCE PRACTICAL
#PART 2 : Iris Dataset Statistical Details, Using Built-in Iris Dataset from sklearn

import pandas as pd
from sklearn import datasets

#Load Iris Dataset from sklearn 
# iris.data         → 150×4 numpy array of feature values
# iris.feature_names→ ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# iris.target       → array of 0, 1, 2 (species codes)
# iris.target_names → ['setosa', 'versicolor', 'virginica']
iris = datasets.load_iris()

#convert to DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

#Add species column — maps 0→setosa, 1→versicolor, 2→virginica
df["species"] = pd.Categorical.from_codes(
    iris.target,
    iris.target_names
)

print("\n=== Iris Dataset Sample ===")
print(df.head())

print("\n=== Shape of Dataset ===")
print(df.shape)

print("\n=== Species Present ===")
print(df["species"].unique())

#Statistical Details of Each Species
for species in df["species"].unique():

    #Filter rows for this species only
    #Drop species column — keep only numeric columns for stats
    subset = df[df["species"] == species].drop(columns="species")

    print(f"\n{'='*55}")
    print(f" Species : {species}")
    print(f"{'='*55}")

    #describe() → count, mean, std, min, 25%, 50%, 75%, max
    print("\n=== Statistical Summary using describe() ===")
    print(subset.describe().round(3))

    #Custom percentiles → 10th, 25th, 50th, 75th, 90th
    print("\n=== Percentiles ===")
    pct = subset.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    pct.index = ["10%", "25%", "50%", "75%", "90%"]
    print(pct.round(3))

    #Mean of each feature
    print("\n=== Mean ===")
    print(subset.mean().round(3))

    #Standard Deviation of each feature
    print("\n=== Standard Deviation ===")
    print(subset.std().round(3))
