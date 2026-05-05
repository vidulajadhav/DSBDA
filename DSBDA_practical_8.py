# ============================================================
#  PRACTICAL 8 — Data Visualization : Titanic Dataset
#  1. Explore patterns using Seaborn
#  2. Histogram of Fare column
#  + Line Chart and Scatter Plot
#libraries to install:sudo apt-get install python3-tk   # needed for plot windows on Ubuntu
#                     pip3 install seaborn matplotlib pandas
# If the plot window doesn't appear, add this line at the top of the file:
# pythonimport matplotlib
# matplotlib.use('TkAgg') 
# ============================================================
import os
print("Saving in folder:", os.getcwd())
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ── Load inbuilt Titanic dataset ─────────────────────────────
df = sns.load_dataset("titanic")
print("=== Titanic Dataset ===")
print("Shape:", df.shape)
print(df[["survived","pclass","sex","age","fare","embarked"]].head(8))

# ════════════════════════════════════════════════════════════
# PLOT 1 — HISTOGRAM : Distribution of Fare
# ════════════════════════════════════════════════════════════
plt.figure(figsize=(8, 5))
sns.histplot(df["fare"], bins=40, kde=True, color="steelblue")
plt.title("Practical 8 — Histogram: Distribution of Ticket Fare")
plt.xlabel("Fare (£)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("p8_histogram_fare.png")
plt.show()
print(">> Plot 1 saved: p8_histogram_fare.png")

# ════════════════════════════════════════════════════════════
# PLOT 2 — LINE CHART : Average Fare by Passenger Class
# ════════════════════════════════════════════════════════════
avg_fare = df.groupby("pclass")["fare"].mean().reset_index()

plt.figure(figsize=(7, 4))
sns.lineplot(data=avg_fare, x="pclass", y="fare",
             marker="o", color="tomato", linewidth=2.5)
plt.title("Practical 8 — Line Chart: Average Fare by Passenger Class")
plt.xlabel("Passenger Class (1=First, 2=Second, 3=Third)")
plt.ylabel("Average Fare (£)")
plt.xticks([1, 2, 3])
plt.tight_layout()
plt.savefig("p8_linechart_fare.png")
plt.show()
print(">> Plot 2 saved: p8_linechart_fare.png")

# ════════════════════════════════════════════════════════════
# PLOT 3 — SCATTER PLOT : Age vs Fare (coloured by Survival)
# ════════════════════════════════════════════════════════════
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="age", y="fare",
                hue="survived", palette={0:"red", 1:"green"},
                alpha=0.6)
plt.title("Practical 8 — Scatter Plot: Age vs Fare (by Survival)")
plt.xlabel("Age")
plt.ylabel("Fare (£)")
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ["No (0)", "Yes (1)"], title="Survived")
plt.tight_layout()
plt.savefig("p8_scatter_age_fare.png")
plt.show()
print(">> Plot 3 saved: p8_scatter_age_fare.png")

# ── Observations ────────────────────────────────────────────
# print("""
# === Observations (Practical 8) ===
# 1. HISTOGRAM : Fare is heavily right-skewed — most passengers paid
#    low fares (< £50), very few paid extremely high fares (outliers).
# 2. LINE CHART : 1st class passengers paid significantly higher average
#    fare than 2nd and 3rd class, showing a steep drop across classes.
# 3. SCATTER   : Passengers who paid higher fares (1st class) had a
#    higher survival rate (green dots clustered at top).
# """)
