# ============================================================
#  DATA SCIENCE PRACTICAL - Linear Regression
#  Dataset : Boston Housing
#  Source  : https://www.kaggle.com/c/boston-housing
#  Goal    : Predict House Price (MEDV) using feature variables
# ============================================================

# ── Q1. Import Libraries ────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# ── Load Boston Housing Dataset ─────────────────────────────
boston = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
df = boston.frame

# Rename target column for clarity
df.rename(columns={"MEDV": "HousePrice"}, inplace=True)

print("=== Dataset Info ===")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ── Feature Description ─────────────────────────────────────
print("\n=== Feature Descriptions ===")
features_info = {
    "CRIM"   : "Per capita crime rate by town",
    "ZN"     : "Proportion of residential land zoned",
    "INDUS"  : "Proportion of non-retail business acres",
    "CHAS"   : "Charles River dummy variable (1=bounds river)",
    "NOX"    : "Nitric oxide concentration",
    "RM"     : "Average number of rooms per dwelling",
    "AGE"    : "Proportion of owner-occupied units built before 1940",
    "DIS"    : "Weighted distance to employment centres",
    "RAD"    : "Index of accessibility to radial highways",
    "TAX"    : "Property tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B"      : "1000(Bk - 0.63)^2 where Bk = proportion of Black residents",
    "LSTAT"  : "% lower status of the population",
    "HousePrice": "Median value of homes in $1000s (TARGET)"
}
for col, desc in features_info.items():
    print(f"  {col:10s} : {desc}")


# ── Data Preprocessing ───────────────────────────────────────
print("\n=== Missing Values ===")
print(df.isnull().sum().sum(), "missing values found")

print("\n=== Basic Statistics ===")
print(df.describe().round(2))


# ── Split Features and Target ────────────────────────────────
X = df.drop(columns=["HousePrice"])
y = df["HousePrice"].astype(float)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")


# ── Build Linear Regression Model ───────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

print("\n=== Model Coefficients ===")
coeff_df = pd.DataFrame({
    "Feature"    : df.drop(columns=["HousePrice"]).columns,
    "Coefficient": model.coef_.round(3)
}).sort_values("Coefficient", ascending=False)
print(coeff_df.to_string(index=False))
print(f"\nIntercept: {model.intercept_:.3f}")


# ── Predictions ──────────────────────────────────────────────
y_pred = model.predict(X_test)

print("\n=== Sample Predictions vs Actual ===")
comparison = pd.DataFrame({
    "Actual"   : y_test.values[:10].round(2),
    "Predicted": y_pred[:10].round(2),
    "Difference": (y_test.values[:10] - y_pred[:10]).round(2)
})
print(comparison.to_string(index=False))


# ── Model Evaluation ─────────────────────────────────────────
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n=== Model Evaluation Metrics ===")
print(f"  Mean Absolute Error  (MAE)  : {mae:.3f}")
print(f"  Mean Squared Error   (MSE)  : {mse:.3f}")
print(f"  Root Mean Sq Error   (RMSE) : {rmse:.3f}")
print(f"  R² Score                    : {r2:.3f}")
print(f"\n  R² = {r2:.3f} means the model explains {r2*100:.1f}% of the variance in house prices.")
