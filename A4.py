#DATA ANALYTICS PRACTICAL: LINEAR REGRESSION USING OWN DATASET WITH MISSING VALUES HANDLING

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

#CREATE OWN DATASET
data = {

    "RM": [
        6.5, 7.2, 6.0, np.nan, 7.5,
        6.3, 5.5, 7.8, 6.9, 5.7,
        6.8, 7.1, 5.9, 6.4, 7.3,
        5.6, np.nan, 7.6, 6.1, 5.4
    ],

    "LSTAT": [
        4.5, 3.2, 8.1, 10.5, 2.8,
        6.2, 12.4, np.nan, 5.0, 11.2,
        4.8, 3.5, 9.6, 6.0, 2.5,
        13.1, 5.4, 1.7, 7.5, 14.0
    ],

    "PTRATIO": [
        15.3, 14.8, 18.2, 19.1, 13.9,
        16.5, 20.0, 13.5, 15.0, np.nan,
        15.2, 14.5, 18.8, 16.2, 13.8,
        20.5, 15.8, 13.2, 17.5, 21.0
    ],

    "TAX": [
        296, 242, 311, 350, 220,
        280, 400, 210, 260, 370,
        255, 240, 330, 290, 225,
        420, 270, 205, 300, 450
    ],

    "CRIM": [
        0.10, 0.05, 0.30, 0.45, 0.02,
        0.15, 0.60, 0.01, 0.12, 0.50,
        0.08, 0.04, 0.35, 0.18, 0.03,
        0.75, 0.14, 0.01, 0.25, 0.90
    ],

    "HousePrice": [
        25.0, 32.0, 22.5, 18.0, 35.0,
        24.0, 15.5, 38.0, 28.0, 17.0,
        29.0, 31.0, 20.0, 26.0, 34.0,
        14.0, 27.0, 39.0, 23.0, 12.5
    ]
}

#CONVERT INTO DATAFRAME
df = pd.DataFrame(data)

#SAVE DATASET LOCALLY
df.to_csv("boston.csv", index=False)

print("Dataset created and saved successfully!")

#LOAD DATASET
df = pd.read_csv("boston.csv")

print("\n=== Dataset Info ===")
print("Shape :", df.shape)

print("\n=== First 5 Rows ===")
print(df.head())

#CHECK MISSING VALUES
print("\n=== Missing Values Before Handling ===")
print(df.isnull().sum())

#HANDLE MISSING VALUES
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\n=== Missing Values After Handling ===")
print(df.isnull().sum())

#BASIC STATISTICS
print("\n=== Statistical Summary ===")
print(df.describe().round(2))

#SPLIT FEATURES AND TARGET
X = df.drop(columns=["HousePrice"])
y = df["HousePrice"]

#FEATURE SCALING
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

#TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

#BUILD LINEAR REGRESSION MODEL
model = LinearRegression()

model.fit(X_train, y_train)

#MODEL COEFFICIENTS
print("\n=== Model Coefficients ===")

coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_.round(3)
})

print(coeff_df.to_string(index=False))

print(f"\nIntercept : {model.intercept_:.3f}")

#PREDICTIONS
y_pred = model.predict(X_test)

print("\n=== Actual vs Predicted ===")

comparison = pd.DataFrame({
    "Actual": y_test.values.round(2),
    "Predicted": y_pred.round(2),
    "Difference": (y_test.values - y_pred).round(2)
})

print(comparison.to_string(index=False))

#MODEL EVALUATION
mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\n=== Evaluation Metrics ===")

print(f"Mean Absolute Error  (MAE)  : {mae:.3f}")
print(f"Mean Squared Error   (MSE)  : {mse:.3f}")
print(f"Root Mean Sq Error   (RMSE) : {rmse:.3f}")
print(f"R2 Score                    : {r2:.3f}")

print(f"\nR2 = {r2:.3f} means the model explains {r2*100:.1f}% variance in house prices.")
