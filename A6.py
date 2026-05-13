#DATA ANALYTICS PRACTICAL - Naive Bayes Classification
#Dataset : Iris (loaded directly from sklearn)
#Goal    : Classify iris flower into 3 species: setosa / versicolor / virginica

#Import Libraries 
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

#STEP 1 : LOAD IRIS DIRECTLY FROM SKLEARN
# iris.data         → 150×4 numpy array of feature values
# iris.feature_names→ 4 column names
# iris.target       → 0, 1, 2 (species codes)
# iris.target_names → ['setosa', 'versicolor', 'virginica']

iris_data = datasets.load_iris()

#Build clean DataFrame
df = pd.DataFrame(
    iris_data.data,
    columns=iris_data.feature_names
)

#Add species name column
df["species"] = pd.Categorical.from_codes(
    iris_data.target,
    iris_data.target_names
)

print("=== Dataset Sample ===")
print(df.head())

print("\n=== Shape of Dataset ===")
print(df.shape)

print("\n=== Statistical Summary ===")
print(df.describe().round(3))

print("\n=== Species Present ===")
print(df["species"].unique())

print("\n=== Class Distribution ===")
print(df["species"].value_counts())

#Feature Descriptions 
print("\n=== Feature Descriptions ===")
feat_desc = {
    "sepal length (cm)": "Length of sepal in centimeters",
    "sepal width (cm)" : "Width of sepal in centimeters",
    "petal length (cm)": "Length of petal in centimeters",
    "petal width (cm)" : "Width of petal in centimeters",
    "species"          : "TARGET — setosa/versicolor/virginica",
}
for col, desc in feat_desc.items():
    print(f"  {col:22s} : {desc}")

#STEP 2 : FEATURES AND TARGET
#X → all 4 numeric feature columns
#y → species column (what we want to predict)
X = df.drop(columns=["species"])
y = df["species"]

print(f"\nFeatures (X) shape : {X.shape}")
print(f"Target   (y) shape : {y.shape}")
print(f"Feature columns    : {list(X.columns)}")

#STEP 3 : TRAIN-TEST SPLIT (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining Samples : {X_train.shape[0]}")
print(f"Testing  Samples : {X_test.shape[0]}")

#STEP 4 : BUILD NAIVE BAYES MODEL
# GaussianNB because:
# → Features are CONTINUOUS numeric values
# → Assumes Gaussian (normal) distribution per feature per class
# → Uses Bayes theorem: P(class|features) ∝ P(features|class) × P(class)

model = GaussianNB()
model.fit(X_train, y_train)

print("\n=== Model Class Priors ===")
prior_df = pd.DataFrame({
    "Species"   : model.classes_,
    "Prior Prob": model.class_prior_.round(4)
})
print(prior_df.to_string(index=False))
print("(Prior = proportion of each class in training data)")

#STEP 5 : PREDICTIONS
y_pred = model.predict(X_test)

print("\n=== Actual vs Predicted (first 10) ===")
comparison = pd.DataFrame({
    "Actual"   : y_test.values[:10],
    "Predicted": y_pred[:10],
    "Correct"  : (y_test.values[:10] == y_pred[:10])
})
print(comparison.to_string(index=False))

#STEP 6 : CONFUSION MATRIX
cm     = confusion_matrix(y_test, y_pred)
labels = model.classes_

print("\n=== Confusion Matrix ===")
cm_df = pd.DataFrame(
    cm,
    index   = [f"Actual: {l}"    for l in labels],
    columns = [f"Predicted: {l}" for l in labels]
)
print(cm_df)

print("\n=== How to Read Confusion Matrix ===")
print("  Rows    → Actual class")
print("  Columns → Predicted class")
print("  Diagonal values → Correct predictions (TP)")
print("  Off-diagonal    → Wrong predictions (FP or FN)")

#STEP 7 : TP, FP, FN, TN PER CLASS
# Multi-class → One-vs-Rest approach for each class:
#   TP = correctly predicted as this class
#   FP = other classes wrongly predicted as this class
#   FN = this class wrongly predicted as other class
#   TN = all others correctly NOT predicted as this class

print("\n=== TP, FP, FN, TN per Class (One-vs-Rest) ===")
for i, label in enumerate(labels):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - TP - FP - FN

    print(f"\n  Class : {label}")
    print(f"  TP = {TP}  → Predicted {label}, Actually {label} ")
    print(f"  FP = {FP}  → Predicted {label}, Actually something else ")
    print(f"  FN = {FN}  → Predicted something else, Actually {label} ")
    print(f"  TN = {TN}  → Correctly predicted as NOT {label} ")

#STEP 8 : OVERALL EVALUATION METRICS
accuracy   = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision  = precision_score(y_test, y_pred, average="macro")
recall     = recall_score(y_test, y_pred,    average="macro")

print("\n=== Overall Evaluation Metrics ===")
print(f"  Accuracy   = correct/total  = {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Error Rate = 1 - Accuracy   = {error_rate:.4f} ({error_rate*100:.2f}%)")
print(f"  Precision  = TP/(TP+FP)     = {precision:.4f}")
print(f"  Recall     = TP/(TP+FN)     = {recall:.4f}")

#STEP 9 : FULL CLASSIFICATION REPORT
print("\n=== Full Classification Report ===")
print(classification_report(
    y_test, y_pred,
    target_names=iris_data.target_names
))
