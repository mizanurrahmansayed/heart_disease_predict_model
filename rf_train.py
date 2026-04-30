import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("heart (1).csv")

print("Dataset Loaded Successfully")
print(df.head())

# -------------------------------
# Encode categorical variables
# -------------------------------
sex_mapping = {'Female': 0, 'Male': 1}
chest_pain_mapping = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
resting_ecg_mapping = {'LVH': 0, 'Normal': 1, 'ST': 2}
exercise_angina_mapping = {'No': 0, 'Yes': 1}
st_slope_mapping = {'Down': 0, 'Flat': 1, 'Up': 2}

df['Sex'] = df['Sex'].map(sex_mapping)
df['ChestPainType'] = df['ChestPainType'].map(chest_pain_mapping)
df['RestingECG'] = df['RestingECG'].map(resting_ecg_mapping)
df['ExerciseAngina'] = df['ExerciseAngina'].map(exercise_angina_mapping)
df['ST_Slope'] = df['ST_Slope'].map(st_slope_mapping)

# -------------------------------
# Feature Engineering (IMPORTANT)
# -------------------------------
df['BP_HR_Ratio'] = df['RestingBP'] / (df['MaxHR'] + 1)
df['Chol_Age_Ratio'] = df['Cholesterol'] / (df['Age'] + 1)
df['Risk_Score'] = (df['Oldpeak'] * 2) + (df['Age'] / 10)
df['Cardiac_Stress'] = df['MaxHR'] / (df['RestingBP'] + 1)

# -------------------------------
# Features and Target
# -------------------------------
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

print("\nFeature Columns:")
print(X.columns)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Train Random Forest Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save Model
# -------------------------------
with open("heart_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved as heart_disease_model.pkl")