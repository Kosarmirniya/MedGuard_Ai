import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

import joblib


BASE_DIR =Path (__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "diabetes_012_health_indicators_BRFSS2015.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:" , df.shape)

TARGET = "Diabetes_012"

X = df.drop(columns=[TARGET])

y = (df[TARGET] > 0).astype(int)

print("\nBinary target distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split (X , y , test_size= 0.20 , random_state=42, stratify=y)

print("\nTrain shape:" , X_train.shape)

print("Test shape:" , X_test.shape)


print("\nCreating Binary Random Forest model ...")
model = RandomForestClassifier(n_estimators=100, random_state=42 , n_jobs=1 , class_weight="balanced")


print("\nTraining binary model ...")
model.fit(X_train , y_train)

print("Training completed !")


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)

print("\n==============================================================================")
print("Binary Model Evaluation")
print("================================================================================")

print("Accuracy:" , round(accuracy * 100 , 2) , "%")

print("\nClassification Report:")
print(classification_report(y_test , y_pred))


MODEL_PATH = BASE_DIR / "app" / "ml" / "risk_model_binary.joblib"

joblib.dump(model , MODEL_PATH)

print("\n===============================================================================")
print("Binary model saved successfully!")
print("Location:" , MODEL_PATH)
print("=================================================================================")