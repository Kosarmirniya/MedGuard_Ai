

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from imblearn.over_sampling import RandomOverSampler


import joblib

from sklearn.metrics import confusion_matrix


# ============================================================
# 1. دریافت دیتاست از فایل محلی
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "diabetes_012_health_indicators_BRFSS2015.csv"

print("Loading dataset...")
print("Dataset path:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. جدا کردن Features و Target
# ============================================================

TARGET = "Diabetes_012"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 3. تقسیم داده‌ها به Train و Test
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nBalancing trainig data...")

ros=RandomOverSampler(random_state=42)

X_train_resampled , y_train_resampled = ros.fit_resample(X_train , y_train)

print("Before balancing:")
print(y_train.value_counts())

print("\nAfter balancing:")
print(y_train_resampled.value_counts())

print("\nTrain shape befor balancing:", X_train.shape)
print("Test shape after balancing:", X_train_resampled.shape)
print("Test shape:" , X_test.shape)


# ============================================================
# 4. ساخت مدل Random Forest
# ============================================================

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1 )


# ============================================================
# 5. آموزش مدل
# ============================================================

print("\nTraining model...")

model.fit(X_train_resampled, y_train_resampled)

print("Training completed!")


# ============================================================
# 6. ارزیابی مدل
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("Model Evaluation")
print("======================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))

# ============================================================
# 7. احتمال هر کلاس
# ============================================================

probabilities = model.predict_proba(X_test)

print("\nExample probabilities:")
print(probabilities[0])

print("\nClasses:")
print(model.classes_)


# ============================================================
# 8. ذخیره مدل
# ============================================================

MODEL_PATH = BASE_DIR / "app" / "ml" / "Risk_model.joblib"

joblib.dump(model, MODEL_PATH)

print("\n======================================")
print("Model saved successfully!")
print("Location:", MODEL_PATH)
print("======================================")




