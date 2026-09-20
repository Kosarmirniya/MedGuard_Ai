import joblib
import pandas as pd
from pathlib import Path

# masir model zakhire

BASE_DIR = Path (__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR /"ml"/"risk_model.joblib"

#bargozari model

model = joblib.load(MODEL_PATH)

#Tartib daqiq vizhgi ha motabeq dataset

FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]


def predict_risk(data):
    """
    دریافت اطلاعات ورودی و پیش‌بینی سطح ریسک
    """

    input_data = pd.DataFrame(
        [data.model_dump()],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities)
    risk_levels ={ 0 : "Low Risk" , 1:"Elevated Risk"}

    risk_level = risk_levels.get(int(prediction) , "Unknown")

    result = {
        "prediction": int(prediction),
        "risk_level": risk_level ,

        "confidence": round(float(confidence) * 100 , 2) ,

        "probabilities": {
            str(int(cls)): round(float(prob), 4)
            for cls, prob in zip(model.classes_, probabilities)
        }
    }

    return result