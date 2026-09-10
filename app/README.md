# MedGuard AI 🏥🤖

An intelligent healthcare risk assessment prototype built with Python, FastAPI, SQLAlchemy, and Machine Learning.

MedGuard AI combines a REST API, patient management, machine learning-based risk estimation, feature importance analysis, and risk assessment history in a simple dashboard.

> ⚠️ This project is an educational and portfolio prototype. It is not clinically validated and should not be used for medical diagnosis or treatment.

---

## 🚀 Features

- Machine Learning-based risk estimation
- Random Forest classification model
- Risk probabilities and confidence score
- Feature importance analysis
- Patient management
- Patient details
- Risk assessment history
- Dashboard statistics
- FastAPI REST API
- SQLAlchemy database integration
- Interactive Swagger API documentation

---

## 🛠️ Technologies

- Python
- FastAPI
- SQLAlchemy
- Scikit-learn
- Pandas
- NumPy
- Joblib
- HTML
- CSS
- JavaScript
- SQLite

---

## 🧠 Machine Learning

The machine learning component uses health-related indicators to estimate diabetes-related risk categories.

The model was trained using the CDC Diabetes Health Indicators dataset.

### Model

- Algorithm: Random Forest Classifier
- Input features: 21 health indicators
- Output: Risk category
- Probability estimates: Included
- Feature importance: Included

---

## 📂 Project Structure

`text
MedGuard_Ai/
│
├── app/
│   ├── api/
│   ├── core/
│   ├── db/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   ├── ml/
│   └── main.py
│
├── data/
├── notebooks/
├── tests/
├── README.md
└── requirements.txt
