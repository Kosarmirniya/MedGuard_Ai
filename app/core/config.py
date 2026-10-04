from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
APP_ENV = os.getenv("APP_ENV" , "development")
DEBUG = APP_ENV == "development"
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is missing.Set it in the .env file.")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

DATABASE_URL = f"sqlite:///{BASE_DIR / 'medguard.db'}"

MODEL_PATH = BASE_DIR /"app"/"ml"/"risk_model.joblib"

FEATURE_IMPORTANCE_PATH = BASE_DIR / "app" / "ml"/"feature_importance.csv"

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS" ,"http://localhost:8000,http://127.0.0.1.8000").split(",")