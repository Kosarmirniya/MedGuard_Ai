import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "diabetes_012_health_indicators_BRFSS2015.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "app" / "ml" / "risk_model.joblib"
DEFAULT_METADATA_PATH = BASE_DIR / "app" / "ml" / "risk_model_metadata.json"
TARGET_COLUMN = "Diabetes_012"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MedGuard AI risk model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_dataset(data_path: Path) -> pd.DataFrame:
    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    n_before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning("Dropped %d rows with missing target values", n_dropped)
    logger.info("Dataset shape: %s", df.shape)
    return df

def merge_target_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the orginal 3-class target (0=healthy ,  1=prediabetic, 
    2=diabetic) into a 2-class target: 0 = low risk(healthy or prediabetic)
    , 1 = high risk(diabetic).

    """
    df = df.copy()
    df[TARGET_COLUMN]= df[TARGET_COLUMN].map({0.0:0, 1.0: 0 , 2.0: 1}).astype(int)
    logger.info("Merged target distribution:\n%s" , df[TARGET_COLUMN]. value_counts())

    return df



def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Train class distribution:\n%s", y_train.value_counts())
    logger.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators: int, random_state: int) -> BalancedRandomForestClassifier:
    """
    BalancedRandomForestClassifier balances classes internally, by drawing a
    balanced bootstrap sample for every tree during fit. No separate external
    resampling step (e.g. SMOTE) is needed or performed here.
    """
    logger.info("Training BalancedRandomForestClassifier (n_estimators=%d)", n_estimators)
    model = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    logger.info("Training completed")
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("Classification report:\n%s", report_text)
    logger.info("Confusion matrix:\n%s", cm)

    # For a medical risk model, recall on the at-risk classes (1, 2) matters
    # more than overall/weighted accuracy, since missing an at-risk patient
    # is costlier than a false alarm.
    
    if "1" in report_dict:
            logger.info("High-risk class -> recall: %.3f , precision: %.3f" , 
             report_dict["1"]["recall"] , report_dict["1"]["precision"] ,
            )

    return {
        "accuracy": report_dict["accuracy"],
        "macro_avg": report_dict["macro avg"],
        "weighted_avg": report_dict["weighted avg"],
        "per_class": {k: v for k, v in report_dict.items() if k not in ("accuracy", "macro avg", "weighted avg")},
        "confusion_matrix": cm.tolist(),
    }
def save_model_and_metadata(model, metrics: dict, feature_names: list, model_path: Path, metadata_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "model_type": type(model).__name__,
        "n_estimators": model.n_estimators,
        "feature_names": feature_names,
        "classes": model.classes_.tolist(),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Model saved to %s", model_path)
    logger.info("Metadata saved to %s", metadata_path)


def main():
    
    args = parse_args()
    df = load_dataset(args.data_path)
    df = merge_target_classes(df)
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.random_state)
    model = train_model(X_train, y_train, args.n_estimators, args.random_state)
    metrics = evaluate_model(model, X_test, y_test)
    save_model_and_metadata(model, metrics, list(X_train.columns), args.model_path, args.metadata_path)


if __name__ == "__main__":
    main()