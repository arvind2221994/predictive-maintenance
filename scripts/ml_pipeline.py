import argparse
import ast
import hashlib
import json
import logging
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT.joinpath("data", "engine_data.csv")
ARTIFACTS = ROOT.joinpath("artifacts")
MODEL_PATH = ROOT.joinpath("models", "best_engine_model.pkl")
FEATURES = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp",
]
TARGET = "Engine Condition"
PACKAGES = ("streamlit", "pandas", "numpy", "scipy", "scikit-learn", "joblib")
SEED = 42
LOGGER = logging.getLogger(__name__)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = FEATURES + [TARGET]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    cleaned = frame.loc[:, required].copy()
    for name in required:
        cleaned[name] = pd.to_numeric(cleaned[name], errors="raise")
    if not np.isfinite(cleaned.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains missing or non-finite sensor/target values.")
    if (cleaned.loc[:, FEATURES[:4]] < 0).any().any():
        raise ValueError("RPM and pressure readings must not be negative.")
    if set(cleaned[TARGET].unique()) != {0, 1}:
        raise ValueError("Engine Condition must contain both numeric class labels 0 and 1.")
    cleaned[TARGET] = cleaned[TARGET].astype(int)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    if cleaned[TARGET].value_counts().min() < 5:
        raise ValueError("At least five distinct records per class are required for the split and 3-fold CV.")
    return cleaned


def prepare() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Add your permitted training dataset at {DATA_PATH} before running the pipeline.")
    raw = pd.read_csv(DATA_PATH)
    cleaned = prepare_frame(raw)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(ARTIFACTS.joinpath("prepared.csv"), index=False)
    write_json(ARTIFACTS.joinpath("data_profile.json"), {
        "source_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "source_rows": len(raw),
        "cleaned_rows": len(cleaned),
        "duplicates_removed": len(raw) - len(cleaned),
        "class_counts": {str(label): int(count) for label, count in cleaned[TARGET].value_counts().items()},
        "feature_columns": FEATURES,
    })
    LOGGER.info("Validated and cleaned %s records; no scaling or outlier clipping applied.", len(cleaned))


def split() -> None:
    frame = pd.read_csv(ARTIFACTS.joinpath("prepared.csv"))
    train_frame, test_frame = train_test_split(
        frame, test_size=0.20, random_state=SEED, stratify=frame[TARGET]
    )
    train_frame.to_csv(ARTIFACTS.joinpath("train.csv"), index=False)
    test_frame.to_csv(ARTIFACTS.joinpath("test.csv"), index=False)
    LOGGER.info("Saved stratified 80/20 split: %s training and %s test records.", len(train_frame), len(test_frame))


def train() -> None:
    train_frame = pd.read_csv(ARTIFACTS.joinpath("train.csv"))
    features, target = train_frame.loc[:, FEATURES], train_frame[TARGET]
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Bagging": BaggingClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }
    grids = {
        "Decision Tree": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3]},
        "Bagging": {"n_estimators": [50, 100], "max_samples": [0.8, 1.0]},
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10]},
        "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]},
        "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = {}
    for name, model in models.items():
        scores[name] = float(cross_val_score(
            model, features, target, cv=cv, scoring="f1", n_jobs=2, error_score="raise"
        ).mean())
        LOGGER.info("%s training-only CV F1: %.4f", name, scores[name])
    selected = max(scores, key=scores.get)
    search = GridSearchCV(
        models[selected], grids[selected], scoring="f1", cv=cv,
        n_jobs=2, error_score="raise",
    )
    search.fit(features, target)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, MODEL_PATH, compress=3)
    write_json(ARTIFACTS.joinpath("selection.json"), {
        "selected_model": selected,
        "candidate_cv_f1": scores,
        "best_cv_f1": float(search.best_score_),
        "best_parameters": search.best_params_,
        "random_seed": SEED,
    })
    LOGGER.info("Tuned and saved %s with parameters %s.", selected, search.best_params_)


def evaluate() -> None:
    test_frame = pd.read_csv(ARTIFACTS.joinpath("test.csv"))
    features, target = test_frame.loc[:, FEATURES], test_frame[TARGET]
    model = joblib.load(MODEL_PATH)
    if list(model.feature_names_in_) != FEATURES or set(model.classes_) != {0, 1}:
        raise ValueError("Trained model has an unexpected feature schema or class labels.")
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)
    positive_index = list(model.classes_).index(1)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Model produced invalid class probabilities.")
    metrics = {
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities[:, positive_index])),
    }
    selection = json.loads(ARTIFACTS.joinpath("selection.json").read_text(encoding="utf-8"))
    profile = json.loads(ARTIFACTS.joinpath("data_profile.json").read_text(encoding="utf-8"))
    write_json(ROOT.joinpath("reports", "metrics.json"), {
        "test_metrics": metrics,
        "training_selection": selection,
        "data_profile": profile,
        "test_rows": len(test_frame),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "package_versions": {name: version(name) for name in PACKAGES},
    })
    LOGGER.info("Held-out test metrics: %s", metrics)


def notebook_app_source() -> str:
    notebook_path = ROOT.joinpath("PredictiveMaintenance.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = [
        cell for cell in notebook["cells"]
        if cell.get("metadata", {}).get("id") == "deploymentWriteApp"
    ]
    if len(cells) != 1:
        raise ValueError("The committed notebook must have one Section 5 deploymentWriteApp cell.")
    assignments = [
        node for node in ast.parse("".join(cells[0]["source"])).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "app_code" for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("The deploymentWriteApp cell must define one literal app_code string.")
    source = ast.literal_eval(assignments[0].value)
    if not isinstance(source, str):
        raise ValueError("app_code must be a string literal, not dynamically executed notebook code.")
    return source.strip() + "\n"


def build() -> None:
    source = notebook_app_source()
    compile(source, "streamlit_app.py", "exec")
    ROOT.joinpath("streamlit_app.py").write_text(source, encoding="utf-8")
    requirements = "\n".join(f"{name}=={version(name)}" for name in PACKAGES) + "\n"
    ROOT.joinpath("requirements.txt").write_text(requirements, encoding="utf-8")
    LOGGER.info("Regenerated the Streamlit app from the committed notebook and pinned six dependencies.")


STAGES = {"prepare": prepare, "split": split, "train": train, "evaluate": evaluate, "build": build}


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-interactive predictive-maintenance ML pipeline.")
    parser.add_argument("stage", choices=[*STAGES, "all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for name in STAGES if args.stage == "all" else [args.stage]:
        LOGGER.info("Starting pipeline stage: %s", name)
        STAGES[name]()


if __name__ == "__main__":
    main()
