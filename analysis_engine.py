import base64
from io import BytesIO
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COL = "target"

FEATURE_DESCRIPTIONS = {
    "age": "age in years",
    "sex": "sex: 1 = male, 0 = female",
    "cp": "chest pain type",
    "trestbps": "resting blood pressure",
    "chol": "serum cholesterol",
    "fbs": "fasting blood sugar > 120 mg/dl",
    "restecg": "resting electrocardiographic results",
    "thalach": "maximum heart rate achieved",
    "exang": "exercise induced angina",
    "oldpeak": "ST depression induced by exercise",
    "slope": "slope of peak exercise ST segment",
    "ca": "number of major vessels colored by fluoroscopy",
    "thal": "thalassemia category",
    "target": "heart disease indicator: 1 = disease, 0 = no disease",
}


def load_dataset(file_storage=None, default_path="data/heart.csv") -> pd.DataFrame:
    if file_storage and getattr(file_storage, "filename", ""):
        df = pd.read_csv(file_storage)
    else:
        df = pd.read_csv(default_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET_COL}'.")
    if df.empty:
        raise ValueError("Dataset is empty.")
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError("Only numeric columns are supported in this demo. Non-numeric: " + ", ".join(non_numeric))


def _round_float(value: Any, ndigits: int = 4):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), ndigits)
    return value


def dataframe_summary(df: pd.DataFrame) -> Dict[str, Any]:
    validate_dataset(df)
    target_counts = df[TARGET_COL].value_counts(dropna=False).sort_index().to_dict()
    missing = df.isna().sum().to_dict()
    duplicates = int(df.duplicated().sum())
    desc = df.describe().T[["mean", "std", "min", "50%", "max"]].round(3)
    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "missing_values": {k: int(v) for k, v in missing.items()},
        "duplicate_rows": duplicates,
        "descriptive_statistics": desc.to_dict(orient="index"),
    }


def target_group_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    validate_dataset(df)
    numeric_cols = [c for c in df.columns if c != TARGET_COL]
    grouped = df.groupby(TARGET_COL)[numeric_cols].mean().round(3)
    diff = (grouped.loc[1] - grouped.loc[0]).sort_values(key=lambda s: s.abs(), ascending=False).round(3)
    return {
        "means_by_target": grouped.to_dict(orient="index"),
        "largest_mean_differences_target1_minus_target0": diff.head(10).to_dict(),
    }


def correlation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    validate_dataset(df)
    corr = df.corr(numeric_only=True)
    target_corr = corr[TARGET_COL].drop(TARGET_COL).sort_values(key=lambda s: s.abs(), ascending=False).round(4)
    return {
        "target_correlations_sorted_abs": target_corr.to_dict(),
        "correlation_matrix": corr.round(3).to_dict(),
    }


def baseline_model(df: pd.DataFrame) -> Dict[str, Any]:
    validate_dataset(df)
    clean_df = df.dropna().drop_duplicates()
    X = clean_df.drop(columns=[TARGET_COL])
    y = clean_df[TARGET_COL]
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    proba = lr.predict_proba(X_test)[:, 1]

    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).round(4)

    return {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "logistic_regression": {
            "accuracy": _round_float(accuracy_score(y_test, pred)),
            "macro_f1": _round_float(f1_score(y_test, pred, average="macro")),
            "roc_auc": _round_float(roc_auc_score(y_test, proba)),
            "classification_report": classification_report(y_test, pred, output_dict=True),
        },
        "random_forest": {
            "accuracy": _round_float(accuracy_score(y_test, rf_pred)),
            "macro_f1": _round_float(f1_score(y_test, rf_pred, average="macro")),
            "roc_auc": _round_float(roc_auc_score(y_test, rf_proba)),
            "top_feature_importances": importances.head(8).to_dict(),
        },
    }


def risk_segments(df: pd.DataFrame) -> Dict[str, Any]:
    validate_dataset(df)
    work = df.copy()
    work["age_group"] = pd.cut(work["age"], bins=[0, 40, 50, 60, 200], labels=["<=40", "41-50", "51-60", "60+"])
    segment_cols = ["age_group", "sex", "cp", "exang", "thal"]
    rows: List[Dict[str, Any]] = []
    for col in segment_cols:
        tmp = work.groupby(col, observed=True)[TARGET_COL].agg(["count", "mean"]).reset_index()
        tmp = tmp[tmp["count"] >= 15]
        for _, r in tmp.iterrows():
            rows.append({
                "segment": col,
                "value": str(r[col]),
                "count": int(r["count"]),
                "target_rate": round(float(r["mean"]), 4),
            })
    rows = sorted(rows, key=lambda x: x["target_rate"], reverse=True)
    return {"segments_sorted_by_target_rate": rows[:15]}


def run_selected_tools(df: pd.DataFrame, tool_names: List[str]) -> Dict[str, Any]:
    registry = {
        "overview": dataframe_summary,
        "target_groups": target_group_statistics,
        "correlations": correlation_analysis,
        "baseline_model": baseline_model,
        "risk_segments": risk_segments,
    }
    results = {}
    for name in tool_names:
        if name in registry:
            results[name] = registry[name](df)
    return results


def available_tools() -> Dict[str, str]:
    return {
        "overview": "Dataset dimensions, missing values, duplicates, target distribution, descriptive statistics.",
        "target_groups": "Mean feature comparison between target=0 and target=1.",
        "correlations": "Correlations with target and full numeric correlation matrix.",
        "baseline_model": "Train/test baseline Logistic Regression and Random Forest with metrics and feature importances.",
        "risk_segments": "Simple age/sex/chest-pain/exercise-angina/thal segments sorted by target rate.",
    }


def _fig_to_base64() -> str:
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_charts(df: pd.DataFrame) -> Dict[str, str]:
    validate_dataset(df)
    charts = {}

    plt.figure(figsize=(5, 3.2))
    df[TARGET_COL].value_counts().sort_index().plot(kind="bar")
    plt.title("Target distribution")
    plt.xlabel("target")
    plt.ylabel("count")
    charts["target_distribution"] = _fig_to_base64()

    plt.figure(figsize=(5, 3.2))
    for target_value in sorted(df[TARGET_COL].dropna().unique()):
        df.loc[df[TARGET_COL] == target_value, "age"].plot(kind="hist", bins=15, alpha=0.55, label=f"target={target_value}")
    plt.title("Age distribution by target")
    plt.xlabel("age")
    plt.legend()
    charts["age_by_target"] = _fig_to_base64()

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(7, 5.6))
    im = plt.imshow(corr, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
    plt.title("Correlation matrix")
    charts["correlation_matrix"] = _fig_to_base64()

    model_info = baseline_model(df)["random_forest"]["top_feature_importances"]
    plt.figure(figsize=(6, 3.6))
    pd.Series(model_info).sort_values().plot(kind="barh")
    plt.title("Random Forest feature importance")
    plt.xlabel("importance")
    charts["feature_importance"] = _fig_to_base64()

    return charts
