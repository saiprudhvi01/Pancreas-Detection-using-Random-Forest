"""
PancreasCare AI â€” ML Training Pipeline
Phase 2-6: Preprocessing, Training, Evaluation, Model Persistence

Target: Stage_at_Diagnosis â†’ mapped to 3-tier risk levels
  Stage I       â†’ Low Risk
  Stage II      â†’ Moderate Risk
  Stage III/IV  â†’ High Risk

Features: 20 pre-diagnosis features only (no data leakage)
Algorithm: Random Forest Classifier
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATASET_PATH = os.path.join(BACKEND_DIR, "dataset", "pancreatic_cancer_prediction_sample.csv")
MODEL_DIR = os.path.join(BASE_DIR)
REPORTS_DIR = os.path.join(BACKEND_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

# â”€â”€ Feature Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# POST-DIAGNOSIS columns to EXCLUDE (data leakage prevention)
LEAKAGE_COLUMNS = [
    "Stage_at_Diagnosis",      # This IS the target source
    "Survival_Time_Months",    # Post-diagnosis outcome
    "Treatment_Type",          # Post-diagnosis clinical decision
    "Survival_Status",         # Post-diagnosis outcome
]

TARGET_SOURCE = "Stage_at_Diagnosis"

# Risk level mapping
STAGE_TO_RISK = {
    "Stage I": "Low Risk",
    "Stage II": "Moderate Risk",
    "Stage III": "High Risk",
    "Stage IV": "High Risk",
}

RISK_LABELS = ["Low Risk", "Moderate Risk", "High Risk"]

# Feature groups (from dataset inspection)
NUMERICAL_FEATURES = ["Age"]

BINARY_FEATURES = [
    "Smoking_History", "Obesity", "Diabetes", "Chronic_Pancreatitis",
    "Family_History", "Hereditary_Condition", "Jaundice",
    "Abdominal_Discomfort", "Back_Pain", "Weight_Loss",
    "Development_of_Type2_Diabetes", "Alcohol_Consumption",
]

CATEGORICAL_FEATURES = [
    "Gender", "Country", "Physical_Activity_Level",
    "Diet_Processed_Food", "Access_to_Healthcare",
    "Urban_vs_Rural", "Economic_Status",
]

ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def load_and_prepare_data():
    """Load dataset, create target, remove leakage columns."""
    print("=" * 60)
    print("PHASE 1: Loading Dataset")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Remove duplicates
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  Removed {n_dups} duplicate rows -> {df.shape[0]} rows remaining")

    # Create target variable
    print("\nPHASE 2: Creating Target Variable")
    print("-" * 40)
    df["risk_level"] = df[TARGET_SOURCE].map(STAGE_TO_RISK)
    print("  Mapping:")
    for stage, risk in STAGE_TO_RISK.items():
        count = (df[TARGET_SOURCE] == stage).sum()
        print(f"    {stage} -> {risk} ({count} records)")

    print(f"\n  Target distribution:")
    for risk in RISK_LABELS:
        count = (df["risk_level"] == risk).sum()
        pct = count / len(df) * 100
        print(f"    {risk}: {count} ({pct:.1f}%)")

    # Select features (exclude leakage columns)
    X = df[ALL_FEATURES].copy()
    y = df["risk_level"].copy()

    print(f"\n  Features selected: {len(ALL_FEATURES)}")
    print(f"  Leakage columns excluded: {LEAKAGE_COLUMNS}")

    return X, y, df


def build_preprocessing_pipeline():
    """Build sklearn ColumnTransformer for preprocessing."""
    print("\nPHASE 3: Building Preprocessing Pipeline")
    print("-" * 40)

    # Numerical: StandardScaler for Age
    # Binary: pass through (already 0/1)
    # Categorical: OrdinalEncoder

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    print(f"  Numerical features ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
    print(f"  Binary features ({len(BINARY_FEATURES)}): {BINARY_FEATURES}")
    print(f"  Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")

    return preprocessor


def train_model(X_train, y_train, preprocessor):
    """Train Random Forest with RandomizedSearchCV."""
    print("\nPHASE 4: Training Random Forest")
    print("-" * 40)

    # Build full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])

    # Hyperparameter search space
    param_distributions = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [10, 15, 20, 25, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    }

    print("  Running RandomizedSearchCV (20 iterations, 3-fold CV)...")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=20,
        cv=3,
        scoring="f1_weighted",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(f"\n  Best CV Score (weighted F1): {search.best_score_:.4f}")
    print(f"  Best Parameters:")
    for param, value in search.best_params_.items():
        print(f"    {param}: {value}")

    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_model(pipeline, X_test, y_test):
    """Complete model evaluation."""
    print("\nPHASE 5: Model Evaluation")
    print("=" * 60)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Core metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    class_labels = pipeline.classes_.tolist()
    precision_per = precision_score(y_test, y_pred, average=None, labels=class_labels, zero_division=0)
    recall_per = recall_score(y_test, y_pred, average=None, labels=class_labels, zero_division=0)
    f1_per = f1_score(y_test, y_pred, average=None, labels=class_labels, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    # ROC-AUC (One-vs-Rest for multiclass)
    try:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=class_labels)
        roc_auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        roc_auc_weighted = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
    except Exception:
        roc_auc_macro = None
        roc_auc_weighted = None

    # Classification report
    report_str = classification_report(y_test, y_pred, labels=class_labels, zero_division=0)

    # Feature importance
    rf_model = pipeline.named_steps["classifier"]
    feature_importances = rf_model.feature_importances_

    # Get feature names after transformation
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = (
        NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    )

    importance_dict = dict(zip(feature_names, feature_importances.tolist()))
    importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Print results
    print(f"\n  Accuracy:           {accuracy:.4f}")
    print(f"  Precision (macro):  {precision_macro:.4f}")
    print(f"  Recall (macro):     {recall_macro:.4f}")
    print(f"  F1 Score (macro):   {f1_macro:.4f}")
    print(f"  Precision (weighted): {precision_weighted:.4f}")
    print(f"  Recall (weighted):    {recall_weighted:.4f}")
    print(f"  F1 Score (weighted):  {f1_weighted:.4f}")
    if roc_auc_macro is not None:
        print(f"  ROC-AUC (macro):    {roc_auc_macro:.4f}")
        print(f"  ROC-AUC (weighted): {roc_auc_weighted:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"  Labels: {class_labels}")
    for i, row in enumerate(cm):
        print(f"    {class_labels[i]}: {row.tolist()}")

    print(f"\n  Classification Report:")
    print(report_str)

    print(f"\n  Top 10 Feature Importances:")
    for name, imp in importance_sorted[:10]:
        bar = "â–ˆ" * int(imp * 100)
        print(f"    {name:35s} {imp:.4f} {bar}")

    # Build metrics dictionary
    per_class_metrics = {}
    for i, label in enumerate(class_labels):
        per_class_metrics[label] = {
            "precision": round(float(precision_per[i]), 4),
            "recall": round(float(recall_per[i]), 4),
            "f1_score": round(float(f1_per[i]), 4),
            "support": int(cm[i].sum()),
        }

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "roc_auc_macro": round(float(roc_auc_macro), 4) if roc_auc_macro else None,
        "roc_auc_weighted": round(float(roc_auc_weighted), 4) if roc_auc_weighted else None,
        "confusion_matrix": cm.tolist(),
        "class_labels": class_labels,
        "per_class_metrics": per_class_metrics,
        "classification_report": report_str,
        "feature_importance": importance_sorted,
    }

    return metrics


def save_artifacts(pipeline, metrics, best_params, cv_score, X, y, df):
    """Save model, metadata, and evaluation reports."""
    print("\nPHASE 6: Saving Artifacts")
    print("=" * 60)

    # 1. Save complete pipeline (includes preprocessor + model)
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"  [OK] Pipeline saved: {model_path}")

    # 2. Compute categorical feature metadata from the DATA (not from the pipeline)
    cat_options = {}
    for feat in CATEGORICAL_FEATURES:
        cat_options[feat] = sorted(df[feat].dropna().unique().tolist())

    # 3. Feature metadata for API
    feature_metadata = {
        "numerical_features": NUMERICAL_FEATURES,
        "binary_features": BINARY_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURES,
        "feature_labels": {
            "Age": "Age (years)",
            "Gender": "Gender",
            "Country": "Country",
            "Smoking_History": "Smoking History",
            "Obesity": "Obesity",
            "Diabetes": "Diabetes",
            "Chronic_Pancreatitis": "Chronic Pancreatitis",
            "Family_History": "Family History of Pancreatic Cancer",
            "Hereditary_Condition": "Hereditary Genetic Condition",
            "Jaundice": "Jaundice",
            "Abdominal_Discomfort": "Abdominal Discomfort / Pain",
            "Back_Pain": "Back Pain",
            "Weight_Loss": "Unexplained Weight Loss",
            "Development_of_Type2_Diabetes": "Recent Development of Type 2 Diabetes",
            "Alcohol_Consumption": "Regular Alcohol Consumption",
            "Physical_Activity_Level": "Physical Activity Level",
            "Diet_Processed_Food": "Processed Food in Diet",
            "Access_to_Healthcare": "Access to Healthcare",
            "Urban_vs_Rural": "Living Area",
            "Economic_Status": "Economic Status",
        },
        "feature_descriptions": {
            "Age": "Your current age in years",
            "Gender": "Your biological sex",
            "Country": "Country of residence",
            "Smoking_History": "Do you have a history of smoking?",
            "Obesity": "Have you been diagnosed with obesity (BMI â‰¥ 30)?",
            "Diabetes": "Have you been diagnosed with diabetes?",
            "Chronic_Pancreatitis": "Have you been diagnosed with chronic pancreatitis?",
            "Family_History": "Does your family have a history of pancreatic cancer?",
            "Hereditary_Condition": "Do you have a hereditary genetic condition?",
            "Jaundice": "Have you experienced jaundice (yellowing of skin/eyes)?",
            "Abdominal_Discomfort": "Do you experience persistent abdominal discomfort or pain?",
            "Back_Pain": "Do you experience persistent back pain?",
            "Weight_Loss": "Have you experienced unexplained weight loss?",
            "Development_of_Type2_Diabetes": "Have you recently developed Type 2 Diabetes?",
            "Alcohol_Consumption": "Do you regularly consume alcohol?",
            "Physical_Activity_Level": "How would you describe your physical activity level?",
            "Diet_Processed_Food": "How much processed food is in your diet?",
            "Access_to_Healthcare": "How would you rate your access to healthcare?",
            "Urban_vs_Rural": "Do you live in an urban or rural area?",
            "Economic_Status": "What is your economic status?",
        },
        "categorical_options": cat_options,
        "numerical_ranges": {
            "Age": {"min": 30, "max": 90, "step": 1},
        },
        "feature_groups": {
            "Personal Information": ["Age", "Gender", "Country", "Urban_vs_Rural", "Economic_Status"],
            "Medical History": ["Diabetes", "Chronic_Pancreatitis", "Family_History",
                               "Hereditary_Condition", "Development_of_Type2_Diabetes"],
            "Symptoms": ["Jaundice", "Abdominal_Discomfort", "Back_Pain", "Weight_Loss"],
            "Lifestyle & Risk Factors": ["Smoking_History", "Obesity", "Alcohol_Consumption",
                                          "Physical_Activity_Level", "Diet_Processed_Food",
                                          "Access_to_Healthcare"],
        },
    }

    metadata_path = os.path.join(MODEL_DIR, "feature_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(feature_metadata, f, indent=2)
    print(f"  âœ“ Feature metadata saved: {metadata_path}")

    # 4. Training configuration
    training_config = {
        "algorithm": "Random Forest Classifier",
        "library": "scikit-learn",
        "dataset": "pancreatic_cancer_prediction_sample.csv",
        "dataset_rows": len(df),
        "dataset_columns": df.shape[1],
        "features_used": len(ALL_FEATURES),
        "target_source": TARGET_SOURCE,
        "risk_mapping": STAGE_TO_RISK,
        "risk_labels": RISK_LABELS,
        "classification_type": "multiclass (3 classes)",
        "train_test_split": 0.2,
        "best_hyperparameters": best_params,
        "cv_score": round(float(cv_score), 4),
        "training_date": datetime.now().isoformat(),
        "leakage_columns_excluded": LEAKAGE_COLUMNS,
        "class_distribution": {
            label: int((y == label).sum()) for label in RISK_LABELS
        },
    }

    config_path = os.path.join(MODEL_DIR, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2)
    print(f"  âœ“ Training config saved: {config_path}")

    # 5. Evaluation metrics
    metrics_path = os.path.join(MODEL_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  âœ“ Evaluation metrics saved: {metrics_path}")

    # 6. Evaluation report (text)
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("PancreasCare AI â€” Model Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Algorithm: Random Forest Classifier\n")
        f.write(f"Dataset: {len(df)} records, {len(ALL_FEATURES)} features\n")
        f.write(f"Target: {TARGET_SOURCE} â†’ 3-tier risk levels\n\n")
        f.write(f"Accuracy: {metrics['accuracy']}\n")
        f.write(f"Precision (macro): {metrics['precision_macro']}\n")
        f.write(f"Recall (macro): {metrics['recall_macro']}\n")
        f.write(f"F1 (macro): {metrics['f1_macro']}\n")
        f.write(f"F1 (weighted): {metrics['f1_weighted']}\n")
        if metrics.get("roc_auc_macro"):
            f.write(f"ROC-AUC (macro): {metrics['roc_auc_macro']}\n")
        f.write(f"\nClassification Report:\n{metrics['classification_report']}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"Labels: {metrics['class_labels']}\n")
        for i, row in enumerate(metrics['confusion_matrix']):
            f.write(f"  {metrics['class_labels'][i]}: {row}\n")
        f.write(f"\nFeature Importances:\n")
        for name, imp in metrics['feature_importance']:
            f.write(f"  {name}: {imp:.4f}\n")
    print(f"  âœ“ Report saved: {report_path}")

    print("\n  All artifacts saved successfully!")
    return model_path


def main():
    """Run complete training pipeline."""
    print("\n" + "=" * 60)
    print("  PancreasCare AI â€” ML Training Pipeline")
    print("=" * 60 + "\n")

    # Phase 1-2: Load and prepare data
    X, y, df = load_and_prepare_data()

    # Phase 3: Build preprocessing
    preprocessor = build_preprocessing_pipeline()

    # IMPORTANT: Split BEFORE fitting preprocessor (no data leakage)
    print("\n  Train/Test Split: 80/20 (stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")

    # Phase 4: Train model
    pipeline, best_params, cv_score = train_model(X_train, y_train, preprocessor)

    # Phase 5: Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)

    # Phase 6: Save
    save_artifacts(pipeline, metrics, best_params, cv_score, X, y, df)

    print("\n" + "=" * 60)
    print("  Training Pipeline Complete!")
    print("=" * 60 + "\n")

    return pipeline, metrics


if __name__ == "__main__":
    main()

