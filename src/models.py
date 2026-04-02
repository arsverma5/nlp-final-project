import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay,
)

from preprocessing import RecipeProcessor, TIME_LABELS

CSV_PATH  = "../data/raw/recipes_augmented.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
FIGURES_DIR = "../figures"   

import os
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_and_prepare(csv_path: str, processor: RecipeProcessor):
    """
    Load raw CSV and build the full feature matrix + both target vectors.

    Feature matrix X combines:
      - TF-IDF on the merged ingredients + cleaned-directions corpus
      - time_mention_count (# of explicit time expressions in raw directions)
      - ingredient_count (# of parsed ingredients)

    Targets:
      y_reg: total_minutes (float, for regression)
      y_clf: time_bucket (ordered category, for classification)

    Returns
    -------
    X: sparse matrix
    y_reg: pd.Series
    y_clf: pd.Series
    df: cleaned dataframe 
    feature_names : list[str] (TF-IDF terms + scalar names, for importance plots)
    """
    df = pd.read_csv(csv_path)

    # parse target 
    df["total_minutes"] = df["timing"].apply(processor.parse_total_time)
    df = df.dropna(subset=["total_minutes"]).reset_index(drop=True)

    # remove outliers (> 3 days) for more accurate preds
    df = df[df["total_minutes"] <= 4320].reset_index(drop=True)

    # classification target
    df["time_bucket"] = processor.bucketize_time(df["total_minutes"])

    # scalar features 
    # count BEFORE stripping time mentions from directions
    df["time_mention_count"] = df["directions"].apply(processor.count_time_mentions)

    # parse ingredient count via count, no full parse
    df["ingredient_count"] = df["ingredients"].apply(
        lambda x: len(str(x).split(",")) if isinstance(x, str) else 0
    )

    # text features
    print("Build corpus and fit TF-IDF")
    corpus = processor.build_corpus(df["ingredients"], df["directions"])
    tfidf = processor.fit_tfidf(corpus)

    scalar = processor.build_scalar_features(df)
    X = processor.hstack_features(tfidf, scalar)

    tfidf_names = processor.get_feature_names()
    scalar_names = [c for c in ["time_mention_count", "ingredient_count"] if c in df.columns]
    feature_names = tfidf_names + scalar_names

    y_reg = df["total_minutes"].astype(float)
    y_clf = df["time_bucket"].astype(str)   

    print(f"Dataset: {X.shape[0]:,} recipes | {X.shape[1]:,} features")
    return X, y_reg, y_clf, df, feature_names


#regression
# # candidates for regression/classification are dicts so we have the option to do more
# models in the future

def run_regression(X_train, X_test, y_train, y_test):
    """Train & evaluate regression baselines"""
    candidates = {
        "Linear Regression" : LinearRegression(),
    }

    fitted = {}
    print("REGRESSION (target: total minutes)")

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        print(f"\n{name}")
        print(f"MAE (mean absolute error): {mae:.1f} min")
        print(f"RMSE (root mean squre error): {rmse:.1f} min")
        print(f"R^2: {r2:.4f}")
        fitted[name] = (model, preds)

    return fitted


# classification 
def run_classification(X_train, X_test, y_train, y_test):
    """Train & evaluate classification baselines. Returns dict of fitted models."""
    # sklearn logistic regression applies regularization by default, I think L1?
    candidates = {
        "Logistic Regression"  : LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
    }

    fitted = {}
    print("CLASSIFICATION  (target: time bucket)")

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
        f1_wt = f1_score(y_test, preds, average="weighted", zero_division=0)
        print(f"\n{name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        print(f"F1 (wtd): {f1_wt:.4f}")
        print(classification_report(y_test, preds, zero_division=0))
        fitted[name] = (model, preds)

    return fitted


# graphs

def plot_time_distribution(df: pd.DataFrame):
    """Histogram of cooking times + bucket breakdown side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: raw minutes
    cap = df["total_minutes"].quantile(0.98)
    axes[0].hist(df["total_minutes"].clip(upper=cap), bins=60, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Cooking Time Distribution")
    axes[0].set_xlabel("Total Minutes")
    axes[0].set_ylabel("Count")

    # right: bucket bar chart
    bucket_counts = df["time_bucket"].value_counts().reindex(TIME_LABELS)
    axes[1].bar(bucket_counts.index, bucket_counts.values, color="#DD8452", edgecolor="white")
    axes[1].set_title("Recipes per Time Bucket")
    axes[1].set_xlabel("Bucket")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = f"{FIGURES_DIR}/time_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()


def plot_predicted_vs_actual(y_test, preds, model_name: str):
    """Scatter of predicted vs actual cooking times for a regression model."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.3, s=12, color="#4C72B0", label="Predictions")
    lims = [0, max(y_test.max(), np.max(preds))]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Minutes")
    ax.set_ylabel("Predicted Minutes")
    ax.set_title(f"Predicted vs Actual — {model_name}")
    ax.legend()
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    path = f"{FIGURES_DIR}/pred_vs_actual_{safe_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_test, preds, model_name: str):
    """Confusion matrix heatmap for a classification model."""
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, preds,
        display_labels=TIME_LABELS,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = f"{FIGURES_DIR}/confusion_{safe_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def plot_top_features(model, feature_names: list[str], model_name: str, top_n: int = 20):
    """
    Bar chart of the top N most important features.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef[0] if coef.ndim > 1 else coef)
    else:
        print(f"  Skipping feature importance plot for {model_name} (no coef/importances).")
        return

    # safety check - feature_names may be shorter than importances due to scalar append
    n = min(len(importances), len(feature_names))
    importances  = importances[:n]
    feature_names = feature_names[:n]

    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="#55A868",
    )
    ax.set_title(f"Top {top_n} Features — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=","")
    path = f"{FIGURES_DIR}/feature_importance_{safe_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap for numeric recipe features."""
    num_cols = ["total_minutes", "time_mention_count", "ingredient_count", "rating"]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/correlation_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    processor = RecipeProcessor()

    X, y_reg, y_clf, df, feature_names = load_and_prepare(CSV_PATH, processor)

    # shared train/test split
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # graphs on full dataset 
    plot_time_distribution(df)
    plot_correlation_heatmap(df)

    # regression
    reg_models = run_regression(X_train, X_test, yr_train, yr_test)

    for name, (model, preds) in reg_models.items():
        plot_predicted_vs_actual(yr_test, preds, name)
        plot_top_features(model, feature_names, name)

    # classification 
    clf_models = run_classification(X_train, X_test, yc_train, yc_test)

    for name, (model, preds) in clf_models.items():
        plot_confusion_matrix(yc_test, preds, name)
        plot_top_features(model, feature_names, name)

    print("\nDone.")


if __name__ == "__main__":
    main()