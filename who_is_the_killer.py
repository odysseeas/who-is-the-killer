"""Who Is The Killer? - Pattern Recognition course project (University of Piraeus).

Predicts which of 8 serial killers committed each crime in the "Piraeus Vice"
dataset, comparing a generative model (Gaussian Naive Bayes) with discriminative
ones (Logistic Regression, SVM, MLP), and explores the data with PCA and k-means.

Usage:
    python who_is_the_killer.py data/crimes.csv
    python who_is_the_killer.py data/crimes.csv --out figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

CONTINUOUS = [
    "hour_float", "latitude", "longitude", "victim_age",
    "temp_c", "humidity", "dist_precinct_km", "pop_density",
]
CATEGORICAL = ["weapon_code", "scene_type", "weather", "vic_gender"]
FEATURES = CONTINUOUS + CATEGORICAL
TARGET = "killer_id"
SEED = 42


def make_preprocessor():
    """Scale continuous features and one-hot encode categorical ones."""
    return ColumnTransformer([
        ("num", StandardScaler(), CONTINUOUS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def make_models():
    """Fresh, independent pipelines (each model gets its own preprocessor)."""
    return {
        # Naive Bayes assumes Gaussian features, so it only gets the continuous ones.
        "Gaussian Naive Bayes": make_pipeline(
            ColumnTransformer([("num", StandardScaler(), CONTINUOUS)]), GaussianNB()
        ),
        "Logistic Regression": make_pipeline(
            make_preprocessor(), LogisticRegression(max_iter=1000)
        ),
        "MLP (100, 50)": make_pipeline(
            make_preprocessor(),
            MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=SEED),
        ),
        "SVM (RBF kernel)": make_pipeline(
            make_preprocessor(), SVC(kernel="rbf", random_state=SEED)
        ),
        # Same SVM, but mistakes on rare killers cost more (weights inverse to class size).
        "SVM (RBF, balanced weights)": make_pipeline(
            make_preprocessor(), SVC(kernel="rbf", class_weight="balanced", random_state=SEED)
        ),
    }


def save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=120)
    plt.close(fig)


def exploratory_plots(train, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=train, x=TARGET, hue=TARGET, palette="viridis", legend=False, ax=ax)
    ax.set(title="Crimes per killer (training set)", xlabel="Killer", ylabel="Crimes")
    save(fig, out_dir, "killer_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=train, x="longitude", y="latitude", hue=TARGET,
                    palette="tab10", s=25, alpha=0.7, ax=ax)
    ax.set(title="Crime locations by killer")
    save(fig, out_dir, "spatial_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(train["victim_age"], kde=True, color="skyblue", ax=axes[0])
    axes[0].set(title="Victim age")
    sns.histplot(train["hour_float"], kde=True, color="orange", ax=axes[1])
    axes[1].set(title="Hour of crime")
    save(fig, out_dir, "feature_histograms.png")

    # Gaussian MLE for one class: the sample covariance of killer 3's crimes.
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(train.loc[train[TARGET] == 3, CONTINUOUS].cov(),
                annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set(title="Covariance matrix (killer 3)")
    save(fig, out_dir, "covariance_killer3.png")


def evaluate_models(train, val, out_dir):
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]

    rows, fitted = [], {}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        fitted[name] = (model, pred)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_val, pred),
            "Balanced accuracy": balanced_accuracy_score(y_val, pred),
            "Macro F1": f1_score(y_val, pred, average="macro"),
        })

    results = pd.DataFrame(rows).set_index("Model").round(4)
    print("\nValidation results\n", results.to_string(), sep="")
    results.to_csv(out_dir / "results.csv")

    # The best model is chosen by macro F1, so small killers count as much as big ones.
    best = results["Macro F1"].idxmax()
    model, pred = fitted[best]
    labels = sorted(y_val.unique())
    print(f"\nPer-class report for {best}\n")
    print(classification_report(y_val, pred, labels=labels, digits=3))

    cm = confusion_matrix(y_val, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(title=f"Confusion matrix - {best}", xlabel="Predicted killer", ylabel="True killer")
    save(fig, out_dir, "confusion_matrix.png")
    return best


def unsupervised(train, out_dir):
    X = make_preprocessor().fit_transform(train[FEATURES])
    X = X.toarray() if hasattr(X, "toarray") else X
    y = train[TARGET]

    pca = PCA().fit(X)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), pca.explained_variance_[:10], marker="o", linestyle="--")
    ax.set(title="Scree plot (first 10 eigenvalues)", xlabel="Component", ylabel="Eigenvalue")
    save(fig, out_dir, "pca_eigenvalues.png")

    X_2d = PCA(n_components=2).fit_transform(X)
    clusters = KMeans(n_clusters=8, random_state=SEED, n_init=10).fit_predict(X)

    # How well do the clusters match the real killers? 1 = perfect, 0 = random.
    ari = adjusted_rand_score(y, clusters)
    nmi = normalized_mutual_info_score(y, clusters)
    print(f"k-means (k=8) vs true killers: ARI = {ari:.3f}, NMI = {nmi:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y, palette="tab10",
                    s=25, alpha=0.7, ax=axes[0])
    axes[0].set(title="PCA - true killers")
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=clusters, palette="viridis",
                    s=25, alpha=0.7, ax=axes[1])
    axes[1].set(title=f"PCA - k-means clusters (ARI = {ari:.2f})")
    save(fig, out_dir, "pca_kmeans_comparison.png")


def predict_test(train, val, test, model_name, out_dir):
    """Retrain the best model on TRAIN + VAL and predict the unsolved TEST crimes."""
    full = pd.concat([train, val])
    model = make_models()[model_name]
    model.fit(full[FEATURES], full[TARGET])
    submission = pd.DataFrame({
        "incident_id": test["incident_id"],
        TARGET: model.predict(test[FEATURES]),
    })
    submission.to_csv(out_dir / "submission.csv", index=False)
    print(f"Saved {len(submission)} predictions to {out_dir / 'submission.csv'}")

    # If the TEST crimes come with their true killer, score the final model once on them.
    if test[TARGET].notna().all():
        truth, pred = test[TARGET].astype(int), submission[TARGET]
        print(f"TEST ({model_name}, trained on TRAIN + VAL): "
              f"accuracy = {accuracy_score(truth, pred):.4f}, "
              f"balanced accuracy = {balanced_accuracy_score(truth, pred):.4f}, "
              f"macro F1 = {f1_score(truth, pred, average='macro'):.4f}")


def location_baseline(train, val):
    """How far do we get with latitude and longitude alone?"""
    where = ["latitude", "longitude"]
    model = make_pipeline(ColumnTransformer([("num", StandardScaler(), where)]),
                          SVC(kernel="rbf", random_state=SEED))
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(val[FEATURES])
    print(f"Location-only SVM (VAL): accuracy = {accuracy_score(val[TARGET], pred):.4f}, "
          f"macro F1 = {f1_score(val[TARGET], pred, average='macro'):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="path to the Piraeus Vice CSV file")
    parser.add_argument("--out", type=Path, default=Path("figures"),
                        help="folder for figures and results (default: figures)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    train = df[df["split"] == "TRAIN"].copy()
    val = df[df["split"] == "VAL"].copy()
    test = df[df["split"] == "TEST"].copy()
    # TEST crimes are unsolved (empty killer_id), which makes pandas read the column as
    # float. Killer ids are whole numbers, so convert them back to int.
    for part in (train, val):
        part[TARGET] = part[TARGET].astype(int)
    print(f"Loaded {len(df)} crimes: {len(train)} train, {len(val)} validation, {len(test)} test")

    args.out.mkdir(parents=True, exist_ok=True)
    exploratory_plots(train, args.out)
    best = evaluate_models(train, val, args.out)
    location_baseline(train, val)
    unsupervised(train, args.out)
    predict_test(train, val, test, best, args.out)


if __name__ == "__main__":
    main()
