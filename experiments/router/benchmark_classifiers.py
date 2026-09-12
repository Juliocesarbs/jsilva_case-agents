import json
import time
from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "router_training_data.json"

RANDOM_STATE = 42
N_SPLITS = 5


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]

    return np.array(texts), np.array(labels)


def build_models():

    tfidf_params = {
        "lowercase": True,
        "strip_accents": "unicode",
        "ngram_range": (1, 2),
        "sublinear_tf": True,
    }

    models = {
        "logistic_regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "linear_svm": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                (
                    "classifier",
                    LinearSVC(
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "random_forest": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_model(name, model, texts, labels):

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    accuracies = []
    f1_scores = []
    inference_times = []

    for train_index, test_index in cv.split(texts, labels):

        x_train = texts[train_index]
        x_test = texts[test_index]

        y_train = labels[train_index]
        y_test = labels[test_index]

        model.fit(x_train, y_train)

        start = time.perf_counter()

        predictions = model.predict(x_test)

        elapsed_ms = (time.perf_counter() - start) * 1000

        accuracies.append(
            accuracy_score(y_test, predictions)
        )

        f1_scores.append(
            f1_score(
                y_test,
                predictions,
                pos_label="AGENT",
            )
        )

        inference_times.append(elapsed_ms)

    return {
        "model": name,
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "f1_mean": np.mean(f1_scores),
        "inference_ms_mean": np.mean(inference_times),
    }


def main():

    texts, labels = load_data()

    print(f"\nDataset: {len(texts)} exemplos")

    unique, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique, counts):
        print(f"{label}: {count}")

    models = build_models()

    results = []

    print("\nExecutando benchmark...\n")

    for name, model in models.items():

        result = evaluate_model(
            name,
            model,
            texts,
            labels,
        )

        results.append(result)

    results = sorted(
        results,
        key=lambda item: item["accuracy_mean"],
        reverse=True,
    )

    print("=" * 75)

    print(
        f"{'MODEL':25}"
        f"{'ACC':>10}"
        f"{'STD':>10}"
        f"{'F1':>10}"
        f"{'MS':>10}"
    )

    print("=" * 75)

    for result in results:

        print(
            f"{result['model']:25}"
            f"{result['accuracy_mean']:>10.3f}"
            f"{result['accuracy_std']:>10.3f}"
            f"{result['f1_mean']:>10.3f}"
            f"{result['inference_ms_mean']:>10.3f}"
        )

    print("=" * 75)


if __name__ == "__main__":
    main()