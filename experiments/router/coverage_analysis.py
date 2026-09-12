import json
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "router_training_data.json"

RANDOM_STATE = 42
N_SPLITS = 5

THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
]


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = np.array([item["text"] for item in data])
    labels = np.array([item["label"] for item in data])

    return texts, labels


def build_model():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="word",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def get_oof_results(texts, labels):

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for train_index, test_index in cv.split(texts, labels):

        model = build_model()

        x_train = texts[train_index]
        x_test = texts[test_index]

        y_train = labels[train_index]
        y_test = labels[test_index]

        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        scores = model.decision_function(x_test)

        for text, expected, predicted, score in zip(
            x_test,
            y_test,
            predictions,
            scores,
        ):
            results.append(
                {
                    "text": text,
                    "expected": expected,
                    "predicted": predicted,
                    "distance": abs(float(score)),
                    "correct": expected == predicted,
                }
            )

    return results


def evaluate_threshold(results, threshold):

    local = [
        item
        for item in results
        if item["distance"] >= threshold
    ]

    fallback = [
        item
        for item in results
        if item["distance"] < threshold
    ]

    total = len(results)

    total_errors = sum(
        not item["correct"]
        for item in results
    )

    local_errors = sum(
        not item["correct"]
        for item in local
    )

    captured_errors = sum(
        not item["correct"]
        for item in fallback
    )

    coverage = len(local) / total

    fallback_rate = len(fallback) / total

    if local:
        local_accuracy = (
            sum(item["correct"] for item in local)
            / len(local)
        )
    else:
        local_accuracy = 0.0

    if total_errors:
        error_capture_rate = (
            captured_errors / total_errors
        )
    else:
        error_capture_rate = 0.0

    return {
        "threshold": threshold,
        "coverage": coverage,
        "fallback_rate": fallback_rate,
        "local_accuracy": local_accuracy,
        "captured_errors": captured_errors,
        "local_errors": local_errors,
        "error_capture_rate": error_capture_rate,
    }


def main():

    texts, labels = load_data()

    results = get_oof_results(
        texts,
        labels,
    )

    print("\n" + "=" * 105)
    print("COVERAGE x FALLBACK ANALYSIS")
    print("=" * 105)

    print(
        f"{'THRESHOLD':>10}"
        f"{'COVERAGE':>12}"
        f"{'FALLBACK':>12}"
        f"{'LOCAL ACC':>12}"
        f"{'ERR CAP':>12}"
        f"{'ERR LEFT':>12}"
        f"{'CAP RATE':>12}"
    )

    print("-" * 105)

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            results,
            threshold,
        )

        print(
            f"{result['threshold']:>10.2f}"
            f"{result['coverage']:>12.1%}"
            f"{result['fallback_rate']:>12.1%}"
            f"{result['local_accuracy']:>12.1%}"
            f"{result['captured_errors']:>12}"
            f"{result['local_errors']:>12}"
            f"{result['error_capture_rate']:>12.1%}"
        )

    print("=" * 105)


if __name__ == "__main__":
    main()