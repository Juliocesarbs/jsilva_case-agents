import json
import time
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "router_training_data.json"

RANDOM_STATE = 42
N_SPLITS = 5


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = np.array([item["text"] for item in data])
    labels = np.array([item["label"] for item in data])

    return texts, labels


def word_tfidf():
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def char_tfidf():
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )


def build_models():
    return {
        "word_tfidf": Pipeline(
            [
                ("features", word_tfidf()),
                (
                    "classifier",
                    LinearSVC(random_state=RANDOM_STATE),
                ),
            ]
        ),

        "char_tfidf": Pipeline(
            [
                ("features", char_tfidf()),
                (
                    "classifier",
                    LinearSVC(random_state=RANDOM_STATE),
                ),
            ]
        ),

        "word_char_tfidf": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", word_tfidf()),
                            ("char", char_tfidf()),
                        ]
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(random_state=RANDOM_STATE),
                ),
            ]
        ),
    }


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
    print("Classificador: LinearSVC")
    print("\nExecutando benchmark de features...\n")

    models = build_models()

    results = []

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
        f"{'FEATURES':25}"
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