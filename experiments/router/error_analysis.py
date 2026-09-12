import json
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
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


def main():
    texts, labels = load_data()
    model = build_model()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Cada previsão é feita por um modelo que NÃO treinou
    # com aquele exemplo.
    predictions = cross_val_predict(
        model,
        texts,
        labels,
        cv=cv,
    )

    print("\n" + "=" * 70)
    print("OUT-OF-FOLD ERROR ANALYSIS")
    print("=" * 70)

    accuracy = accuracy_score(labels, predictions)

    print(f"\nAccuracy OOF: {accuracy:.3f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            labels,
            predictions,
            digits=3,
        )
    )

    label_order = ["FAST_PATH", "AGENT"]

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=label_order,
    )

    print("Confusion Matrix")
    print(f"Labels: {label_order}")
    print(matrix)

    print("\n" + "=" * 70)
    print("ERROS")
    print("=" * 70)

    errors = []

    for text, expected, predicted in zip(
        texts,
        labels,
        predictions,
    ):
        if expected != predicted:
            errors.append(
                {
                    "text": text,
                    "expected": expected,
                    "predicted": predicted,
                }
            )

    for index, error in enumerate(errors, start=1):

        print(f"\nErro #{index}")
        print(f"Query     : {error['text']}")
        print(f"Esperado  : {error['expected']}")
        print(f"Predito   : {error['predicted']}")

    print("\n" + "=" * 70)
    print(f"Total de erros: {len(errors)}")
    print("=" * 70)


if __name__ == "__main__":
    main()