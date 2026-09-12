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

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for fold, (train_index, test_index) in enumerate(
        cv.split(texts, labels),
        start=1,
    ):

        model = build_model()

        x_train = texts[train_index]
        x_test = texts[test_index]

        y_train = labels[train_index]
        y_test = labels[test_index]

        model.fit(x_train, y_train)

        predictions = model.predict(x_test)

        scores = model.decision_function(x_test)

        classes = model.named_steps["classifier"].classes_

        for text, expected, predicted, score in zip(
            x_test,
            y_test,
            predictions,
            scores,
        ):

            # Para classificação binária do LinearSVC,
            # o sinal indica o lado da fronteira.
            # A magnitude indica a distância da fronteira.
            distance = abs(float(score))

            results.append(
                {
                    "fold": fold,
                    "text": text,
                    "expected": expected,
                    "predicted": predicted,
                    "score": float(score),
                    "distance": distance,
                    "correct": expected == predicted,
                    "classes": classes.tolist(),
                }
            )

    results = sorted(
        results,
        key=lambda item: item["distance"],
    )

    print("\nClasses do modelo:")
    print(results[0]["classes"])

    print("\n" + "=" * 90)
    print("QUERIES MAIS PRÓXIMAS DA FRONTEIRA")
    print("=" * 90)

    for item in results[:15]:

        status = "OK" if item["correct"] else "ERRO"

        print(
            f"\n[{status}] "
            f"distance={item['distance']:.4f} "
            f"score={item['score']:.4f}"
        )

        print(f"Query    : {item['text']}")
        print(f"Esperado : {item['expected']}")
        print(f"Predito  : {item['predicted']}")

    errors = [
        item
        for item in results
        if not item["correct"]
    ]

    print("\n" + "=" * 90)
    print("CONFIANÇA DOS ERROS")
    print("=" * 90)

    for item in errors:

        print(
            f"\ndistance={item['distance']:.4f} "
            f"score={item['score']:.4f}"
        )

        print(f"Query    : {item['text']}")
        print(f"Esperado : {item['expected']}")
        print(f"Predito  : {item['predicted']}")

    correct_distances = [
        item["distance"]
        for item in results
        if item["correct"]
    ]

    error_distances = [
        item["distance"]
        for item in results
        if not item["correct"]
    ]

    print("\n" + "=" * 90)
    print("RESUMO")
    print("=" * 90)

    print(
        f"Distância média - acertos: "
        f"{np.mean(correct_distances):.4f}"
    )

    print(
        f"Distância média - erros:   "
        f"{np.mean(error_distances):.4f}"
    )

    print(
        f"Mediana - acertos: "
        f"{np.median(correct_distances):.4f}"
    )

    print(
        f"Mediana - erros:   "
        f"{np.median(error_distances):.4f}"
    )


if __name__ == "__main__":
    main()