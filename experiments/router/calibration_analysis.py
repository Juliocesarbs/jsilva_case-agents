import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3

THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAINING_DATA_PATH = (
    ROOT_DIR
    / "data"
    / "router_training_data.json"
)


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

def load_training_data():
    with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]

    return np.array(texts), np.array(labels)


# ============================================================
# MODELO
# ============================================================

def build_base_pipeline():
    """
    Pipeline escolhido nos experimentos anteriores:
    Word TF-IDF + Linear SVM.
    """

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
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


def build_calibrated_model():
    """
    Calibra o LinearSVC usando CV interna.

    sigmoid:
        Platt Scaling.

    É uma escolha conservadora para dataset pequeno.
    """

    base_pipeline = build_base_pipeline()

    model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=INNER_SPLITS,
    )

    return model


# ============================================================
# NESTED CROSS-VALIDATION
# ============================================================

def run_nested_cv(texts, labels):
    """
    Outer CV:
        mede generalização.

    Inner CV:
        utilizada internamente pelo CalibratedClassifierCV.

    Portanto o exemplo usado para avaliação externa nunca participa
    da calibração do modelo que o classificou.
    """

    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_true = []
    all_pred = []
    all_confidence = []
    all_prob_agent = []
    all_texts = []
    all_latency_ms = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(texts, labels),
        start=1,
    ):
        X_train = texts[train_idx]
        X_test = texts[test_idx]

        y_train = labels[train_idx]
        y_test = labels[test_idx]

        model = build_calibrated_model()

        model.fit(X_train, y_train)

        for text, true_label in zip(X_test, y_test):
            start = time.perf_counter()

            probabilities = model.predict_proba([text])[0]

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            predicted_index = np.argmax(probabilities)

            predicted_label = model.classes_[predicted_index]

            confidence = probabilities[predicted_index]

            agent_index = list(model.classes_).index("AGENT")
            prob_agent = probabilities[agent_index]

            all_true.append(true_label)
            all_pred.append(predicted_label)
            all_confidence.append(confidence)
            all_prob_agent.append(prob_agent)
            all_texts.append(text)
            all_latency_ms.append(latency_ms)

        print(f"Fold {fold}/{OUTER_SPLITS} concluído.")

    return pd.DataFrame(
        {
            "text": all_texts,
            "true_label": all_true,
            "predicted_label": all_pred,
            "confidence": all_confidence,
            "prob_agent": all_prob_agent,
            "latency_ms": all_latency_ms,
        }
    )


# ============================================================
# MÉTRICAS DE CALIBRAÇÃO
# ============================================================

def evaluate_calibration(results):
    y_true = results["true_label"].values
    y_pred = results["predicted_label"].values

    y_true_binary = (
        results["true_label"] == "AGENT"
    ).astype(int)

    prob_agent = results["prob_agent"].values

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    brier = brier_score_loss(
        y_true_binary,
        prob_agent,
    )

    probabilities_2d = np.column_stack(
        [
            1 - prob_agent,
            prob_agent,
        ]
    )

    loss = log_loss(
        y_true_binary,
        probabilities_2d,
        labels=[0, 1],
    )

    print("\n" + "=" * 70)
    print("RESULTADOS DO SVM CALIBRADO")
    print("=" * 70)

    print(f"\nAccuracy OOF: {accuracy:.3f}")
    print(f"Brier Score:  {brier:.4f}")
    print(f"Log Loss:     {loss:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            digits=3,
        )
    )

    print(
        f"Latência média de inferência: "
        f"{results['latency_ms'].mean():.3f} ms"
    )


# ============================================================
# ACERTOS VS ERROS
# ============================================================

def analyze_confidence(results):
    results = results.copy()

    results["correct"] = (
        results["true_label"]
        == results["predicted_label"]
    )

    correct = results[
        results["correct"]
    ]

    errors = results[
        ~results["correct"]
    ]

    print("\n" + "=" * 70)
    print("CONFIANÇA — ACERTOS VS ERROS")
    print("=" * 70)

    print(
        f"\nConfiança média - acertos: "
        f"{correct['confidence'].mean():.4f}"
    )

    if len(errors) > 0:
        print(
            f"Confiança média - erros:   "
            f"{errors['confidence'].mean():.4f}"
        )

    print(
        f"\nMediana - acertos: "
        f"{correct['confidence'].median():.4f}"
    )

    if len(errors) > 0:
        print(
            f"Mediana - erros:   "
            f"{errors['confidence'].median():.4f}"
        )

    print("\nCasos ordenados por menor confiança:")

    display_columns = [
        "confidence",
        "true_label",
        "predicted_label",
        "correct",
        "text",
    ]

    print(
        results
        .sort_values("confidence")
        [display_columns]
        .to_string(index=False)
    )


# ============================================================
# COVERAGE × FALLBACK
# ============================================================

def evaluate_abstention(results):
    results = results.copy()

    results["correct"] = (
        results["true_label"]
        == results["predicted_label"]
    )

    total_errors = (
        ~results["correct"]
    ).sum()

    rows = []

    for threshold in THRESHOLDS:

        local_mask = (
            results["confidence"]
            >= threshold
        )

        fallback_mask = ~local_mask

        local_results = results[
            local_mask
        ]

        fallback_results = results[
            fallback_mask
        ]

        coverage = (
            len(local_results)
            / len(results)
        )

        fallback_rate = (
            len(fallback_results)
            / len(results)
        )

        if len(local_results) > 0:
            local_accuracy = (
                local_results["correct"].mean()
            )
        else:
            local_accuracy = np.nan

        errors_captured = (
            ~fallback_results["correct"]
        ).sum()

        errors_left = (
            ~local_results["correct"]
        ).sum()

        if total_errors > 0:
            capture_rate = (
                errors_captured
                / total_errors
            )
        else:
            capture_rate = 0

        rows.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "fallback": fallback_rate,
                "local_accuracy": local_accuracy,
                "errors_captured": errors_captured,
                "errors_left": errors_left,
                "capture_rate": capture_rate,
            }
        )

    summary = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print("COVERAGE × FALLBACK — PROBABILIDADE CALIBRADA")
    print("=" * 90)

    print(
        f"\n{'THRESHOLD':>10}"
        f"{'COVERAGE':>12}"
        f"{'FALLBACK':>12}"
        f"{'LOCAL ACC':>14}"
        f"{'ERR CAP':>12}"
        f"{'ERR LEFT':>12}"
        f"{'CAP RATE':>12}"
    )

    for _, row in summary.iterrows():

        local_acc = row["local_accuracy"]

        local_acc_text = (
            f"{local_acc:.1%}"
            if not np.isnan(local_acc)
            else "N/A"
        )

        print(
            f"{row['threshold']:>10.2f}"
            f"{row['coverage']:>11.1%}"
            f"{row['fallback']:>11.1%}"
            f"{local_acc_text:>14}"
            f"{int(row['errors_captured']):>12}"
            f"{int(row['errors_left']):>12}"
            f"{row['capture_rate']:>11.1%}"
        )

    return summary


# ============================================================
# ERROS
# ============================================================

def print_errors(results):
    results = results.copy()

    errors = results[
        results["true_label"]
        != results["predicted_label"]
    ]

    print("\n" + "=" * 70)
    print("ERROS DO MODELO CALIBRADO")
    print("=" * 70)

    if errors.empty:
        print("\nNenhum erro OOF encontrado.")
        return

    for _, row in errors.sort_values(
        "confidence"
    ).iterrows():

        print(
            f"\nQuery: {row['text']}"
        )

        print(
            f"Esperado: {row['true_label']}"
        )

        print(
            f"Predito:  {row['predicted_label']}"
        )

        print(
            f"Confiança: {row['confidence']:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    texts, labels = load_training_data()

    print("=" * 70)
    print("EXPERIMENTO 06")
    print("Linear SVM + Probability Calibration + Abstention")
    print("=" * 70)

    print(
        f"\nDataset: {len(texts)} exemplos"
    )

    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    for label, count in zip(
        unique,
        counts,
    ):
        print(
            f"{label}: {count}"
        )

    print(
        "\nExecutando Nested Cross-Validation..."
    )

    results = run_nested_cv(
        texts,
        labels,
    )

    evaluate_calibration(
        results,
    )

    analyze_confidence(
        results,
    )

    evaluate_abstention(
        results,
    )

    print_errors(
        results,
    )


if __name__ == "__main__":
    main()