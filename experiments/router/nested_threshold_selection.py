import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


RANDOM_STATE = 42

OUTER_SPLITS = 5
INNER_SPLITS = 3

THRESHOLDS = np.arange(0.50, 0.91, 0.05)

# Regra mínima desejada para uma decisão local.
MIN_AGENT_RECALL = 0.90

# Penalização dos erros.
# AGENT -> FAST_PATH é tratado como mais grave.
COST_AGENT_TO_FAST = 3.0
COST_FAST_TO_AGENT = 1.0

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAINING_DATA_PATH = (
    ROOT_DIR
    / "data"
    / "router_training_data.json"
)


def load_training_data():
    with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = np.array(
        [item["text"] for item in data]
    )

    labels = np.array(
        [item["label"] for item in data]
    )

    return texts, labels


def build_base_pipeline():
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
    return CalibratedClassifierCV(
        estimator=build_base_pipeline(),
        method="sigmoid",
        cv=INNER_SPLITS,
    )


def predict_with_confidence(model, texts):
    probabilities = model.predict_proba(texts)

    predicted_indices = np.argmax(
        probabilities,
        axis=1,
    )

    predictions = model.classes_[
        predicted_indices
    ]

    confidences = np.max(
        probabilities,
        axis=1,
    )

    return predictions, confidences


def compute_asymmetric_error_cost(
    y_true,
    y_pred,
):
    total_cost = 0.0

    for true_label, pred_label in zip(
        y_true,
        y_pred,
    ):
        if true_label == pred_label:
            continue

        if (
            true_label == "AGENT"
            and pred_label == "FAST_PATH"
        ):
            total_cost += COST_AGENT_TO_FAST

        elif (
            true_label == "FAST_PATH"
            and pred_label == "AGENT"
        ):
            total_cost += COST_FAST_TO_AGENT

    return total_cost


def evaluate_threshold(
    y_true,
    y_pred,
    confidence,
    threshold,
):
    local_mask = (
        confidence >= threshold
    )

    fallback_mask = ~local_mask

    local_count = local_mask.sum()
    fallback_count = fallback_mask.sum()

    total_count = len(y_true)

    coverage = (
        local_count / total_count
    )

    fallback_rate = (
        fallback_count / total_count
    )

    if local_count == 0:
        return {
            "threshold": threshold,
            "coverage": coverage,
            "fallback_rate": fallback_rate,
            "local_accuracy": np.nan,
            "agent_recall_local": np.nan,
            "agent_to_fast_errors": 0,
            "fast_to_agent_errors": 0,
            "error_cost": np.nan,
        }

    local_true = y_true[
        local_mask
    ]

    local_pred = y_pred[
        local_mask
    ]

    local_accuracy = accuracy_score(
        local_true,
        local_pred,
    )

    agent_present = np.any(
        local_true == "AGENT"
    )

    if agent_present:
        agent_recall = recall_score(
            local_true,
            local_pred,
            pos_label="AGENT",
        )
    else:
        agent_recall = np.nan

    agent_to_fast = np.sum(
        (local_true == "AGENT")
        & (local_pred == "FAST_PATH")
    )

    fast_to_agent = np.sum(
        (local_true == "FAST_PATH")
        & (local_pred == "AGENT")
    )

    error_cost = compute_asymmetric_error_cost(
        local_true,
        local_pred,
    )

    return {
        "threshold": threshold,
        "coverage": coverage,
        "fallback_rate": fallback_rate,
        "local_accuracy": local_accuracy,
        "agent_recall_local": agent_recall,
        "agent_to_fast_errors": int(
            agent_to_fast
        ),
        "fast_to_agent_errors": int(
            fast_to_agent
        ),
        "error_cost": error_cost,
    }


def select_threshold(
    y_true,
    y_pred,
    confidence,
):
    candidates = []

    for threshold in THRESHOLDS:
        metrics = evaluate_threshold(
            y_true,
            y_pred,
            confidence,
            threshold,
        )

        candidates.append(
            metrics
        )

    df = pd.DataFrame(
        candidates
    )

    # Primeiro tentamos encontrar thresholds
    # que respeitem o recall mínimo de AGENT.
    valid = df[
        (
            df["agent_recall_local"]
            >= MIN_AGENT_RECALL
        )
        & (
            df["coverage"] > 0
        )
    ].copy()

    if not valid.empty:
        # Critério:
        # 1. menor custo assimétrico de erro
        # 2. maior coverage
        # 3. maior accuracy local
        valid = valid.sort_values(
            by=[
                "error_cost",
                "coverage",
                "local_accuracy",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )

        selected = valid.iloc[0]

    else:
        # fallback metodológico:
        # se nenhum threshold atender recall mínimo,
        # escolhemos menor custo e depois maior coverage.
        fallback_df = df[
            df["coverage"] > 0
        ].copy()

        fallback_df = fallback_df.sort_values(
            by=[
                "error_cost",
                "coverage",
            ],
            ascending=[
                True,
                False,
            ],
        )

        selected = fallback_df.iloc[0]

    return float(
        selected["threshold"]
    ), df


def get_inner_oof_predictions(
    texts,
    labels,
):
    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_true = []
    all_pred = []
    all_confidence = []

    for train_idx, valid_idx in inner_cv.split(
        texts,
        labels,
    ):
        X_train = texts[
            train_idx
        ]

        y_train = labels[
            train_idx
        ]

        X_valid = texts[
            valid_idx
        ]

        y_valid = labels[
            valid_idx
        ]

        model = build_calibrated_model()

        model.fit(
            X_train,
            y_train,
        )

        predictions, confidences = (
            predict_with_confidence(
                model,
                X_valid,
            )
        )

        all_true.extend(
            y_valid
        )

        all_pred.extend(
            predictions
        )

        all_confidence.extend(
            confidences
        )

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_confidence),
    )


def run_nested_threshold_experiment(
    texts,
    labels,
):
    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    outer_rows = []

    all_test_true = []
    all_test_pred = []
    all_test_confidence = []
    all_test_fallback = []

    selected_thresholds = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        outer_cv.split(
            texts,
            labels,
        ),
        start=1,
    ):
        X_train = texts[
            train_idx
        ]

        y_train = labels[
            train_idx
        ]

        X_test = texts[
            test_idx
        ]

        y_test = labels[
            test_idx
        ]

        # ====================================================
        # ETAPA 1
        # threshold escolhido apenas no outer train
        # ====================================================

        (
            inner_true,
            inner_pred,
            inner_confidence,
        ) = get_inner_oof_predictions(
            X_train,
            y_train,
        )

        selected_threshold, threshold_table = (
            select_threshold(
                inner_true,
                inner_pred,
                inner_confidence,
            )
        )

        selected_thresholds.append(
            selected_threshold
        )

        # ====================================================
        # ETAPA 2
        # treina modelo final do fold
        # ====================================================

        model = build_calibrated_model()

        model.fit(
            X_train,
            y_train,
        )

        (
            test_pred,
            test_confidence,
        ) = predict_with_confidence(
            model,
            X_test,
        )

        test_metrics = evaluate_threshold(
            y_test,
            test_pred,
            test_confidence,
            selected_threshold,
        )

        fallback_mask = (
            test_confidence
            < selected_threshold
        )

        all_test_true.extend(
            y_test
        )

        all_test_pred.extend(
            test_pred
        )

        all_test_confidence.extend(
            test_confidence
        )

        all_test_fallback.extend(
            fallback_mask
        )

        outer_rows.append(
            {
                "fold": fold,
                "selected_threshold":
                    selected_threshold,
                **test_metrics,
            }
        )

        print(
            f"\nFold {fold}"
        )

        print(
            f"Threshold selecionado: "
            f"{selected_threshold:.2f}"
        )

        print(
            f"Coverage no teste: "
            f"{test_metrics['coverage']:.1%}"
        )

        print(
            f"Fallback no teste: "
            f"{test_metrics['fallback_rate']:.1%}"
        )

        if not np.isnan(
            test_metrics[
                "local_accuracy"
            ]
        ):
            print(
                f"Accuracy local: "
                f"{test_metrics['local_accuracy']:.1%}"
            )

        print(
            "AGENT -> FAST_PATH no fluxo local: "
            f"{test_metrics['agent_to_fast_errors']}"
        )

    return (
        pd.DataFrame(
            outer_rows
        ),
        np.array(
            all_test_true
        ),
        np.array(
            all_test_pred
        ),
        np.array(
            all_test_confidence
        ),
        np.array(
            all_test_fallback
        ),
        selected_thresholds,
    )


def summarize_results(
    fold_results,
    y_true,
    y_pred,
    confidence,
    fallback_mask,
    thresholds,
):
    local_mask = (
        ~fallback_mask
    )

    total = len(
        y_true
    )

    local_count = (
        local_mask.sum()
    )

    fallback_count = (
        fallback_mask.sum()
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "RESULTADO FINAL — NESTED THRESHOLD SELECTION"
    )

    print(
        "=" * 90
    )

    print(
        "\nThresholds selecionados por fold:"
    )

    print(
        [
            round(
                threshold,
                2,
            )
            for threshold
            in thresholds
        ]
    )

    print(
        f"\nThreshold médio: "
        f"{np.mean(thresholds):.3f}"
    )

    print(
        f"Mediana: "
        f"{np.median(thresholds):.3f}"
    )

    print(
        f"\nCoverage global: "
        f"{local_count / total:.1%}"
    )

    print(
        f"Fallback global: "
        f"{fallback_count / total:.1%}"
    )

    if local_count > 0:
        local_true = y_true[
            local_mask
        ]

        local_pred = y_pred[
            local_mask
        ]

        local_accuracy = accuracy_score(
            local_true,
            local_pred,
        )

        print(
            f"Accuracy local global: "
            f"{local_accuracy:.1%}"
        )

        if np.any(
            local_true == "AGENT"
        ):
            agent_recall = recall_score(
                local_true,
                local_pred,
                pos_label="AGENT",
            )

            print(
                f"Recall AGENT local: "
                f"{agent_recall:.1%}"
            )

        agent_to_fast = np.sum(
            (local_true == "AGENT")
            & (
                local_pred
                == "FAST_PATH"
            )
        )

        fast_to_agent = np.sum(
            (
                local_true
                == "FAST_PATH"
            )
            & (
                local_pred
                == "AGENT"
            )
        )

        print(
            "AGENT -> FAST_PATH locais: "
            f"{agent_to_fast}"
        )

        print(
            "FAST_PATH -> AGENT locais: "
            f"{fast_to_agent}"
        )

        print(
            "Custo assimétrico total: "
            f"{compute_asymmetric_error_cost(local_true, local_pred):.1f}"
        )

    print(
        "\nResumo por fold:"
    )

    display_columns = [
        "fold",
        "selected_threshold",
        "coverage",
        "fallback_rate",
        "local_accuracy",
        "agent_recall_local",
        "agent_to_fast_errors",
        "fast_to_agent_errors",
        "error_cost",
    ]

    print(
        fold_results[
            display_columns
        ].to_string(
            index=False
        )
    )


def main():
    texts, labels = (
        load_training_data()
    )

    print(
        "=" * 90
    )

    print(
        "EXPERIMENTO 07"
    )

    print(
        "Nested Threshold Selection"
    )

    print(
        "=" * 90
    )

    print(
        f"\nDataset: "
        f"{len(texts)} exemplos"
    )

    print(
        "\nObjetivo:"
    )

    print(
        "- escolher threshold apenas com dados internos"
    )

    print(
        "- avaliar em outer folds não utilizados "
        "na seleção"
    )

    print(
        "- penalizar AGENT -> FAST_PATH "
        "mais fortemente"
    )

    (
        fold_results,
        y_true,
        y_pred,
        confidence,
        fallback_mask,
        selected_thresholds,
    ) = run_nested_threshold_experiment(
        texts,
        labels,
    )

    summarize_results(
        fold_results,
        y_true,
        y_pred,
        confidence,
        fallback_mask,
        selected_thresholds,
    )


if __name__ == "__main__":
    main()