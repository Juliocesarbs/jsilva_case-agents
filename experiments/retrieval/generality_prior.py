import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[2]
TOOLS_PATH = ROOT / "data" / "tools_registry.json"
EVAL_PATH = ROOT / "data" / "eval_dataset.json"

ALPHAS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
N_SPLITS = 5


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_name(name):
    return name.replace("_", " ")


def token_count(name):
    return len(name.split("_"))


def reciprocal_rank(ranking, expected):
    for position, name in enumerate(ranking, start=1):
        if name == expected:
            return 1 / position
    return 0.0


def build_scores(tools, queries):
    tool_names = [tool["name"] for tool in tools]
    tool_texts = [normalize_name(name) for name in tool_names]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    tool_matrix = vectorizer.fit_transform(tool_texts)

    scores = []

    for query in queries:
        query_vector = vectorizer.transform([query])

        similarity = cosine_similarity(
            query_vector,
            tool_matrix,
        )[0]

        scores.append(similarity)

    return np.array(scores), tool_names


def rank_with_prior(
    scores,
    tool_names,
    alpha,
):
    lengths = np.array(
        [token_count(name) for name in tool_names],
        dtype=float,
    )

    adjusted_scores = scores / np.power(
        lengths,
        alpha,
    )

    indexes = np.argsort(adjusted_scores)[::-1]

    return [
        tool_names[index]
        for index in indexes
    ]


def evaluate(
    indexes,
    score_matrix,
    expected_tools,
    tool_names,
    alpha,
):
    top1 = []
    hit2 = []
    reciprocal_ranks = []

    for index in indexes:
        ranking = rank_with_prior(
            score_matrix[index],
            tool_names,
            alpha,
        )

        expected = expected_tools[index]

        top1.append(
            int(ranking[0] == expected)
        )

        hit2.append(
            int(expected in ranking[:2])
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                ranking,
                expected,
            )
        )

    return {
        "top1": np.mean(top1),
        "hit2": np.mean(hit2),
        "mrr": np.mean(reciprocal_ranks),
    }


def select_alpha(
    train_indexes,
    score_matrix,
    expected_tools,
    tool_names,
):
    candidates = []

    for alpha in ALPHAS:
        metrics = evaluate(
            train_indexes,
            score_matrix,
            expected_tools,
            tool_names,
            alpha,
        )

        candidates.append(
            (
                alpha,
                metrics,
            )
        )

    best_alpha, _ = max(
        candidates,
        key=lambda item: (
            item[1]["hit2"],
            item[1]["mrr"],
            item[1]["top1"],
            -item[0],
        ),
    )

    return best_alpha


def main():
    tools = load_json(TOOLS_PATH)
    eval_data = load_json(EVAL_PATH)

    agent_cases = [
        item
        for item in eval_data
        if item["expected_route"] == "AGENT"
        and item.get("expected_tool")
    ]

    queries = [
        item["query"]
        for item in agent_cases
    ]

    expected_tools = [
        item["expected_tool"]
        for item in agent_cases
    ]

    score_matrix, tool_names = build_scores(
        tools,
        queries,
    )

    splitter = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42,
    )

    oof_rankings = [None] * len(queries)
    selected_alphas = []

    print(f"Queries AGENT: {len(queries)}")
    print(f"Alphas candidatos: {ALPHAS}\n")

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(queries),
        start=1,
    ):
        alpha = select_alpha(
            train_idx,
            score_matrix,
            expected_tools,
            tool_names,
        )

        selected_alphas.append(alpha)

        for index in test_idx:
            oof_rankings[index] = rank_with_prior(
                score_matrix[index],
                tool_names,
                alpha,
            )

        fold_metrics = {
            "top1": np.mean(
                [
                    oof_rankings[index][0]
                    == expected_tools[index]
                    for index in test_idx
                ]
            ),
            "hit2": np.mean(
                [
                    expected_tools[index]
                    in oof_rankings[index][:2]
                    for index in test_idx
                ]
            ),
            "mrr": np.mean(
                [
                    reciprocal_rank(
                        oof_rankings[index],
                        expected_tools[index],
                    )
                    for index in test_idx
                ]
            ),
        }

        print(
            f"Fold {fold}: "
            f"alpha={alpha:.2f} | "
            f"Top1={fold_metrics['top1']:.1%} | "
            f"Hit@2={fold_metrics['hit2']:.1%} | "
            f"MRR={fold_metrics['mrr']:.3f}"
        )

    top1 = np.mean(
        [
            ranking[0] == expected
            for ranking, expected in zip(
                oof_rankings,
                expected_tools,
            )
        ]
    )

    hit2 = np.mean(
        [
            expected in ranking[:2]
            for ranking, expected in zip(
                oof_rankings,
                expected_tools,
            )
        ]
    )

    mrr = np.mean(
        [
            reciprocal_rank(ranking, expected)
            for ranking, expected in zip(
                oof_rankings,
                expected_tools,
            )
        ]
    )

    print("\nResultado OOF")
    print("-" * 35)
    print(f"Top-1: {top1:.1%}")
    print(f"Hit@2: {hit2:.1%}")
    print(f"MRR:   {mrr:.3f}")

    print("\nAlphas selecionados")
    print("-" * 35)
    print(selected_alphas)
    print(
        f"Média:   {np.mean(selected_alphas):.3f}"
    )
    print(
        f"Mediana: {np.median(selected_alphas):.3f}"
    )


if __name__ == "__main__":
    main()