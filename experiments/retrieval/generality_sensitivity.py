import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
TOOLS_PATH = ROOT / "data" / "tools_registry.json"
EVAL_PATH = ROOT / "data" / "eval_dataset.json"

ALPHAS = np.arange(0.0, 1.51, 0.1)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def reciprocal_rank(ranking, expected):
    for position, name in enumerate(ranking, start=1):
        if name == expected:
            return 1 / position
    return 0.0


def evaluate(
    similarities,
    expected_tools,
    tool_names,
    token_counts,
    alpha,
):
    penalties = np.power(token_counts, alpha)

    top1_hits = []
    top2_hits = []
    reciprocal_ranks = []

    for scores, expected in zip(
        similarities,
        expected_tools,
    ):
        adjusted_scores = scores / penalties
        indexes = np.argsort(adjusted_scores)[::-1]

        ranking = [
            tool_names[index]
            for index in indexes
        ]

        top1_hits.append(
            int(ranking[0] == expected)
        )

        top2_hits.append(
            int(expected in ranking[:2])
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                ranking,
                expected,
            )
        )

    return {
        "top1": np.mean(top1_hits),
        "hit2": np.mean(top2_hits),
        "mrr": np.mean(reciprocal_ranks),
    }


def main():
    with open(
        TOOLS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        tools = json.load(file)

    with open(
        EVAL_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        eval_data = json.load(file)

    agent_cases = [
        item
        for item in eval_data
        if item["expected_route"] == "AGENT"
        and item.get("expected_tool")
    ]

    tool_names = [
        tool["name"]
        for tool in tools
    ]

    tool_texts = [
        tool["name"].replace("_", " ")
        for tool in tools
    ]

    token_counts = np.array(
        [
            len(tool["name"].split("_"))
            for tool in tools
        ],
        dtype=float,
    )

    queries = [
        item["query"]
        for item in agent_cases
    ]

    expected_tools = [
        item["expected_tool"]
        for item in agent_cases
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    tool_matrix = vectorizer.fit_transform(
        tool_texts
    )

    query_matrix = vectorizer.transform(
        queries
    )

    similarities = cosine_similarity(
        query_matrix,
        tool_matrix,
    )

    print(f"Queries AGENT: {len(agent_cases)}")

    print(
        f"\n{'ALPHA':<10}"
        f"{'TOP1':>10}"
        f"{'HIT@2':>10}"
        f"{'MRR':>10}"
    )

    print("-" * 40)

    results = []

    for alpha in ALPHAS:
        metrics = evaluate(
            similarities,
            expected_tools,
            tool_names,
            token_counts,
            alpha,
        )

        results.append(
            {
                "alpha": alpha,
                **metrics,
            }
        )

        print(
            f"{alpha:<10.1f}"
            f"{metrics['top1']:>9.1%}"
            f"{metrics['hit2']:>9.1%}"
            f"{metrics['mrr']:>10.3f}"
        )

    best = max(
        results,
        key=lambda result: (
            result["hit2"],
            result["mrr"],
            result["top1"],
        ),
    )

    print("\nMelhor ponto observado")
    print("-" * 40)
    print(f"Alpha: {best['alpha']:.1f}")
    print(f"Top-1: {best['top1']:.1%}")
    print(f"Hit@2: {best['hit2']:.1%}")
    print(f"MRR:   {best['mrr']:.3f}")


if __name__ == "__main__":
    main()